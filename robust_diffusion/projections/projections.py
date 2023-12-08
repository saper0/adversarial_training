import torch 
import time
from scipy import sparse as sp
import torch
import cvxpy as cp
import numpy as np
from typing import Any, Optional
import time
from numba import jit
import logging

class BaseProjection():
    def __init__(self, local_budget, global_budget) -> None:
        self.global_budget = global_budget
        self.local_budget = local_budget
        self.n = len(local_budget)
        return

    def get_local_violated_node_idx(self, cost, edge_index):
        """get idx of all nodes which violate local constraints

        Parameters
        ----------
        cost
            tensor of cost per edge
        edge_index
            tensor of edge idx (from_node, to_node) for cost 

        Returns
        ----------
        idx of nodes which violate local budget
            tensor 

        """
        R = torch.sparse_coo_tensor(edge_index, cost, (self.n, self.n))
        # sum over dim 1 to get violations
        taken_per_node_row = torch.sparse.sum(R, dim=1)
        taken_per_node_col = torch.sparse.sum(R, dim=0)
        taken_per_node = (taken_per_node_row + taken_per_node_col).coalesce()
        mask = taken_per_node.values() > self.local_budget[taken_per_node.indices()]+1e-4
        idx_violated = taken_per_node.indices()[mask]
        return idx_violated, taken_per_node

    def check_constraints(self, values, edge_index, return_stats = False):
        """check if any constraints are violated

        Parameters
        ----------
        values
            tensor of cost per edge
        edge_index
            tensor of edge idx (from_node, to_node) for cost 

        Returns
        ----------
        bool whether any constraints are violated

        """
        # values in [0,1]
        smaller_1 = (values > 1).sum() == 0
        larger_0 =  (values < 0).sum() == 0
        # local fulfilled
        local_violations, _ = self.get_local_violated_node_idx(values, edge_index)
        no_locals = len(local_violations) == 0
        # global fulfilled
        no_globals = values.sum() <= self.global_budget
        if not return_stats:
            return no_globals & no_locals & smaller_1 & larger_0
        else:
            return {'global': no_globals, 'gloval_val': (values.sum(), self.global_budget), 'local': no_locals, 'smaller_1': smaller_1, 'larger_0': larger_0}

    def get_local_violated_edge_idx(self, node_idx_violated, edge_index):
        mapping = torch.zeros(self.n).bool()
        mapping[node_idx_violated]=True # True at pos of violated nodes
        mask_col = mapping[edge_index[0]] # True if col in violated nodes
        mask_row = mapping[edge_index[1]] # True if row in violated nodes
        mask_row_or_col = torch.logical_or(mask_col, mask_row) # True if either row or col in violated nodes
        mask_row_and_col = torch.logical_and(mask_col, mask_row) # True if both row and col in violated nodes
        return mask_row_or_col, mask_row_and_col

    def get_connected_edges(self, node_idx, edge_index):
        mapping = torch.zeros(self.n).bool()
        mapping[node_idx]=True # True at pos of node
        mask_col = mapping[edge_index[0]]
        mask_row = mapping[edge_index[1]]
        mask_row_or_col = torch.logical_or(mask_col, mask_row)
        return mask_row_or_col.nonzero().squeeze()

    def greedy_global_budget(self, values, cost, eps=1e-2):
        ''' assumes cost sorted by values'''
        cost_cum = torch.cumsum(cost,0)
        values[cost_cum>=self.global_budget-eps]=0
        return values


class GreedyKnapsackProjection(BaseProjection):
    def __init__(self, local_budget, global_budget) -> None:
        super().__init__(local_budget, global_budget)
        return

    def select(self, values, edge_index):
        cost = torch.ones_like(values)
        values, edge_index =  self.greedy_knapsack(values, cost, edge_index)
        assert self.check_constraints((values>0).int(), edge_index)
        return (values>0).int()

    def project(self, values, edge_index):
        cost = values.clone()
        values, edge_index =  self.greedy_knapsack(values, cost, edge_index)
        if not self.check_constraints(values, edge_index):
            logging.info(f'constraints are: {self.check_constraints(values, edge_index, True)}')
        return values

    def greedy_knapsack(self, values, cost, edge_index):
        # sort everything by value descending
        order = torch.argsort(-values)
        values = values[order]
        cost = cost[order]
        edge_index = edge_index[:,order]

        # clip cost and values
        cost = torch.clamp(cost, 0, 1)
        values = torch.clamp(values, 0, 1)

        # get local violations
        node_idx, _ = self.get_local_violated_node_idx(cost, edge_index)
        edge_mask_or, edge_mask_and = self.get_local_violated_edge_idx(node_idx, edge_index)
        edge_idx_or = torch.arange(len(cost))[edge_mask_or] # idx of cost corresponding to violated edges

        # enforce local budget
        if edge_idx_or.shape[0] > 0:
            values = self.greedy_local_budget_numba_wrapper(values, cost, edge_index, edge_idx_or)
            values = torch.tensor(values).to(0)

        # values changed due to local budget -> resort
        #order = torch.argsort(-values)
        #values = values[order]
        #cost = cost[order]
        #edge_index = edge_index[:,order]

        # global budget
        cost[values==0]=0
        if cost.sum() > self.global_budget:
           # values = NodeL2Projection.project_L2(budget=self.global_budget, values=values, eps= 1e-5, inplace= False)
           values = self.greedy_global_budget(values, cost)

        # print(f'mass after global: {values.sum()}')
        # print(f'sort: {sort-start}')
        # print(f'find local: {find_local-sort}')
        # print(f'local: {enforce_local-find_local}')
        # print(f'global: {end-enforce_local}')

        # resort values, edge index to the order they were recieved
        order_backward = torch.argsort(order)
        values = values[order_backward]
        edge_index = edge_index[:, order_backward]

        return values, edge_index
    '''
    def greedy_local_budget_no_jit(self, values, cost, edge_index, edge_idx_violated):
        row, col = edge_index[0], edge_index[1]
        cumsum = torch.cumsum(cost, 0) # for every idx the cost if every edge is taken
        local_budget = self.local_budget.clone() - 1e-5 # subtraction for stability
        local_budget[local_budget<0]=0
        

        for idx in edge_idx_violated:
            if (cumsum[idx]-cost[idx]) - self.global_budget >= 0:   # global budget is 0
                #print(f'global budget reached with edge {idx}')
                return values
            elif cumsum[idx] - self.global_budget > 0: # global budgest is not enough
                values[idx] = self.global_budget - (cumsum[idx]-values[idx])

            n_col = col[idx]
            n_row = row[idx]
            edge_cost = cost[idx]

            if (local_budget[n_col] == 0) or (local_budget[n_row] == 0): # case budget 0
                values[idx] = 0
                cumsum[idx:] -= edge_cost # free global budget
            elif (local_budget[n_col] < edge_cost) or (local_budget[n_row] < edge_cost): # case only part of the cost available
                min_budget = min(local_budget[n_col], local_budget[n_row]).clone()
                local_budget[n_row] -= min_budget
                local_budget[n_col] -= min_budget
                values[idx] = min_budget
                cumsum[idx:] -= (edge_cost - min_budget) # free global budget
            else: # case budget free
                local_budget[n_col] -= edge_cost
                local_budget[n_row] -= edge_cost

        return values
        '''

    def greedy_local_budget_numba_wrapper(self, values, cost, edge_index, edge_idx_violated):
        row, col = edge_index[0].cpu().numpy(), edge_index[1].cpu().numpy()
        local_budget = self.local_budget.clone().cpu().numpy()
        values = values.cpu().numpy()
        edge_idx_violated = edge_idx_violated.cpu().numpy()
        cost = cost.cpu().numpy()
        global_budget = self.global_budget
        values = self.greedy_local_budget(values, cost, row, col, edge_idx_violated, local_budget, global_budget)
        return values


    @staticmethod
    @jit(nopython=True, cache=True)
    def greedy_local_budget(values, cost, row, col, edge_idx_violated, local_budget, global_budget):
        '''iterates thorough (sorted!) values and greedily enforces local budget'''

        for i, idx in enumerate(edge_idx_violated):
            # set variables
            if i >0:
                last_idx = edge_idx_violated[i-1]
            else:
                last_idx = -1
            n_col = col[idx]
            n_row = row[idx]
            edge_cost = cost[idx]

            # global budget as 
            global_budget -= cost[last_idx+1:idx].sum()
            if global_budget < edge_cost:
                #TODO improve here
                values[idx:] = 0
                return values
            
            # local budget
            if (local_budget[n_row] < edge_cost) or (local_budget[n_col] < edge_cost):
                values[idx] = 0
            else:
                local_budget[n_row] -= edge_cost
                local_budget[n_col] -= edge_cost
                global_budget -= edge_cost

        return values



class NodeL2Projection(BaseProjection):
    def __init__(self, local_budget, global_budget) -> None:
        super().__init__(local_budget, global_budget)
        #self.local_budget = torch.ones_like(local_budget)
        return

    def project(self, values, edge_index):

        # sort idx by budget_excess/node_degree
        R = torch.sparse_coo_tensor(edge_index, values, (self.n, self.n))
        # sum over dim 1 to get violations
        taken_per_node_row = torch.sparse.sum(R, dim=1)
        taken_per_node_col = torch.sparse.sum(R, dim=0)
        taken_per_node = (taken_per_node_row + taken_per_node_col).coalesce()
        
        val = (taken_per_node.values() - self.local_budget[taken_per_node.indices().squeeze()])/self.local_budget[taken_per_node.indices().squeeze()]
        violated_idx = (val>0).nonzero()


        if len(violated_idx)>0:
            violated_idx_orig= taken_per_node.indices().squeeze()[violated_idx]
            order = torch.argsort(val[violated_idx].squeeze())
            node_idx = violated_idx_orig[order]
            
            for idx in node_idx:
                edges_idx = self.get_connected_edges(idx, edge_index)
                values[edges_idx] = self.project_L2(budget=self.local_budget[idx], values=values[edges_idx], eps= 1e-2, inplace= False)

        # global  budget
        values = self.project_L2(budget=self.global_budget, values=values, eps= 1e-5, inplace= False)

        return values

    def select(self, values, edge_index):
        ''' greedy heuristic as default selection (like topk for PR-BCD)'''
        greedy = GreedyKnapsackProjection(self.local_budget, self.global_budget)
        return greedy.select(values, edge_index)
    

    @staticmethod
    def project_L2(budget: int, values: torch.Tensor, eps: float = 0, inplace: bool = False):
        if not inplace:
            values = values.clone()

        if torch.clamp(values, 0, 1).sum() > budget:
            left = (values - 1).min()
            right = values.max()
            miu = NodeL2Projection.bisection(values, left, right, budget)
            values.data.copy_(torch.clamp(
                values - miu, min=eps, max=1 - eps
            ))
        else:
            values.data.copy_(torch.clamp(
                values, min=eps, max=1 - eps
            ))
        return values

    @staticmethod
    def bisection(edge_weights, a, b, n_perturbations, epsilon=1e-5, iter_max=1e5):
        def func(x):
            return torch.clamp(edge_weights - x, 0, 1).sum() - n_perturbations

        miu = a
        for i in range(int(iter_max)):
            miu = (a + b) / 2
            # Check if middle point is root
            if (func(miu) == 0.0):
                break
            # Decide the side to repeat the steps
            if (func(miu) * func(a) < 0):
                b = miu
            else:
                a = miu
            if ((b - a) <= epsilon):
                break
        return miu


class ExactL2Projection(BaseProjection):
    pass

class ExactKnapsackProjection(BaseProjection):
    pass

class ExactProjection(BaseProjection):
    def __init__(self, edge_index, edge_weight, local_budget, global_budget, objective: str = 'knapsack', p: Any = 2) -> None:
        super().__init__(edge_index, edge_weight, local_budget, global_budget)
        assert objective in ['knapsack', 'LPproj']
        self.objective = objective
        self.p = p
        return

    def project(self, values, edge_index, proj_to = (0,1), cost_strategy = 'prob_mass'):

        row_idx = edge_index[0]
        col_idx = edge_index[1]
        local_budget = self.local_budget.clone()
        global_budget = self.global_budget
        
        m = len(values)
        n = len(local_budget)

        assert self.p == 'inf' or self.p >= 1, 'p must be "inf" or >= 1'

        if cost_strategy == 'prob_mass':
            cost = torch.clamp(values, 0, 1)
            x = cp.Variable(m)
            constraints = [proj_to[0] <= x, x <= proj_to[1], x <= cost.cpu(), sum(x) <= global_budget]
        elif cost_strategy == 'constant_1':
            cost = torch.ones_like(values)
            x = cp.Variable(m, boolean=True)
            constraints = [sum(x) <= global_budget]
        else:
            assert False

        # TODO implemenmt local and global check
        # get local violations
        node_idx = self.get_local_violated_node_idx(cost, edge_index)
        edge_mask_or, edge_mask_and = ExactProjection.get_local_violated_edge_idx(node_idx, edge_index, len(self.local_budget))
        m_hat = edge_mask_or.sum().int()
        #edge_idx_or = torch.arange(len(cost))[edge_mask_or]

        # define objective
        if self.objective == 'knapsack':
            objective = cp.Maximize(x.T @ values.cpu())
        elif self.objective == 'LPproj':
            if cost_strategy == 'prob_mass':
                objective = cp.Minimize(cp.norm(x - values.cpu(), self.p))
            elif cost_strategy == 'constant_1':
                objective = cp.Maximize(x.T @ values.cpu())
                #objective = cp.Minimize(cp.norm(x*values.cpu() - values.cpu(), self.p)) # x is binary here -> adjust cost 
        else:
            assert False

        # add local constraints (if violated)
        if local_budget is not None and row_idx is not None and col_idx is not None and m_hat>0:
            D = (sp.coo_matrix((cost[edge_mask_or].cpu(), (row_idx[edge_mask_or].cpu(), np.arange(m_hat))), shape=(n, m_hat))
                 + sp.coo_matrix((cost[edge_mask_or].cpu(), (col_idx[edge_mask_or].cpu(), np.arange(m_hat))), shape=(n, m_hat)))
            constraints += [D @ x[edge_mask_or] <= local_budget.cpu()]

        prob = cp.Problem(objective, constraints)

        prob.solve()

        return torch.tensor(x.value, device=values.device).float(), edge_index