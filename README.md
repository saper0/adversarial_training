# Adversarial Training for Graph Neural Networks: Pitfalls, Solutions, and New Directions

This codebase has been used to generate the results found in the NeurIPS 2023 paper *Adversarial Training for Graph Neural Networks: Pitfalls, Solutions, and New Directions*.

# Installation

The code requires the following packages and has been tested with the given versions:

```
python 3.9.7
pytorch 1.10.2
cudatoolkit 11.3.1
torchvision 0.11.3
pyg 2.0.3
sacred 0.8.2
tqdm 4.62.3
scipy 1.7.3
torchtyping 0.1.4
seml 0.3.6
numba 0.54.1
filelock 3.4.2
numpy 1.20.3
scikit-learn 1.0.2
tinydb 4.6.1
tinydb-serialization 2.1.0
tqdm 4.62.3
ogb 1.3.2
torchtyping 0.1.4
cvxpy 1.2.1
```

For the experiments done using `Soft-Median` or `Randomized Smoothing`, additional packages are necessary and explained below, in their respective sections.

## Main Package 

Thereafter one can install the actual module via (alternatively use `python install .`):
```bash
pip install .
```

# Structure

Besides the standard python artifacts we provide:

- `cache`: for the pretrained models / attacked adjacency matrices
- `config`: the configuration files grouped by experiments
- `data`: for storing the datasets
- `experiments`: source code defining the types of experiments
- `kernels`: the custom kernel package (only relevant for soft-median)
- `log`: for logging the output of experiments
- `robust_diffusion`: the source code

# Experiments

All experiments use [seml](https://github.com/TUM-DAML/seml). For an introduction into *seml*, we refer to the [official examples](https://github.com/TUM-DAML/seml/tree/master/examples) on the *seml* github repository. 

By default all the results of the experiments will be logged into `./log`.

## Adversarial Training

### Inductive Case

The experiment code can be found in `experiments/experiment_adv_train_inductive.py`. The corresponding *seml* experiment configuration files can be found in [config/train](config/train/) under the name of `adv_ind_lrbcd.yaml` for LR-BCD and `adv_ind_prbcd.yaml` for PR-BCD.

Exemplary, adversarial training with the LR-BCD configuration file can be performed by:

```
seml [mongodb-collection-name] add config/train/adv_ind_lrbcd.yaml start
```

Optionally, the experiments can be run locally by adding the `--local` flag:

```
seml [mongodb-collection-name] add config/train/adv_ind_lrbcd.yaml start --local
```

### Transductive Case

The experiment code for adversarial training with PGD in the transductive case can be found in `experiments/experiment_adv_train_transductive_pgd.py`. The respectective experiment configuration files can be found in [config/train](config/train/) under `adv_trans_pgd.yaml` for moderate adversaries and for `adv_trans_pgd_perfect_rob.yaml` for a strong adversary, where GPRGNN achieves perfect robustness.

## Robustness Evaluation

The experiment code for evaluating the robustness of trained models can be found in `experiments/experiment_global_attack_direct.py`. The corresponding experiment configuration files are in `config/attack_evasion_global_direct` and prefixed `lrbcd_` for LR-BCD adversarial training, `prbcd_` for PR-BCD adversarial training, and `pgd_` for (transductive) PGD adversarial training.

For evaluation, we use the locally stored models in the `cache` folder (unless specified differently).

## Certifiable Robustness

Requires the following additional packages:

```
https://github.com/abojchevski/sparse_smoothing.git
gmpy2
statsmodels
```

The experiment code for certifying accuracy using sparse smoothing can be found in `experiments/experiment_sparse_smoothing.py` and the relevant configuration files in `config/sparse_smoothing`.  

## Soft Median

For using the soft median defense, custom cuda kernels found in `kernels` have to be installed. For the instructions, we refer to [Robustness of Graph Neural Networks at Scale](https://github.com/sigeisler/robustness_of_gnns_at_scale). The corresponding experiment config file is `config/train/ind_soft_median.yaml`.

# Cite

Please cite our paper if you use this code in your own work:

```
@inproceedings{
    gosch2023adversarial,
    title={Adversarial Training for Graph Neural Networks: Pitfalls, Solutions, and New Directions},
    author={Lukas Gosch and Simon Geisler and Daniel Sturm and Bertrand Charpentier and Daniel Z{\"u}gner and Stephan G{\"u}nnemann},
    booktitle={Thirty-seventh Conference on Neural Information Processing Systems (NeurIPS)},
    year={2023},
 }
```

# Contact

For questions and feedback, please do not hesitate to contact:

Lukas Gosch, l (dot) gosch (at) tum (dot) de, Technical University of Munich
Simon Geisler, s (dot) geisler (at) tum (dot) de, Technical University of Munich

# Other Notes and Credit

This codebase has been developed out of a fork of 

- [Robustness of Graph Neural Networks at Scale](https://github.com/sigeisler/robustness_of_gnns_at_scale)

and adapted for this project. As a result, there is a significant overlap of code. We thank the authors for making their code public and the development team of *PyTorch Geometric* as well as *seml*.

