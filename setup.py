import logging
import subprocess

from setuptools import setup, find_packages

import torch

if torch.cuda.is_available():
    cuda_v = f"cu{torch.version.cuda.replace('.', '')}"
else:
    cuda_v = "cpu"

torch_v = torch.__version__.split('.')
torch_v = '.'.join(torch_v[:-1] + ['0'])


def system(command: str):
    output = subprocess.check_output(command, shell=True)
    logging.info(output)


system(f'pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-{torch_v}+{cuda_v}.html')
system(f'pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-{torch_v}+{cuda_v}.html')
system(f'pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-{torch_v}+{cuda_v}.html')
system(f'pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-{torch_v}+{cuda_v}.html')

install_requires = [
    'filelock',
    'numba',
    'numpy',
    'pandas',
    'sacred',
    'scikit-learn',
    'scipy',
    'seaborn',
    'tabulate',
    'tinydb',
    'tinydb-serialization',
    'tqdm',
    'ogb',
    'torchtyping',
    'torch-geometric'
]

setup(
    name='robust_diffusion',
    version='1.0.0',
    author='',
    description='Implementation & experiments for the paper "Adversarial Training of Graph Neural Networks"',
    url='https://github.com/sigeisler/robustness_of_gnns_at_scale',
    packages=['robust_diffusion'] + find_packages(),
    install_requires=install_requires,
    zip_safe=False,
    package_data={'robust_diffusion': ['kernels/csrc/custom.cpp', 'kernels/csrc/custom_kernel.cu']},
    include_package_data=True
)
