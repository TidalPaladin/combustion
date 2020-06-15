# Combustion

[![CircleCI](https://circleci.com/gh/TidalPaladin/combustion/tree/master.svg?style=svg)](https://circleci.com/gh/TidalPaladin/combustion/tree/master)
[![Documentation Status](https://readthedocs.org/projects/combustion/badge/?version=latest)](https://combustion.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/TidalPaladin/combustion/branch/master/graph/badge.svg)](https://codecov.io/gh/TidalPaladin/combustion)

Combustion is a collection of layers/models/helper functions that I
have found useful or incorporated into a deep learning project at
some point.

Combustion is designed with PyTorch in mind, with some emphasis on the following 
3rd party libraries:
* PyTorch-Lightning, a high level API for model training
* Hydra, a library that enables YAML based configuration of hyperparameters

## Installation

Install combustion with pip using

```
pip install .
```

## Documentation

TODO build docs and add link.


## Development

Multiple make recipes are provided to aid in development:
* `pre-commit` - Installs a pre-commit hook to run code quality tests
* `quality` - Runs code quality tests
* `style` - Automatically formats code (using `black` and `autopep8`)
* `test` - Runs all tests
* `test-%` - Runs specific tests by pattern match (via `pytest -k` flag)
* `test-pdb-%` - Runs specific tests with debugging on failure (via `pytest --pdb` flag)


## References
* [PyTorch Lightning](https://github.com/PytorchLightning/pytorch-lightning)
* [Hydra](https://github.com/facebookresearch/hydra)
