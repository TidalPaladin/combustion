# Combustion

[![CircleCI](https://circleci.com/gh/TidalPaladin/combustion/tree/master.svg?style=svg)](https://circleci.com/gh/TidalPaladin/combustion/tree/master)
[![Documentation Status](https://readthedocs.org/projects/combustion/badge/?version=latest)](https://combustion.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/TidalPaladin/combustion/branch/master/graph/badge.svg)](https://codecov.io/gh/TidalPaladin/combustion)

Combustion is a collection of layers/models/helper functions for deep learning.

Combustion is designed with PyTorch in mind, with some emphasis on the following 
3rd party libraries:
* PyTorch-Lightning, a high level API for model training
* Hydra, a library that enables YAML based configuration of hyperparameters

## Installation

Install combustion core with pip using

```
pip install .
```

Optional dependencies are provided via the following extras:
  * `hdf5` - HDF5 serialization (`combustion.data`)
  * `vision` - Vision helpers (`combustion.vision`)
  * `points` - Point cloud manipulation (`combustion.points`)
  * `macs` - For counting multiply accumulate operations

Development dependencies can be installed with

```
pip install combustion[dev] && pip install combustion[points]
```

## Development

Multiple make recipes are provided to aid in development:
* `ci-test` - Runs tests that will be run by CircleCI
* `ci-quality` - Runs quality checks that will be run by CircleCI
* `docs` - Builds documentation
* `pre-commit` - Installs a pre-commit hook to run code quality tests
* `quality` - Runs code quality tests
* `style` - Automatically formats code (using `black` and `autopep8`)
* `tag-version` - Adds a git tag based on `version.txt`.
* `test` - Runs all tests, including slow model tests
* `test-%` - Runs specific tests by pattern match (via `pytest -k` flag)
* `test-pdb-%` - Runs specific tests with debugging on failure (via `pytest --pdb` flag)

## References
* [PyTorch Lightning](https://github.com/PytorchLightning/pytorch-lightning)
* [Hydra](https://github.com/facebookresearch/hydra)
