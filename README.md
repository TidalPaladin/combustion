# Combustion

Combustion is a partial library and template repository for creating 
deep learning projects. The structure of the repository aims to keep
utilities and helper functions separate from the overall template,
allowing for easy reuse of low level features while still allowing
for an easy transition between projects in different domains.


## Overview of Features

The core components of Combustion are as follows:
* Low level utilities (e.g. bottlenecked convolutions) that make up the
  library portion of Combustion
* Dockerfiles that enable easy containerization of a newly developed 
  application
* CI/CD/devops features (provided via a Makefile) like code formatting,
  test running, etc.

Combustion is designed with following 3rd party libraries in mind:
* PyTorch-Lightning, a high level API for model training
* Hydra, a library that enables YAML based configuration of hyperparameters


## Installation

First create a virtual environment to store dependencies.

```
make venv
```

This make recipe can also be used to activate the virtual environment after install.

TODO docker build instructions here

## Usage

A project template is provided in `src/project`. The existing `__main__.py` file
demonstrates a basic training / testing example using PyTorch-Lightning and Hydra.
Modify the `project` directory as needed.

### Configuration

Hydra allows for hyperparameters and runtime properties to be easily configured
using YAML files. The modularity of using multiple YAML files has many advantages
over command line configuration. See the Hydra 
[documentation](https://github.com/facebookresearch/hydra) for more details. 

Some noteable features of Hydra to be aware of are:
* Composition of multiple YAML files when specifying a runtime configuration
* Ability to specify YAML values at the command line
* Ability to specify YAML values at the command line


## Development

Multiple make recipes are provided to aid in development:
* `pre-commit` - Installs a pre-commit hook to run code quality tests
* `quality` - Runs code quality tests
* `style` - Automatically formats code (using `black` and `autopep8`)
* `test` - Runs all tests
* `test-%` - Runs specific tests by pattern match (via `pytest -k` flag)
* `test-pdb-%` - Runs specific tests with debugging on failure (via `pytest --pdb` flag)

## To Do
* CircleCI CI/CD pipeline
* Improve documentation for Combustion modules
* Add / fix tests as needed


## References
* [PyTorch Lightning](https://github.com/PytorchLightning/pytorch-lightning)
* [Hydra](https://github.com/facebookresearch/hydra)
