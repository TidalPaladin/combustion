#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple check list from AllenNLP repo: https://github.com/allenai/allennlp/blob/master/setup.py
To create the package for pypi.
1. Change the version in __init__.py, setup.py as well as docs/source/conf.py.
2. Commit these changes with the message: "Release: VERSION"
3. Add a tag in git to mark the release: "git tag VERSION -m'Adds tag VERSION for pypi' "
   Push the tag to git: git push --tags origin master
4. Build both the sources and the wheel. Do not change anything in setup.py between
   creating the wheel and the source distribution (obviously).
   For the wheel, run: "python setup.py bdist_wheel" in the top level directory.
   (this will build a wheel for the python version you use to build it).
   For the sources, run: "python setup.py sdist"
   You should now have a /dist directory with both .whl and .tar.gz source versions.
5. Check that everything looks correct by uploading the package to the pypi test server:
   twine upload dist/* -r pypitest
   (pypi suggest using twine as other methods upload files via plaintext.)
   You may have to specify the repository url, use the following command then:
   twine upload dist/* -r pypitest --repository-url=https://test.pypi.org/legacy/
   Check that you can install it in a virtualenv by running:
   pip install -i https://testpypi.python.org/pypi combustion
6. Upload the final version to actual pypi:
   twine upload dist/* -r pypi
7. Copy the release notes from RELEASE.md to the tag in github once everything is looking hunky-dory.
8. Update the documentation commit in .circleci/deploy.sh for the accurate documentation to be displayed
9. Update README.md to redirect to correct documentation.
"""


from setuptools import find_packages, setup


extras = {}

extras["sklearn"] = ["scikit-learn"]

extras["testing"] = ["pytest", "pytest-mock", "pytest-cov", "pytest-xdist"]
extras["docs"] = [
    "recommonmark",
    "sphinx",
    "sphinx-markdown-tables",
    "sphinx-rtd-theme",
]
extras["quality"] = [
    "black",
    "isort",
    "flake8",
    "pre-commit",
]

extras["dev"] = extras["testing"] + extras["quality"] + ["scikit-learn", "torch"]

setup(
    name="combustion",
    version="1.0.0",
    author="Scott Chase Waggener",
    author_email="tidalpaladin@gmail.com",
    description="Helpers for PyTorch model training/testing",
    keywords="deep learning pytorch",
    license="Apache",
    url="https://github.com/TidalPaladin/combustion",
    package_dir={"": "src"},
    packages=find_packages("src"),
    install_requires=[
        # helper for defining decorators
        "decorator",
        "matplotlib",
        "numpy",
        # progress bars in model download and training scripts
        # "tqdm >= 4.27",
        # high level training api
        "pytorch-lightning",
        # hparams
        "hydra-core",
        "torchvision",
        # visualizing models
        # "torchviz",
        "torch",
        "scipy",
        "pynvml",
        "kornia",
        # "sklearn",
        "h5py",
        "Pillow-SIMD",
        # "pytorch-model-summary",
    ],
    extras_require=extras,
    python_requires=">=3.7.0",
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
