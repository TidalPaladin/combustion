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


import os

from setuptools import find_packages, setup


TORCH = "torch>=1.6.0,<=2.0.0"

extras = {}

extras["testing"] = ["pytest", "pytest-mock", "pytest-cov", "pytest-xdist"]
extras["macs"] = ["thop"]
extras["hdf5"] = ["h5py<3.0"]
extras["vision"] = ["kornia", "opencv-python", "torchvision", "Pillow"]
extras["points"] = ["torch-scatter<2.0.5"]

extras["dev"] = extras["testing"] + extras["macs"] + extras["vision"] + ["scipy", "pandas"]


def write_version_info():
    # get version
    cwd = os.getcwd()
    version = open("version.txt", "r").read().strip()

    sha = "Unknown"
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd).decode("ascii").strip()
    except Exception:
        pass

    if os.getenv("COMBUSTION_BUILD_VERSION"):
        version = os.getenv("COMBUSTION_BUILD_VERSION")
    elif sha != "Unknown":
        version += "+" + sha[:7]

    version_path = os.path.join(cwd, "src", "combustion", "version.py")
    with open(version_path, "w") as f:
        f.write("__version__ = '{}'\n".format(version))
        f.write("git_version = {}\n".format(repr(sha)))

    return version


def install(version):
    setup(
        name="combustion",
        version=version,
        author="Scott Chase Waggener",
        author_email="tidalpaladin@gmail.com",
        description="Helpers for PyTorch model training/testing",
        keywords="deep learning pytorch",
        license="Apache",
        url="https://github.com/TidalPaladin/combustion",
        package_dir={"": "src"},
        packages=find_packages("src"),
        install_requires=[
            "decorator",
            "matplotlib",
            "numpy",
            "pytorch-lightning>=1.0.0",
            "torchmetrics>=0.3",
            "hydra-core>=1.0.0",
            TORCH,
            "packaging",
            "pynvml",
        ],
        extras_require=extras,
        python_requires=">=3.8.0",
        classifiers=[
            "Intended Audience :: Developers",
            "Intended Audience :: Education",
            "Intended Audience :: Science/Research",
            "Operating System :: OS Independent",
            "Programming Language :: Python :: 3.7",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
    )


if __name__ == "__main__":
    version = write_version_info()
    install(version)
