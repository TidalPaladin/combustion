.PHONY: docs docker docker-dev clean clean-venv check ci-test pre-commit quality run style tag-version test venv upload upload-test

PY_VER=python3.8
PY_VER_SHORT=py$(shell echo $(PY_VER) | sed 's/[^0-9]*//g')
QUALITY_DIRS=src tests setup.py
CLEAN_DIRS=src tests
VENV=$(shell pwd)/venv
PYTHON=$(VENV)/bin/python

SPHINXBUILD=$(VENV)/bin/sphinx-build
#SPHINXOPTS=-W
export SPHINXBUILD
export SPHINXOPTS


LINE_LEN=120
DOC_LEN=120


check: 
	$(MAKE) style
	$(MAKE) docs
	$(MAKE) test


clean: 
	find $(CLEAN_DIRS) -path '*/__pycache__/*' -delete
	find $(CLEAN_DIRS) -type d -name '__pycache__' -empty -delete
	find $(CLEAN_DIRS) -name '*@neomake*' -type f -delete
	find $(CLEAN_DIRS) -name '*.pyc' -type f -delete
	find $(CLEAN_DIRS) -name '*,cover' -type f -delete
	cd docs && make clean
	rm -rf docs/api docs/generated
	rm -rf dist

clean-venv:
	rm -rf $(VENV)

quality: $(VENV)/bin/activate-quality
	$(PYTHON) -m black --check --line-length $(LINE_LEN) --target-version $(PY_VER_SHORT) $(QUALITY_DIRS)
	$(PYTHON) -m flake8 --max-doc-length $(DOC_LEN) --max-line-length $(LINE_LEN) $(QUALITY_DIRS) 

run: $(VENV)/bin/activate
	$(PYTHON) -m efficientdet_fcos

style: $(VENV)/bin/activate-quality
	$(PYTHON) -m autoflake -r -i --remove-all-unused-imports --remove-unused-variables $(QUALITY_DIRS)
	$(PYTHON) -m isort $(QUALITY_DIRS)
	$(PYTHON) -m autopep8 -a -r -i --max-line-length=$(LINE_LEN) $(QUALITY_DIRS)
	$(PYTHON) -m black --line-length $(LINE_LEN) --target-version $(PY_VER_SHORT) $(QUALITY_DIRS)

venv: $(VENV)/bin/activate

$(VENV)/bin/activate: requirements.txt ../../src/combustion
	test -d $(VENV) || $(PY_VER) -m venv $(VENV)
	$(PYTHON) -m pip install -U pip 
	$(PYTHON) -m pip install -e ../..[vision]
	$(PYTHON) -m pip install -r requirements.txt
	touch $(VENV)/bin/activate

$(VENV)/bin/activate-%: requirements.%.txt
	test -d $(VENV) || $(PY_VER) -m venv $(VENV)
	$(PYTHON) -m pip install -U pip 
	$(PYTHON) -m pip install -r $<
	touch $(VENV)/bin/activate-$*
