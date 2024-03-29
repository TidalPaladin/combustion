.PHONY: docs docker clean clean-venv check ci-test pre-commit quality run style tag-version test venv upload upload-test

PROJECT=combustion
PY_VER=python3.9
PY_VER_SHORT=py$(shell echo $(PY_VER) | sed 's/[^0-9]*//g')
QUALITY_DIRS=combustion tests setup.py
CLEAN_DIRS=combustion tests
VENV=$(shell pwd)/venv
PYTHON=$(VENV)/bin/python

SPHINXBUILD=$(VENV)/bin/sphinx-build
export SPHINXBUILD
export SPHINXOPTS

LINE_LEN=120
DOC_LEN=120

VERSION := $(shell cat version.txt)

check: 
	$(MAKE) style
	$(MAKE) quality
	$(MAKE) types
	$(MAKE) test

ci-test: $(VENV)/bin/activate-test
	$(PYTHON) -m pytest \
		-rs \
		--cov=./combustion \
		--cov-report=xml \
		--cov-report=term \
		-s -v \
		-m "not ci_skip" \
		./tests/

docs: $(VENV)/bin/activate-docs
	cd docs && make html 

docker: 
	docker build \
		--target release \
		-t $(PROJECT):latest \
		--file ./docker/Dockerfile \
		./

clean: 
	find $(CLEAN_DIRS) -path '*/__pycache__/*' -delete
	find $(CLEAN_DIRS) -type d -name '__pycache__' -empty -delete
	find $(CLEAN_DIRS) -name '*@neomake*' -type f -delete
	find $(CLEAN_DIRS) -name '*.pyc' -type f -delete
	find $(CLEAN_DIRS) -name '*,cover' -type f -delete
	rm -rf docs/api docs/generated
	rm -rf dist

clean-venv:
	rm -rf $(VENV)

node_modules:
	npm install

package: venv
	rm -rf dist
	$(PYTHON) -m pip install --upgrade setuptools wheel
	export COMBUSTION_BUILD_VERSION=$(VERSION) && $(PYTHON) setup.py sdist bdist_wheel

pre-commit: 
	pre-commit install

quality: $(VENV)/bin/activate-quality
	$(MAKE) clean
	$(PYTHON) -m black --check --line-length $(LINE_LEN) --target-version $(PY_VER_SHORT) $(QUALITY_DIRS)
	$(PYTHON) -m flake8 --max-doc-length $(DOC_LEN) --max-line-length $(LINE_LEN) $(QUALITY_DIRS) 

DATA_PATH=$(shell pwd)/examples/basic/data
CONF_PATH=$(shell pwd)/examples/basic/conf
OUTPUT_PATH=$(shell pwd)/examples/basic/outputs

run: docker
	mkdir -p ./outputs ./data ./conf
	docker run --rm -it --name $(PROJECT) \
		--gpus all \
		--shm-size 8G \
		-v $(DATA_PATH):/app/data \
		-v $(CONF_PATH):/app/conf \
		-v $(OUTPUT_PATH):/app/outputs \
		$(PROJECT):latest \
		-c "python examples/basic"

style: $(VENV)/bin/activate-quality
	$(PYTHON) -m autoflake -r -i --remove-all-unused-imports --remove-unused-variables $(QUALITY_DIRS)
	$(PYTHON) -m isort $(QUALITY_DIRS)
	$(PYTHON) -m autopep8 -a -r -i --max-line-length=$(LINE_LEN) $(QUALITY_DIRS)
	$(PYTHON) -m black --line-length $(LINE_LEN) --target-version $(PY_VER_SHORT) $(QUALITY_DIRS)

tag-version: 
	git tag -a "$(VERSION)"

test: $(VENV)/bin/activate-test
	$(PYTHON) -m pytest \
		-rs \
		--cov=./combustion \
		--cov-report=xml \
		--cov-report=term \
		./tests/

test-%: $(VENV)/bin/activate-test
	$(PYTHON) -m pytest -rs -k $* ./tests/ 

test-pdb-%: $(VENV)/bin/activate-test
	$(PYTHON) -m pytest -rs --pdb -k $* ./tests/ 

types: $(VENV)/bin/activate-dev node_modules ## checks types with pyright
	npx --no-install pyright -p pyrightconfig.json 

upload: package
	$(PYTHON) -m pip install --upgrade twine
	$(PYTHON) -m twine upload --repository combustion dist/*

upload-test: package
	$(PYTHON) -m pip install --upgrade twine
	$(PYTHON) -m twine upload --repository testpypi dist/*


venv: $(VENV)/bin/activate

$(VENV)/bin/activate: setup.py requirements.txt
	test -d $(VENV) || $(PY_VER) -m venv $(VENV)
	$(PYTHON) -m pip install -U pip 
	$(PYTHON) -m pip install -e .
	touch $(VENV)/bin/activate

$(VENV)/bin/activate-%: requirements.%.txt
	test -d $(VENV) || $(PY_VER) -m venv $(VENV)
	$(PYTHON) -m pip install -U pip 
	$(PYTHON) -m pip install -Ur $<
	touch $(VENV)/bin/activate-$*
