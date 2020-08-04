.PHONY: docs docker docker-dev clean clean-venv pre-commit quality run style test venv 

PY_VER=py37
QUALITY_DIRS=src tests setup.py
CLEAN_DIRS=src tests
VENV=$(shell pwd)/venv
PYTHON=$(VENV)/bin/python3

SPHINXBUILD=$(VENV)/bin/sphinx-build
#SPHINXOPTS=-W
export SPHINXBUILD
export SPHINXOPTS


LINE_LEN=120
DOC_LEN=120

VERSION := $(shell cat version.txt)

ci-test: venv
	$(PYTHON) -m pytest \
		-rs \
		--cov=./src \
		--cov-report=xml \
		-s -v \
		-m "not ci_skip" \
		./tests/


docs:
	#$(VENV)/bin/sphinx-apidoc -d 1 -E --implicit-namespaces -o docs src/combustion
	cd docs && make html 

docker: 
	docker build \
		--target release \
		-t combustion:latest \
		--file ./docker/Dockerfile \
		./

docker-dev:
	docker build \
		--target dev \
		-t combustion:latest-dev \
		--file ./docker/Dockerfile \
		./

clean: 
	find $(CLEAN_DIRS) -path '*/__pycache__/*' -delete
	find $(CLEAN_DIRS) -type d -name '__pycache__' -empty -delete
	find $(CLEAN_DIRS) -name '*@neomake*' -type f -delete
	find $(CLEAN_DIRS) -name '*,cover' -type f -delete
	cd docs && make clean
	rm -rf docs/api docs/generated
	rm -rf dist

clean-venv:
	rm -rf $(VENV)

package: 
	rm -rf dist
	$(PYTHON) -m pip install --upgrade setuptools wheel
	export COMBUSTION_BUILD_VERSION=$(VERSION) && $(PYTHON) setup.py sdist bdist_wheel

pre-commit: venv
	pre-commit install

quality: 
	black --check --line-length $(LINE_LEN) --target-version $(PY_VER) $(QUALITY_DIRS)
	flake8 --max-doc-length $(DOC_LEN) --max-line-length $(LINE_LEN) $(QUALITY_DIRS) 

DATA_PATH=$(shell pwd)/examples/basic/data
CONF_PATH=$(shell pwd)/examples/basic/conf
OUTPUT_PATH=$(shell pwd)/examples/basic/outputs

run: docker
	mkdir -p ./outputs ./data ./conf
	docker run --rm -it --name combustion \
		--gpus all \
		--shm-size 8G \
		-v $(DATA_PATH):/app/data \
		-v $(CONF_PATH):/app/conf \
		-v $(OUTPUT_PATH):/app/outputs \
		combustion:latest \
		-c "python examples/basic"

style: 
	autoflake -r -i --remove-all-unused-imports --remove-unused-variables $(QUALITY_DIRS)
	isort --recursive $(QUALITY_DIRS)
	autopep8 -a -r -i --max-line-length=$(LINE_LEN) $(QUALITY_DIRS)
	black --line-length $(LINE_LEN) --target-version $(PY_VER) $(QUALITY_DIRS)

tag-version: 
	git tag -a "$(VERSION)"

test: venv
	$(PYTHON) -m pytest \
		-rs \
		--cov=./src \
		--cov-report=xml \
		-s -v \
		./tests/

test-%: venv
	$(PYTHON) -m pytest -rs -k $* -s -v ./tests/ 

test-pdb-%: venv
	$(PYTHON) -m pytest -rs --pdb -k $* -s -v ./tests/ 

upload: package
	$(PYTHON) -m pip install --upgrade twine
	$(PYTHON) -m twine upload --repository pypi dist/*

upload-test: package
	$(PYTHON) -m pip install --upgrade twine
	$(PYTHON) -m twine upload --repository testpypi dist/*

venv: $(VENV)/bin/activate

$(VENV)/bin/activate: setup.py 
	test -d $(VENV) || virtualenv $(VENV)
	$(PYTHON) -m pip install -U pip
	$(PYTHON) -m pip install -e .[dev]
	$(PYTHON) -m pip install git+https://github.com/pytorch/pytorch_sphinx_theme.git
	touch $(VENV)/bin/activate
