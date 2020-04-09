.PHONY: clean clean-venv pre-commit quality style test venv 

PY_VER=py37
QUALITY_DIRS=src tests
CLEAN_DIRS=src tests
VENV=venv
PYTHON=$(VENV)/bin/python3

LINE_LEN=120
DOC_LEN=120

clean: 
	find $(CLEAN_DIRS) -path '*/__pycache__/*' -delete
	find $(CLEAN_DIRS) -type d -name '__pycache__' -empty -delete
	find $(CLEAN_DIRS) -name '*@neomake*' -type f -delete

clean-venv:
	rm -rf $(VENV)

pre-commit: venv
	pre-commit install

quality: venv
	black --check --line-length $(LINE_LEN) --target-version $(PY_VER) $(QUALITY_DIRS)
	flake8 --max-doc-length $(DOC_LEN) --max-line-length $(LINE_LEN) $(QUALITY_DIRS) 

style: venv
	autoflake -r -i --remove-all-unused-imports --remove-unused-variables $(QUALITY_DIRS)
	isort --recursive $(QUALITY_DIRS)
	autopep8 -a -r -i --max-line-length=$(LINE_LEN) $(QUALITY_DIRS)
	black --line-length $(LINE_LEN) --target-version $(PY_VER) $(QUALITY_DIRS)

test: venv
	$(PYTHON) -m pytest -n auto --dist=loadfile -s -v ./tests/

test-%: venv
	$(PYTHON) -m pytest -k $* -s -v ./tests/ 

test-pdb-%: venv
	$(PYTHON) -m pytest --pdb -k $* -s -v ./tests/ 

venv: $(VENV)/bin/activate

$(VENV)/bin/activate: setup.py src/combustion
	test -d $(VENV) || virtualenv $(VENV)
	$(PYTHON) -m pip install -U pip
	$(PYTHON) -m pip install -e .[dev]
	touch $(VENV)/bin/activate
