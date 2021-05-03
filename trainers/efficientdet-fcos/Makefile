.PHONY: docker docker-dev clean clean-venv pre-commit quality run style test venv 

PY_VER=py37
QUALITY_DIRS=$(PROJECT) 
CLEAN_DIRS=$(PROJECT) 
VENV=venv
PYTHON=$(VENV)/bin/python3

LINE_LEN=120
DOC_LEN=120

CONFIG_FILE := Makefile.config
ifeq ($(wildcard $(CONFIG_FILE)),)
$(error $(CONFIG_FILE) not found. See $(CONFIG_FILE).example.)
endif
include $(CONFIG_FILE)

clean: 
	find $(CLEAN_DIRS) -path '*/__pycache__/*' -delete
	find $(CLEAN_DIRS) -type d -name '__pycache__' -empty -delete
	find $(CLEAN_DIRS) -name '*@neomake*' -type f -delete
	find $(CLEAN_DIRS) -name '*,cover' -type f -delete

clean-venv:
	rm -rf $(VENV)

quality: 
	black --check --line-length $(LINE_LEN) --target-version $(PY_VER) $(QUALITY_DIRS)
	flake8 --max-doc-length $(DOC_LEN) --max-line-length $(LINE_LEN) $(QUALITY_DIRS) 

run: 
	venv/bin/python -m efficientdet_fcos

prep:
	mkdir -p ./outputs ./data ./conf

style: 
	autoflake -r -i --remove-all-unused-imports --remove-unused-variables $(QUALITY_DIRS)
	isort --recursive $(QUALITY_DIRS)
	autopep8 -a -r -i --max-line-length=$(LINE_LEN) $(QUALITY_DIRS)
	black --line-length $(LINE_LEN) --target-version $(PY_VER) $(QUALITY_DIRS)

venv: $(VENV)/bin/activate

$(VENV)/bin/activate: requirements.txt ../../src/combustion
	test -d $(VENV) || python3 -m venv $(VENV)
	$(PYTHON) -m pip install -U pip
	$(PYTHON) -m pip install -e ../..[dev]
	$(PYTHON) -m pip install -r requirements.txt
	touch $(VENV)/bin/activate
