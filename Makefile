.PHONY: clean quality style test venv 

LINE_LEN=119
PY_VER=py37
QUALITY_DIRS=tests src 
VENV=.env

clean: 
	find src tests -name '__pycache__' -type d -exec rm -rf {} \;
	find src tests -name '*@neomake*' -type f -delete
	rm -rf .env

quality:
	black --check --line-length $(LINE_LEN) --target-version $(PY_VER) $(QUALITY_DIRS)
	isort --check-only --recursive $(QUALITY_DIRS)
	flake8 $(QUALITY_DIRS)

style:
	black --line-length $(LINE_LEN) --target-version $(PY_VER) $(QUALITY_DIRS)
	isort --recursive $(QUALITY_DIRS)

test: venv
	. .env/bin/activate; python -m pytest -n auto --dist=loadfile -s -v ./tests/

test-%: venv
	. .env/bin/activate; python -m pytest -k $* -s -v ./tests/ 

test-pdb-%: venv
	. .env/bin/activate; python -m pytest --pdb -k $* -s -v ./tests/ 

venv: .env/bin/activate

.env/bin/activate: setup.py
	test -d .env || virtualenv .env
	. .env/bin/activate; pip install -Ue .[dev]
