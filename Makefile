.PHONY: quality style test 

LINE_LEN=119
PY_VER=py37
QUALITY_DIRS=tests src 

quality:
	black --check --line-length $(LINE_LEN) --target-version $(PY_VER) $(QUALITY_DIRS)
	isort --check-only --recursive $(QUALITY_DIRS)
	flake8 $(QUALITY_DIRS)

style:
	black --line-length $(LINE_LEN) --target-version $(PY_VER) $(QUALITY_DIRS)
	isort --recursive $(QUALITY_DIRS)

test:
	python -m pytest -n auto --dist=loadfile -s -v ./tests/

test-%:
	python -m pytest -k $* -s -v ./tests/ 

test-pdb-%:
	python -m pytest --pdb -k $* -s -v ./tests/ 

clean: 
	find src tests -name '__pycache__' -type d -exec rm -rf {} \;
	find src tests -name '*@neomake*' -type f -delete
