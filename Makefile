.PHONY: setup

setup:
	pipenv --rm || true
	pip cache purge
	pipenv install --skip-lock --python 3.12
