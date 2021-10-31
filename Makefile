.DEFAULT_GOAL := help

help:
		@echo "Please use 'make <target>' where <target> is one of"
		@echo ""
		@echo "  setup       create python virtual environment"
		@echo "  lint        lint code"
		@echo "  test        run all the tests"
		@echo "  all         runs both lint and test commands"
		@echo ""
		@echo "Check the Makefile to know exactly what each target is doing."

setup:
	conda env create --file starter/environment.yml
	conda activate ml_census_fastapi

test:
	pytest -vv -p no:logging

lint:
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

all: lint test
