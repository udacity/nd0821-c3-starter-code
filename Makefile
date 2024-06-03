
createconda:
	conda create -n census-classifier "python=3.10" scikit-learn pandas numpy pytest jupyter jupyterlab fastapi uvicorn -c conda-forge
activate:
	@echo "Run 'conda activate census-classifier' to activate the environment."
deactivate:
	@echo "Run 'conda deactivate' to deactivate the environment."

install:
	./scripts/run_in_conda.sh census-classifier "conda install -y flake8 pytest pytest-xdist"

lint:
	./scripts/run_in_conda.sh census-classifier "flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics && flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics && \
	autopep8 --in-place --aggressive -r ./src && \
	pylint ./src"

test:
	./scripts/run_in_conda.sh census-classifier "pytest -n 4"

train:
	./scripts/run_in_conda.sh census-classifier "python3 src/train_model.py"

score:
	./scripts/run_in_conda.sh census-classifier "python3 src/score_model.py"