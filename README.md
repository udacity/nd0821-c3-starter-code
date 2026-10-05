Working in a command line environment is recommended for ease of use with git and dvc. If on Windows, WSL1 or 2 is recommended.

The repository root is the project root. Run installation, tests, DVC, and deployment commands from here. `requirements.txt`, `setup.py`, `main.py`, `sanitycheck.py`, and `data/census.csv` are at the root. `starter/` contains the Python package; for example, import `process_data` with `from starter.ml.data import process_data`.

The training script and API are exercises to complete. Once implemented, run the training script with `python -m starter.train_model` and serve the API from the root with `uvicorn main:app`. Run the rubric helper from the root with `python sanitycheck.py tests` after writing API tests.

# Environment Set up
* **Option 1: Using pip and venv (Recommended)**
    * Ensure you have Python 3.13 installed
    * Create virtual environment: `python3.13 -m venv .venv`
    * Activate environment: `source .venv/bin/activate` (On Windows: `.venv\Scripts\activate`)
    * From the repository root, install dependencies: `pip install -r requirements.txt`
    * Install the local package: `pip install -e .`

* **Option 2: Using conda**
    * Download and install conda if you don't have it already.
    * conda create -n [envname] "python=3.13" scikit-learn pandas numpy pytest jupyter jupyterlab fastapi uvicorn pydantic httpx matplotlib seaborn -c conda-forge
    * Install git either through conda ("conda install git") or through your CLI, e.g. sudo apt-get git.
    * From the repository root, install the local package: `pip install -e .`

## Repositories
* Create a directory for the project and initialize git.
    * As you work on the code, continually commit changes. Trained models you want to use in production must be committed to the repository you submit.
* Connect your local Git repository to GitHub or Azure Repos, following the corresponding course workflow.
* Set up GitHub Actions or Azure Pipelines on your repository. You can use one of the pre-made GitHub Actions if at a minimum it runs pytest and flake8 on push and requires both to pass without error.
    * Make sure you configure CI to use Python 3.13 (same version as development).
    * Note: Add flake8 to requirements.txt if you want to use it for linting: `pip install flake8`

# Data
* Use `data/census.csv`. DVC is optional; follow the course workflow you selected.
* This data is messy, try to open it in pandas and see what you get.
* To clean it, use your favorite text editor to remove all spaces.

# Model
* Using the starter code, write a machine learning model that trains on the clean data and saves the model. Complete any function that has been started.
* Include the final trained model and all fitted preprocessing artifacts needed for inference (such as the encoder and label binarizer) in the submitted repository/files, whether saved as `.pkl`, `.joblib`, or another format. Verify that a fresh clone or extracted submission can load them and run inference without retraining.
* If a required artifact is ignored, add only that file with `git add -f path/to/artifact.pkl` (using its actual path). Review the staged files before committing and pushing.
* Write unit tests for at least 3 functions in the model code.
* Write a function that outputs the performance of the model on slices of the data.
    * Suggestion: for simplicity, the function can just output the performance on slices of just the categorical features.
* Write a model card using the provided template.

# API Creation
*  Create a RESTful API using FastAPI this must implement:
    * GET on the root giving a welcome message.
    * POST that does model inference.
    * Type hinting must be used.
    * Use a Pydantic model to ingest the body from POST. This model should contain an example.
   	 * Hint: the data has names with hyphens and Python does not allow those as variable names. Do not modify the column names in the csv and instead use the functionality of FastAPI/Pydantic/etc to deal with this.
* Write 3 unit tests to test the API (one for the GET and two for POST, one that tests each prediction).

# API Deployment
* Use Render, Heroku, or another cloud application platform that meets the rubric. Heroku no longer offers its former free tier; check your provider's pricing.
* Use the repository root as the deployment root; it contains `requirements.txt` and `main.py`.
* Deploy only tested code from the protected `main` or `master` branch, after both `pytest` and `flake8` pass. Use the same Python version as development.
    * **GitHub:** use a provider integration or GitHub Actions workflow with automatic deployment gated on successful CI.
    * **Azure DevOps:** use an `azure-pipelines.yml` deployment stage that depends on successful CI and runs only for the protected `main` branch. Keep Azure Repos as the source repository.
* Store deployment credentials in authorized secrets, never in project files or pipeline YAML. Follow the course's GitHub to Azure DevOps Translation Guide for platform-specific setup.
* Account for local and hosted path differences, and run tests and lint locally before pushing.
* Write a script using `requests` to POST to the deployed API and print the prediction and HTTP status code. Include the rubric's required deployment and live-request screenshots.
