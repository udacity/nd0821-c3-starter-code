[//]: # (Image References)
[image0]: ./screenshots/MLOps_proj3_tree.PNG "proj3 structure"
[image1]: ./screenshots/continuous_deployment.png "github action"
[image2]: ./plots/numFeats_outlierDist_sex_boxplot.png "feat dist by sex plot"
[image3]: ./plots/general_dist_age-hoursPerWeek_boxplot.png "hours-per-week by age boxplots"
[image4]: ./plots/salary_dist_hoursPerWeek-age-sex_plot.png "salary dist by age sex plot"
[image5]: ./plots/capitalGain_dist_age-hoursPerWeek-sex_plot.png "capital gain dist by hours-per-week sex"
[image6]: ./plots/sex_plot.png  "sex plot"
[image7]: ./screenshots/education-group_people-count.PNG "education people count"
[image8]: ./plots/eduLevel_dist_age-race_plot.png "education level grouping by age race"
[image9]: ./screenshots/example.PNG "fastapi income negative"
[image10]: ./screenshots/MLOps_proj3_FastAPI_docsPredictPersonIncomeNegativeExample_ResponseCode.PNG "fastapi income negative response"
[image11]: ./screenshots/MLOps_proj3_Render_createNewWebService.PNG "render web service"
[image12]: ./screenshots/MLOps_proj3_Render_webservice_live.PNG "render web service life"
[image13]: ./screenshots/MLOps_proj3_Render_webservice_live_test_status.PNG "render web service test"
[image14]: ./screenshots/live_post.png "render web service script result"
[image15]: ./screenshots/render_auto-deploy_settings.PNG "render deploy settings"
[image16]: ./screenshots/render_deployed_censusproject.PNG "render app deployed"
[image17]: ./screenshots/render_deployed-app_browser-homepage.PNG "render app welcome"
[image18]: https://readthedocs.org/projects/pycodestyle/badge/ "Inline docs"

[![made-with-python](https://img.shields.io/badge/Made%20with-Python3.11-1f425f.svg?style=flat&logo=python3.11)](https://www.python.org/)
[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg?style=flat&logo=appveyor)](https://opensource.org/license/mit/)
![Inline docs][image18]
[![Known Vulnerabilities](https://snyk.io/test/github/IloBe/US_CensusData_Classifier_PipelineWithDeployment/badge.svg?style=flat-square)](https://snyk.io/test/github/IloBe/US_CensusData_Classifier_PipelineWithDeployment)

# US Census Data - Creating and Deploying a Classifier Pipeline as Web Service

This is the third project of the course <i>MLOps Engineer Nanodegree</i> by Udacity, called <i>Deploying a Scalable Pipeline in Production</i>. Its instructions are available in Udacity's [repository](https://github.com/udacity/nd0821-c3-starter-code/tree/master/starter).

We develop a classification model artifact for production on public available US Census Bureau data and monitor the model performance on various data slices as business goal.

Regarding data science goals for this classification prediction, we start with the ETL (Extract, Transform, Load) transformer pipeline including EDA (Exploratory Data Analysis) activities, diagrams and reports, followed by the ML (Machine Learning) pipeline for the investigated prediction model, in our case a _binary XGBoost Classifier_. The estimator is selected by using cross validation concept with early stopping for the training phase. This best estimator evaluated by metrics is selected as deployment artifact together with the associated column transformer used for data preprocessing.

General information about the deployed XGBoost classifier, the used data, their training condition and evaluation results can be found in the [Model Card](https://github.com/IloBe/US_CensusData_Classifier_PipelineWithDeployment/blob/master/model_card.md) description.

Regarding software engineering principles, beside documentation, logging and python style, we create _unit tests_. Slice validation and the tests are incorporated into a _CI/CD framework_ using GitHub Actions. Then, the model is deployed using the [_FastAPI_](https://fastapi.tiangolo.com/) web framework and [_Render_](https://dashboard.render.com/#) as open-source web service.

The unit tests are written via [_pytest_](https://docs.pytest.org/en/7.1.x/explanation/goodpractices.html#src-layout) for GET and POST prediction requests for the FastAPI component as well as for the mentioned data and model task parts. All unit test results are reported in associated html files of the [tests directory](https://github.com/IloBe/US_CensusData_Classifier_PipelineWithDeployment/tree/master/tests).

All project relevant configuration values, including model hyperparameter ranges for the cross validation concept, are handled via specific configuration _config.yml_ file.

For versioning tasks, [_git_](https://git-scm.com/) and [_dvc_](https://dvc.org/doc/use-cases/versioning-data-and-models), handled with ignore files content, are chosen. If a remote storage, like AWS S3 or Azure shall be used as future task, dvc[all] for the selected dvc version is installed via requirements.txt file as well for specific configuration. By now, only dvc 'local' remote is set.


## Environment Set Up
* Working in a command line environment is recommended for ease of use with git and dvc. Working on Windows, [WSL2 and Ubuntu (Linux)](https://ubuntu.com/tutorials/install-ubuntu-on-wsl2-on-windows-11-with-gui-support#1-overview) is chosen for this project implementation.
* We expect you have at least Python 3.11.13 installed, furthermore having forked this project repo locally and activate it in your virtual environment to work on for your own. So, in your root directory `path/to/census-project` create a new virtual environment depending on the selected OS and use the supplied _requirements.txt_ file to install the needed libraries via the following process:

### User Process
This project uses Conda for environment management and pip-tools for dependency locking.

1.  **Create and activate the Conda environment:**
    ```bash
    conda create --name my-project-env python=3.11.13
    conda activate my-project-env
    ```

2.  **Install dependencies:**
    Use the locked requirements file for a reproducible installation.
    ```bash
    pip install -r requirements.txt
    ```

### Developer Workflow
If updates are needed, put them in the top-level <i>requirements.in</i> file. There the directly needed packages are listed. <i>pip-compile</i> resolves the search of necessary dependencies together with the <i>pyproject.toml</i> file and creates the final <i>requirements.txt</i> file.

1.  Add or modify a package in `requirements.in`.
2.  Regenerate the lock file:
    ```bash
    pip-compile requirements.in
    ```
3.  Install the new packages:
    ```bash
    pip install -r requirements.txt
    ```
4.  Commit **both** `requirements.in` and `requirements.txt` to Git.


## Project Structure
* Main coding files are stored in the ``src`` and test scripts in the ``tests`` project root subdirectories. The FastAPI RESTful web application is called via _main.py_ file stored in the src directory, but associated schemas and request examples data are part of the src/app subdirectory. All administrative asset files, like plots, screenshots, configuration, logs, as well as model and dataset files are stored in their own directories in parallel to the source code.<br>

* The general project structure looks like:<br>
![proj3 structure][image0]


* In our GitHub repository an automatic Action script is set up to check amongst others dependencies, linting and unit testing.
![github action][image1]

<br>

## Data
* The download raw _census.csv_ file is preprocessed and stored as new .csv file. Both files are committed and versioned with _dvc_.
* Some exploratory data analysis is implemented and visualised. They are stored as .png plot or screenshot files in the associated directories.

Examples are the following ones, regarding amongst others distributions of hours-per-week, salary, capital-gain and education by few feature attributes like age, sex or race. As investigated there is some bias according man (twice as much as women) and white people. Furthermore, it is interesting that according capital gain female representatives earn much often a much higher value for less working hours compared to man. In general, people work >40 hours per week if they are between 25 and 60 years old.

Several other insights are visualised and stored as .png files. So, have a look there if you are interested in further analysis.

![feat dist by sex plot][image2]

![hours-per-week by age boxplots][image3]

![salary dist by age sex plot][image4]

![capital gain dist by hours-per-week sex][image5]

![sex plot][image6]

![education people count][image7]

![education level grouping by age race][image8]

<br>

## Model
* As machine learning model that trains on the clean data _XGBoost Classifier_ is selected and the best found and evaluated estimator is stored as pickle file (...artifact.pkl) in the associated model directory.
* Additionally, a function exists that outputs the performance of the model on slices of the categorical features. Performance evaluation metrics of such categorical census feature slices are stored in a _slice_output.txt_ file. As an example, the metric block looks like:
    
  ```
    workclass - Private:
    Precision: 0.83, Recall: 0.66, Fbeta: 0.73
    Confusion Matrix: 
    [[2907  119]
    [ 297  572]]  
    
    workclass - Self-emp-not-inc:
    Precision: 0.83, Recall: 0.57, Fbeta: 0.68
    Confusion Matrix: 
    [[358  16]
    [ 58  77]]
    
    ...
  ```
* As mentioned, the [Model Card](https://github.com/IloBe/US_CensusData_Classifier_PipelineWithDeployment/blob/master/model_card.md) informs about our found insights of the binary classification estimator including evaluation diagrams and general metrics.
<br>


## API Creation
* As web framework to create a RESTful API [_FastAPI_](https://fastapi.tiangolo.com/) is chosen for app implementation. A _pydantic_ _BaseModel_ instance handels the  POST body, e.g. dealing with hyphens in data feature names which is not allowed in Python.

* As high performance ASGI server [uvicorn](https://www.uvicorn.org/) is selected. The FastAPI web app _uvicorn_ server can be started in the projects root directory via CLI python command:
    
  ```
    python ./src/main.py
  ```  
  
There in "__main__" it calls
    
  ```
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
  ```
  
Remember, this code is for development purpose, in production the reload option shall be set to False resp. not used. In other words, the start command e.g. on our render deployment web service (see below) is:<br>
uvicorn src.main:app --host 0.0.0.0 --port 8000

* So, locally we start our implemented browser web application from project root with

  ```
  http://127.0.0.1:8000/docs
  or
  http://localhost:8000/docs
  ```

As an examples regarding the use case of having a person earning <=50K as income, you are going to get the following UI's:  

![fastapi income negative][image9]

![fastapi income negative response][image10]

<br>

## API Deployment
* As open-source tool for our web service deployment, we use [Render](https://render.com/docs) and a free account there. From the Render.com landing page, click the "Get Started" button to open the sign-up page. You can create an account by linking your GitHub, GitLab, or Google account or provide your email and password. Then, the render account must be connected with our GitHub account, so, the usage of render services is guaranteed. Have in mind, shell and jobs are not supported for free instance types. As stated by FastAPI company tiangolo "For a web API, it normally involves putting it in a remote machine, with a server program that provides good performance, stability, etc, so that your users can access the application efficiently and without interruptions or problems." But using a free account, the service is limited.

* Our new application is deployed from our public GitHub repository by creating a new [Web Service](https://render.com/docs/web-services) for this specific project GitHub URL. 

![render web service][image11]

<br>

* Because default render Python version is 3.7 and this version has issues with dvc, the environment variable PYTHON_VERSION has to be configured being version 3.10.9.
* After selection, render starts its advanced deployment configuation, some parameters are already set, some have to be set manually appropriately. Render guides you through with easy to handle UI's.

![render deploy settings][image15]

<br>

* That's it. Implement coding changes, push to the GitHub repository, and the app will automatically redeploy each time, but it will only deploy if your continuous integration action passes.

![render app deployed][image16]

<br>

* Regarding the automatically created render census-project app link used as browser link
  ```
  https://census-project-xki0.onrender.com
  ```
  we get the welcome page message
  
![render app welcome][image17]

<br>

* Have in mind: if you rely on your CI/CD to fail before fixing an issue, it slows down your deployment. Fix issues early, e.g. by running an ensemble linter like flake8 locally before committing changes.
* For checking the render deployment, a python file exists that uses the httpx module to do one GET and POST on the live render web service and prints its results. 

On the Render web service site after deployment
![render web service life][image12]
<br>
and as result of the httpx test script for GET and POST
![render web service test][image13]
<br>
![render web service script result][image14]

<br>

## License
This project coding is released under the [MIT](https://github.com/IloBe/US_CensusData_Classifier_PipelineWithDeployment/blob/master/LICENSE.txt) license.
