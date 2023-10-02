[//]: # (Image References)
[image0]: ./sreenshots/MLOps_proj3_tree.PNG "proj3 structure"
[image1]: ./plots/numFeats_outlierDist_sex_boxplot.png "feat dist by sex plot"
[image2]: ./plots/normalDistTest_hours-per-week.PNG "hours-per-week gauss dist or not"
[image3]: ./plots/general_dist_age-hoursPerWeek_boxplot.png "hours-per-week by age boxplots"
[image4]: ./plots/hoursPerWeek-Regression_dist_age-race_plot.png "regression hours-per-week by age race"
[image5]: ./plots/salary_dist_hoursPerWeek-age-sex_plot.png "salary dist by age sex plot"
[image6]: ./plots/capitalGain_dist_age-hoursPerWeek-sex_plot.png "capital gain dist by hours-per-week sex"
[image7]: ./sreenshots/education-group_people-count.PNG "education people-count grouping"
[image8]: ./plots/eduLevel_dist_age-race_plot.png "education level grouping by age race"
[image9]: ./screenshots/MLOps_proj3_FastAPI_gitHubPrecommitHook.PNG "github action"
[image10]: ./screenshots/MLOps_proj3_FastAPI_docsLandingPage.PNG "fastapi landing page"
[image11]: ./screenshots/MLOps_proj3_FastAPI_docsGetRootWelcomeMsg.PNG "fastapi welcome"
[image12]: ./screenshots/MLOps_proj3_FastAPI_docsPredictPersonIncomeNegativeExample.PNG "fastapi income negative"
[image13]: ./screenshots/MLOps_proj3_FastAPI_docsPredictPersonIncomeNegativeExample_ResponseCode.PNG "fastapi income negative response"
[image14]: ./screenshots/render_createNewWebService.PNG "render web service"


# Creating and Deploying a Classifier Pipeline for US Census Data

This is the third project of the course <i>MLOps Engineer Nanodegree</i> by Udacity, called <i>Deploying a Scalable Pipeline in Production</i>. Its instructions are available in Udacity's [repository](https://github.com/udacity/nd0821-c3-starter-code/tree/master/starter).

We develop a classification model on public available US Census Bureau data and monitor the model performance on various data slices as business goal.

Regarding software engineering principles, we create _unit tests_. Slice validation and the tests are incorporated into a _CI/CD framework_ using GitHub Actions. Then, the model is deployed using the FastAPI framework and render as open-source web service.

Regarding data science goals for this classification prediction, we start with the ETL (Extract, Transform, Load) pipeline including EDA (Exploratory Data Analysis) activities and reports, followed by the ML (Machine Learning) pipeline for the investigated prediction model, in our case a binary XGBoost Classifier. The estimator is selected by using cross validation concept with early stopping for the training phase.

General information about the deployed XGBoost classifier, the used data, their training condition and evaluation results can be found in the [Model Card](https://github.com/IloBe/US_CensusData_Classifier_PipelineWithDeployment/blob/master/model_card.md) description.

The Unit tests are written via _pytest_for GET and POST prediction requests for the FastAPI component as well as for the mentioned data and model task parts. All unit test results are reported in associated html files of the [tests directory](https://github.com/IloBe/US_CensusData_Classifier_PipelineWithDeployment/tree/master/tests).

All project relevant configuration values, including model hyperparameter ranges for the cross validation concept, are handled via specific configuration yaml file. For versioning tasks, _git_ and _dvc_, handled with ignore files content, are chosen.


## Environment Set up
* Working in a command line environment is recommended for ease of use with git and dvc. If on Windows, [WSL2 and Ubuntu (Linux)](https://ubuntu.com/tutorials/install-ubuntu-on-wsl2-on-windows-11-with-gui-support#1-overview) is recommended.
* We expect you have at least Python 3.10.9 e.g. via conda installed, furthermore having forked this project repo locally and activate it in your virtual environment to work on it for your own. So, in your root directory `path/to/US-census-project` create a new virtual environment depending on the selected OS and use the supplied _requirements.txt_ file to install the needed libraries e.g. via

  ```
    pip install -r requirements/requirements.txt
  ```
or use

  ```
    conda create -n [envname] "python=3.10.9" scikit-learn pandas numpy pytest jupyter jupyterlab fastapi uvicorn ... <library-list> -c conda-forge
  ```


## Project Structure
* Main coding files are stored in the ``src`` and test scripts in the ``tests`` project root subdirectories. The FastAPI RESTful web application is called via _main.py_ file stored in the src directory, but associated schemas and request examples data are part of the src/app subdirectory. All administrative asset files, like plots, screenshots, configuration, logs, as well as model and dataset files are stored in their own directories in parallel to the source code.<br>

* The general project structure looks like:<br>
![proj3 structure][image0]


* In our GitHub repository an automatic Action script is set up to check amongst others dependencies, linting and unit testing.
![github action][image9]


## Data
* The download raw _census.csv_ file is preprocessed and stored as new .csv file. Both files are committed and versioned with _dvc_.
* Some exploratory data analysis is implemented and visualised. They are stored as .png plot or screenshot files.

Examples are the following ones, regarding amongst others distributions of hours-per-week, education, capital-gain and salary by few feature attributes like age, sex or race. Several other insights are visualised and stored as .png files. So, have a look there if you are interested in further analysis.

![feat dist by sex plot][image1]

![hours-per-week gauss dist or not][image2]

![hours-per-week by age boxplots][image3]

![regression hours-per-week by age race][image4]

![salary dist by age sex plot[][image5]

![capital gain dist by hours-per-week sex][image6]

![education_people_count group][image7]

![education level grouping by age race][image8]


# Model
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
* As mentioned, the model card informs about our found insights of the binary classification estimator including evaluation diagrams and general metrics.


# API Creation
* As Web framework to create a RESTful API _fastapi_ is chosen for app implementation. A _pydantic_ _BaseModel_ instance handels the  POST body, e.g. dealing with hyphens in data feature names which is not allowed in Python.

* As high performance ASGI server [uvicorn](https://www.uvicorn.org/) is selected. The FastAPI web app _uvicorn_ server can be started in the projects root directory via CLI python command:
    
  ```
    python ./src/main.py
  ```  
  
There in "__main__" it calls
    
  ```
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
  ```

Remember, this code is for development purpose, in production the reload option shall be set to False resp. not used. In other words, the start command e.g. on our render deployment web service (see below) is:<br>
uvicorn main:app host="0.0.0.0" port=8000

* So , we start the browser web application with

  ```
  http://127.0.0.1:8000/docs
  or
  http://localhost:8000/docs
  ```

As an examples regarding the use case of having a person earning <=50K as income, you are going to get the following UI's:  

![fastapi landing page][image10]

![fastapi welcome][image11]

![fastapi income negative][image12]

![fastapi income negative response][image13]


# API Deployment
* As open-source tool for our web service deployment, we use [Render](https://render.com/docs) and a free account there. From the Render.com landing page, click the "Get Started" button to open the sign-up page. You can create an account by linking your GitHub, GitLab, or Google account or provide your email and password. Then, the render account must be connected with our GitHub account, so, the usage of render services is guaranteed. Have in mind, shell and jobs are not supported for free instance types.

* Our new application is deployed from our public GitHub repository by creating a new [Web Service](https://render.com/docs/web-services) for this specific GitHub URL. As it is written by FastAPI company tiangolo "For a web API, it normally involves putting it in a remote machine, with a server program that provides good performance, stability, etc, so that your users can access the application efficiently and without interruptions or problems."

![render web service][image14]

    * after selection, render starts its advanced deployment configuation, some parameters are already set, some have to be set manually appropriately. Render guides you through with easy to handle UI's.
    * That's it. Implement coding changes, push to the GitHub repository, and the app will automatically redeploy each time, but it will only deploy if your continuous integration action passes. 
    * Have in mind: if you rely on your CI/CD to fail before fixing an issue, it slows down your deployment. Fix issues early, e.g. by running an ensemble linter like flake8 locally before committing changes.
* For checking the render deployment, a python file exists that uses the httpx module to do one GET and POST on the live render web service and prints its results. 


## License
This project coding is released under the [MIT](https://github.com/IloBe/US_CensusData_Classifier_PipelineWithDeployment/blob/master/LICENSE.txt) license.
