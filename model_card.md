[//]: # (Image References)
[image1]: ./plots/numFeats_outlierDist_sex_boxplot.png "feat dist by sex plot:"
[image2]: ./plots/salary_dist_hoursPerWeek-age-sex_plot.png "salary dist plot:"
[image3]: ./plots/MLOps_Proj3_trainModel_retrainBestCV_fifthRunPart0_2023-09-21.PNG "best estimator params"
[image4]: ./plots/2023-09-21_21-34_best_retrained_xgb-cv_logloss_treeNo_diagram.png "best xgb cls logloss"
[image5]: ./plots/2023-09-21_21-34_best_retrained_xgb-cv_roc-curve_diagram.png "best xgb cls roc auc"
[image6]: ./plots/MLOps_Proj3_trainModel_retrainBestCV_fifthRunPart4_2023-09-21.PNG "best xgb cls eval"

# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
- Developed by: Ilona Brinkmeier, September 2023
- XGBoost Classifier as a single, basic instance and additionally with GridSearchCV as optimisation concept are examined. The best estimator has been found with GridSearchCV, this model card is written for its retrained, best estimator.
- For this binary classification task a predictive modeling pipeline is developed using public adult US census data from [UCI ML repository](https://archive.ics.uci.edu/dataset/20/census+income).

## Intended Use
- Intended to be used to predict if a person earns >50K US$ per year or not based on the labeled adult census data. 

## Training Data
- For the data given 1996-04-30 extraction was done by *Barry Becker* from the 1994 Census database.
- Attribute information is given from the UCI ML repository. Regarding the raw data, the *columns* are:
  - age
  - workclass
  - fnlwgt
  - education
  - education-num
  - marital-status
  - occupation
  - relationship
  - race
  - sex
  - capital-gain
  - capital-loss
  - hours-per-week
  - native-country

with **target label** feature 'salary' which has been converted during preprocessing being numerical:
  - '>50K': 1
  - '<=50K': 0

- The raw numerical feature distribution is:<br>
![feat dist by sex plot:][image1]

- And its salary distribution regarding hours-per-week, age and sex is:<br>
![salary dist plot:][image2]

- Some preprocessing has been done with pipeline implementation. These tasks include, but are not limited to, dealing with the imbalanced distribution of the target label, duplicate rows, wrong labeling or strings, non-normalised features and  one-hot-encoding for the categorical nominal features. 
- <i>fnlwgt</i> has been removed not having an added value for the prediction task, because it seems to have a kind of index functionality
- <i>native_country<i> has been changed having another bin grouping according [German Society for State Studies e.V.](http://www.staatenkunde.de/dgfs/datenbank/db-sb.php?sb=18&k=4)

For both, the raw and the preprocessed data, profiling reports are created as EDA (exploratory data analysis) tasks stored in directory [./src/eda](./src/eda/).

## Evaluation Data
For the validation datasets containing different samples to evaluate trained ML models the GridSearchCV approach from scikit-learn is implemented with 0.2 test size fraction of data used for final testing. 

The following value ranges for the **grid parameters** are handled (note: read in via configuration file), they are used with early stopping mechanism:
- for our *binary classification*, output probability:<br>
  objective: ["binary:logistic"]
- *evaluation metrics* used e.g. for model plots:<br>
  eval_metric: ["logloss"]<br>
  note regarding XGBoost: early stopping uses only the last given metric of a sequence
- specifies the *number of decision trees* to be boosted<br>
  n_estimators: [100, 150, 200, 250]
- *max_depth* is the tree's maximum depth. Increasing it increases the model complexity<br>
  max_depth: [5, 6, 8, 9]
- *learning_rate* shrinks the weights to make the boosting process more conservative<br>
  learning_rate: [0.01, 0.1, 0.5]
- *gamma* specifies the minimum loss reduction required to make a split<br>
  gamma: [0, 1, 10]
- *percentage of columns* to be samples for each tree<br>
  colsample_bytree: [0.5, 0.7, 1]
- *reg_alpha* provides l1 regularization to the weight, higher values are more conservative<br>
  reg_alpha: [0.01, 0.1, 0.5]
- *reg_lambda* provides l2 regularization to the weight, higher values are more conservative<br>
  reg_lambda: [0.01, 0.1, 0.5]

Best found estimator parameters are:
![best estimator params][image3]
    
## Metrics
Few evaluation metrics are included:<br>
Precision, Recall, Fbeta, Confusion Matrix, Classification Report

As evaluation plots given are:<br>
logloss diagram and ROC curve diagram with AUC value

In general, the receiver operating characteristic (ROC) curve is a metric that is used to measure the performance of a classifier model. It depicts the true positive rate concerning the false positive ones. It also highlights the sensitivity of the classifier model. The area under the curve (AUC) is used for general binary classification problems. AUC will measure the whole two-dimensional area that is available under the entire ROC curve. A perfect model would have an AUC of 1, while a random model would have an AUC of 0.5.<br>
The AUC result of the GridSearchCV best estimator is: 0.93

![best xgb cls logloss][image4]

![best xgb cls roc auc][image5]
    
Validation results of the GridSearchCV best estimator are:<br>
![best xgb cls eval][image6]

## Ethical Considerations
No ethical consideration topics regarding data, human life, risk and harms and their needed risk mitigation strategies or fraught model use cases are detected.

## Caveats and Recommendations
- Regarding the data, have in mind that the raw data have among others a bias towards men (twice as many men as women) and white people mainly originally from the U.S., so, scaling activities are mandatory getting appropriate prediction results.
- Regarding the prediction task, the performance of the grid search cross validation approach is already improved compared to the one of the single instance, but still not the best. As future toDo, final tuning of the XGBoost Classifier via <i>Hyperopt</i> library is recommended.
- Additional, feature importance information of the final resulting XGBoost Classifier model is critical to understand the prediction process. As additional future toDo: usage of <i>SHAP</i> diagrams or simple <i>xgb feature_importances_ parameter</i> bar chart of the X_train columns from the GridSearchCV best model result for identifying which features are most relevant for the target variable.
- Last topic as future toDo is the usage of other classifier types and their evaluation compared to the XGBoost Classifier, even though it was often used by teams that won Kaggle competitions.
