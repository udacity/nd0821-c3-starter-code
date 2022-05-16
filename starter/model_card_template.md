# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
The used model is a Random Forest classifier using the default parameters. The model is used to predict if someone is making more than 50 K in salary based on the following attributes

age
workclass
fnlgt
education
education-num
marital-status
occupation
relationship
race
sex
capital-gain
capital-loss
hours-per-week
native-country
salary

## Intended Use

The model should be used to predict the salary based on a set of attributes. The user of this model are students of the a Udacity Nano Degree Program. 

## Training Data
Data can be found here Census Income data set and locally in /data/census.csv

## Evaluation Data

## Metrics
_Please include the metrics used and your model's performance on those metrics._

## Ethical Considerations
The data used to train the model might contain biases. Some attributes that are collected from conducting a survey, such as hours per week, might contain biases based on influence from friends, co-workers or expectations. Not all countries are represented in the native country attribute and the data set is probably not large enough to assume that the model predicts well with native country attribute.
## Caveats and Recommendations
A very simple Random Forest model with default parameters was trained on the data set and no effort was put into training a model with better performance since the Udacity projects was more about CI/CD. The model would also perform better with a larger data set.