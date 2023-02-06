# Model Card

## Model Details
Decision Tree Classifier with default configuration for predictions.

## Intended Use
This model predicts the category of the salary of a person based on it's financials information. The categories are '<=50K' or '>50K'

## Training Data
The original dataset is from the UCI Machine Learning Repository.
Source: https://archive.ics.uci.edu/ml/datasets/census+income
The original data set has 48842 rows, and a 80-20 split was used to break this into a train and test set. No stratification was done. To use the data for training a One Hot Encoder was used on the features and a label binarizer was used on the target class.

## Evaluation Data
20% data is used for evaluation.

## Metrics
Model performance in precision, recall and fbeta are:
precision: 0.608
recall: 0.639
fbeta: 0.623

## Ethical Considerations
Metics were also calculated on data slices. Saved results are in 'slice_output.txt'

## Caveats and Recommendations
The census.csv used in this project differs from the UCI Machine Learning Repository as it has only 32561 rows.