# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Used Logistic Regression classifier for prediction. Default configuration were used for training.
## Intended Use

This model should be used to predict the category of the salary of a person based on it's financials attributes.

## Training Data

Source of data https://archive.ics.uci.edu/ml/datasets/census+income ; 80% of the data is used for training suing gridsearch optimization.

## Evaluation Data

Source of data https://archive.ics.uci.edu/ml/datasets/census+income ; 20% of the data is used to testing the model.

## Metrics

The model was evaluated using  F1 beta score, Precision and Recall. The values fall in the range [0.55, 0.75] for all the metrics.

## Ethical Considerations
For Ethical Considerations the metics were also calculated on data slices. This will drive to a model that may potentially discriminate people; 
further investigation before using it should be done.

## Caveats and Recommendations

The data is biased based on gender. Have data imbalance that need to be investigated.