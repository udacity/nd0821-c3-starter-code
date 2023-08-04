# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
The model is a support vector machine that has been trained on census data with the goal of predicating salary.

## Intended Use
This model is intended to predict salary of a person. The main stakeholders for this model are students and academia

## Training Data
The training data was obtained from UC Irvine Machine Learning repository(https://archive.ics.uci.edu/dataset/20/census+income).
The original dataset has 32561 rows, 15 columns, and 80% of the data was used for training.

## Evaluation Data
20% of the data was used for evaluation of the model.

## Metrics
The performance of the model was evaluated with precision, recall, fbeta metrics.

The most recent model achieves the scores below:

- Precision: 0.9694656488549618
- Recall: 0.16157760814249364
- Fbeta: 0.276990185387132


## Ethical Considerations
The dataset does not have a even distribution of race(85.4% of the column was the same value). This could lead to an inaccurate
prediction of the races that are not properly represented.

## Caveats and Recommendations
It is important to note that the data is quite old, which could be used to say that the model is out of date and
therefore cannot be used as an accurate predication of salary. It would be recommended to either use another dataset that
is more uptodate or to use the dataset only for training purposes.