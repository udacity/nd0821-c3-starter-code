# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

The code on this repository use Random Forest Classifier with default parameters for this classification task.
It was fitted to learn the classification of a person predicting if his or her income is higher or lower than 50K
per year.
This project is part of the Machine Learning DevOps Nanodegree from Udacity.

## Intended Use
The main goal is to predict whether a person salary  is greater or lower than $ 50k per year based on some social-economics
characteristics. Besides, this project intends to apply recent acquired skills to develop MLOps pipeline to deploy ML models
## Training Data
The training data `census.csv` description can be found at: https://archive.ics.uci.edu/ml/datasets/census+income

For training, 80% of the original dataset was randomly selected.

## Evaluation Data

The remaining 20% of the data is part of the test set.

## Metrics

Precision, recall and F-beta metrics were chosen to evaluate the model's performance:

- Precision: 0.7821953327571305
- Recall: 0.5720606826801518
- F-beta: 0.660825118656444

## Ethical Considerations

This dataset was taken from the UCI repository and it is already anonymized. 

Given that the raw dataset has census information from the United States only, the trained model
could be only applied for american people or residents in the United States. On top of this, the model
could have some biases based on sex, race, native-country, and age. Indeed, the dataset has
skew values for categories like race, native-country, and sex variables.

The predictions of the model should be taken carefully according the people on whom this
model is used. A possible solution to avoid any bias in the trained model is to use up sampling and
down sampling techniques.

## Caveats and Recommendations

The dataset was taken from the 1994 Census database.
