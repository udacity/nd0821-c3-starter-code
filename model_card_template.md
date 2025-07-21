# Model Card

For more information, refer to the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This model predicts whether an individual’s annual income exceeds \$50,000. It uses scikit-learn’s `GradientBoostingClassifier` (version 1.2.0), with hyperparameters tuned via `GridSearchCV`. The chosen hyperparameters are:
- `learning_rate`: 1.0
- `max_depth`: 5
- `min_samples_split`: 100
- `n_estimators`: 10

The trained model is saved as a pickle file in the `model` directory. All training steps and evaluation metrics are recorded in `journal.log`.

## Intended Use
The model is intended for educational, academic, or research use to predict if an individual’s income exceeds \$50,000 based on selected features. It is not recommended for commercial or real-world decision-making without further validation.

## Training Data
The model was trained on the Census Income Dataset from the UCI Machine Learning Repository ([link](https://archive.ics.uci.edu/ml/datasets/census+income)), provided in CSV format. The dataset contains 32,561 rows and 15 columns, including the binary target variable "salary," 8 categorical features, and 6 numerical features. Feature descriptions are available at the UCI repository.

The target variable "salary" is imbalanced, with about 75% of samples labeled '<=50K' and 25% labeled '>50K'. Basic cleaning removed leading and trailing whitespace. More details on data exploration and cleaning can be found in the `data_cleaning.ipynb` notebook.

Data was split into training and test sets using an 80-20 stratified split to maintain class distribution. Categorical features were one-hot encoded, and the target variable was binarized for training.

## Metrics
Model performance was evaluated using precision, recall, F-beta score, and the confusion matrix.

Test set results:
- Precision: 0.759
- Recall: 0.643
- F-beta: 0.696

Confusion matrix:
```
[[4625  320]
 [ 560 1008]]
```

## Ethical Considerations
The dataset may not accurately represent salary distributions across all population groups and should not be used to infer income levels for specific demographics.

## Caveats and Recommendations
The data was extracted from the 1994 Census database and is outdated. It should not be considered a current statistical representation of the population. Use this dataset primarily for training and experimentation with ML classification tasks.