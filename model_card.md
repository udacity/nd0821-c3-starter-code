# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Belen Esteve Cogollos created this model in September 2026 as part of the Udacity MLOps nanodegree (project 3). It is a scikit-learn 1.7.2 `RandomForestClassifier` with 100 trees, a maximum depth of 15, a minimum of 2 samples per leaf and `random_state=42`.

Categorical features are one-hot encoded with a `OneHotEncoder` that ignores unknown categories, and the label is binarized with a `LabelBinarizer`. Continuous features are passed to the model unscaled.

The trained model, encoder and label binarizer are saved as `model/model.joblib`, `model/encoder.joblib` and `model/lb.joblib`. All three files are required for inference. The model can be retrained by running `python -m starter.train_model` from the repository root.

## Intended Use
Predict whether a person's annual income is above or below $50K from census attributes. The model is meant for learning and for demonstrating how to deploy an ML pipeline. It should not be used to make decisions about real people, such as for credit, hiring or benefits.

## Training Data
The UCI Census Income ("Adult") dataset, extracted from the 1994 US Census database (https://archive.ics.uci.edu/dataset/20/census+income). It has 32,561 rows, 14 features and the binary label `salary` (`<=50K` or `>50K`). About 24% of rows are `>50K`.

The raw CSV had a space after every comma. These were removed; nothing else was changed. Missing values appear as `?` in `workclass`, `occupation` and `native-country`, and are kept as their own category.

The data was split 80/20 into training and evaluation sets, stratified on `salary` with `random_state=42`. The model was trained on the 80% split (26,048 rows).

The categorical features are `workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`, `sex` and `native-country`. The continuous features are `age`, `fnlgt`, `education-num`, `capital-gain`, `capital-loss` and `hours-per-week`.

## Evaluation Data
The held-out 20% split (6,513 rows), processed with the encoder and label binarizer fitted on the training data.

## Metrics
The model is evaluated with precision, recall and F1 score, treating `>50K` as the positive class. On the evaluation set it achieves a precision of 0.80, a recall of 0.59 and an F1 score of 0.68. In other words, 80% of the people it predicts as earning over $50K really do, but it finds only 59% of all the people who earn over $50K.

| Metric    | Value  |
|-----------|--------|
| Precision | 0.8009 |
| Recall    | 0.5874 |
| F1        | 0.6777 |

The file `slice_output.txt` lists performance for each value of every categorical feature. The table below shows selected slices for `sex` and `race`. Recall is noticeably lower for women and for Black and American Indian/Eskimo people.

| Slice              | n     | Precision | Recall | F1     |
|--------------------|-------|-----------|--------|--------|
| sex = Male         | 4,355 | 0.7897    | 0.6017 | 0.6830 |
| sex = Female       | 2,158 | 0.8803    | 0.5102 | 0.6460 |
| race = White       | 5,533 | 0.8069    | 0.5929 | 0.6836 |
| race = Black       | 662   | 0.8000    | 0.4731 | 0.5946 |
| race = Asian-Pac-Islander | 200 | 0.6667 | 0.6538 | 0.6602 |
| race = Amer-Indian-Eskimo | 73  | 0.7500 | 0.3333 | 0.4615 |

When a slice has no positive predictions or no positive labels, precision or recall is reported as 1.0 (`zero_division=1`). Very small slices (for example n = 1) are not meaningful.

## Ethical Considerations
- The data includes sensitive attributes (`race`, `sex`, `native-country`, `marital-status`) and the model uses them directly as features.
- Recall is lower for women (0.51 vs. 0.60 for men), for Black people (0.47 vs. 0.59 for White people) and for American Indian/Eskimo people (0.33). This means the model misses high earners in these groups more often.
- The data reflects the income gaps and sampling of the 1994 US census. A model trained on it can reproduce those patterns.

## Caveats and Recommendations
- The data is over 30 years old and US-only, and incomes are not adjusted for inflation. Predictions will not transfer to today's population or to other countries.
- Recall is moderate (0.59). If missing high earners is costly, tune the decision threshold or use class weights.
- Hyperparameters were not tuned. K-fold cross-validation and a small grid search would give more reliable estimates.
- Before any real use, run a fuller fairness audit (for example with Aequitas, which is already in `requirements.txt`) and consider removing or mitigating the sensitive features.
