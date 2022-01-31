# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This model will predict if a worker has a salary higher or lower than 50k based on several features such as age, education and native country.

## Intended Use
Employers wanting to estimate a fair compensation for its workers.

## Training Data
80% of the full dataset.

## Evaluation Data
20% of the full dataset.

## Metrics
The model metrics include precision, recall and fbeta. The latest model iteration performed as below:
precision: 0.948
recall: 0.926
fbeta: 0.937

## Ethical Considerations
Sensitive information such as salary, race and country of origin is used. Regulators will likely say you can't do that at all.

## Caveats and Recommendations
Hyperparameter tuning and benchmarking of alternative algorithms other than random forest could be expansions to this project.