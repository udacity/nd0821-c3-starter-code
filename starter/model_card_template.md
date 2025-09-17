# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

- **Developer**: Created as part of the ND0821 ML DevOps Engineering Nanodegree program.
- **Model Date**: June 2025
- **Model Version**: v1.0
- **Model Type**: RandomForestClassifier
- **Algorithms/Parameters**:
  - Ensemble method using decision trees
  - Default hyperparameters except `random_state=42`
- **License**: For educational use


## Intended Use

- **Primary Use Case**: Predicting whether an individual's income exceeds \$50K based on demographic attributes (from census data)
- **Primary Users**: Students, educators, or ML practitioners learning deployment and testing practices
- **Out-of-Scope Use Cases**:
  - Real-world income prediction in sensitive applications (e.g., hiring, credit scoring)
  - Use in production environments without fairness audits


## Training Data

- **Source**: UCI Adult Census Income dataset
- **Features**: Age, workclass, education, marital-status, race, sex, and others
- **Target**: Income bracket (<=50K or >50K)


## Evaluation Data

- **Split**: 20% holdout from original dataset
- **Preprocessing**:
  - One-hot encoding for categorical features
  - Label binarization for the target
  - Sliced evaluation by the `marital-status` feature


## Metrics

#### Global Performance:
- **Precision**: 0.7152
- **Recall**: 0.6110
- **F1 Score**: 0.6590

#### Sliced Performance by `marital-status`:

| Marital Status           | Precision | Recall | F1 Score |
|--------------------------|-----------|--------|----------|
| Married-civ-spouse       | 0.7149    | 0.6475 | 0.6795   |
| Never-married            | 0.7692    | 0.4255 | 0.5479   |
| Married-spouse-absent    | 1.0000    | 0.1429 | 0.2500   |
| Divorced                 | 0.6531    | 0.3902 | 0.4885   |
| Separated                | 1.0000    | 0.3636 | 0.5333   |
| Widowed                  | 0.5000    | 0.1429 | 0.2222   |
| Married-AF-spouse        | 1.0000    | 0.0000 | 0.0000   |


## Ethical Considerations

- Disparities in F1 scores across demographic slices may reflect model bias.
- The model was not audited for fairness, bias, or societal impacts.
- Use in decision-making without fairness evaluation could lead to discriminatory outcomes.



## Caveats and Recommendations

- Performance varies significantly by subgroup — further fairness testing is advised.
- Consider collecting more balanced training data for underrepresented categories.
- **Do not deploy** this model without bias analysis, stakeholder review, and fairness auditing.

