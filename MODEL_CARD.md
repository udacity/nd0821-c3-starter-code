# Model Card

## Model Details
- **Author:** Paweł Bojemski
- **Date:** 2026-11-19
- **Model type:** Random Forest classifier (`sklearn.ensemble.RandomForestClassifier`)
- **Hyperparameters:** n_estimators=100, random_state=42 (all others default)
- **Versions:** scikit-learn 1.7.2, Python 3.13.13
- **Preprocessing:** `OneHotEncoder(handle_unknown="ignore")` on categorical features, `LabelBinarizer` on the label. Both are saved alongside the model in `model/`.

## Intended Use
Predict whether a person's annual income exceeds $50K from census attributes.
Intended for educational purposes (learning an ML deployment workflow). Not for decisions about real people.

## Training Data
- Source: clean census income dataset (`/home/ubuntu/repo/uda-ml-final/data/census.csv`)
- Cleaning: stripped whitespace, treated `?` as missing (strategy: `drop`), normalised label text, removed duplicates (32,561 raw rows -> 30,139 clean rows)
- Rows: 30,139; 80% used for training (24,111 rows), stratified by label
- Features: 14 total, of which 8 categorical: workclass, education, marital-status, occupation, relationship, race, sex, native-country
- Label: `salary` (`<=50K` vs `>50K`), positive-class rate 24.9%

## Evaluation Data
20% hold-out split (6,028 rows), same preprocessing using encoders fitted on the training split only.

## Metrics
| Metric | Value |
|---|---|
| Precision | 0.703 |
| Recall | 0.625 |
| F1 | 0.661 |

Slice metrics for every categorical feature are in `slice_output.txt`
(slices with n >= 30).

Weakest slices (by F1):
- `marital-status` = **Married-spouse-absent** (n=65): precision 0.00, recall 0.00, F1 0.00
- `education` = **9th** (n=100): precision 1.00, recall 0.12, F1 0.22
- `relationship` = **Other-relative** (n=166): precision 0.50, recall 0.14, F1 0.22

Strongest slices (by F1):
- `occupation` = **Machine-op-inspct** (n=392): precision 0.50, recall 0.24, F1 0.33
- `occupation` = **Transport-moving** (n=285): precision 0.44, recall 0.26, F1 0.33
- `education` = **10th** (n=163): precision 1.00, recall 0.20, F1 0.33

## Ethical Considerations
The data contains sensitive attributes (race, sex, native country, marital status).
Performance differs across groups (see slices above), so predictions may be systematically less reliable for some groups.
Slices with few examples give noisy estimates. The data reflects historical societal biases around income, and the model can reproduce them.

## Caveats and Recommendations
- The data is old and may not reflect today's population or incomes.
- No hyperparameter tuning or class-imbalance handling was done. Precision is 0.70 vs recall 0.62 for the `>50K` class, so the model misses more high earners than it falsely flags.
- Not suitable for hiring, lending, or any decision affecting real people without a fairness audit.
- Next steps: tune hyperparameters, try class weights, and evaluate fairness metrics across groups.
