# Model Card

**Model:** Census income classifier (RandomForest)

**Version:** 0.1

**Model Details:**
- **Developed by:** Exercise starter code
- **Model type:** RandomForestClassifier (scikit-learn)
- **Date:** 2026-09-27

**Intended Use:**
- Predict whether an individual's salary is >50K based on census features for educational/demo purposes.

**Training Data:**
- Cleaned census data provided in `data/census.csv`.

**Evaluation Data:**
- Holdout split from the provided dataset used for validation.

**Metrics:**
- Precision, recall, and F1 (fbeta with beta=1).

**Ethical Considerations:**
- Use caution: model may encode biases present in training data (race, sex, etc.).

**Caveats and Recommendations:**
- Perform fairness audits before deployment.
- Retrain with updated data if distribution shifts.
