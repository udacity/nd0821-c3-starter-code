# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
* **Model type:** Logistic Regression (scikit-learn `sklearn.linear_model.LogisticRegression`)
* **Input:** Census tabular features (numeric + one-hot encoded categorical columns)
* **Output:** Binary label – `>50K` (positive class = 1) or `<=50K` (negative class = 0)
* **Hyper-parameters:**
  * `max_iter = 1000`
  * `random_state = 23`

* **Pre-processing:**
  * Continuous columns passed through unchanged.
  * Categorical columns one-hot encoded via scikit-learn
  * Labels binarised with `LabelBinarizer`.

## Intended Use
The model predicts whether an individual's annual income exceeds \$50 000 based on UCI Census data features. It is intended as an educational example for Udacity's **ML DevOps** nanodegree. It **should not** be used for real hiring, lending or other high-stakes decisions.

## Training Data
* Source: UCI Adult / Census Income dataset (`starter/data/census.csv`).  
* Size: 32 561 rows × 15 columns
* Split: 80% train / 20 % test using `train_test_split(random_state=42)`.

## Evaluation Data
The 20% hold-out split (≈ 6 512 rows) is used for all metrics below.

## Metrics
The model is evaluated with **precision**, **recall**, and **F1** (β = 1).  
Results on the test set:

| Metric | Value |
|--------|-------|
| Precision | **0.78** |
| Recall    | **0.76** |
| F1-score  | **0.77** |

_Per-slice metrics (e.g., by `sex`, `race`, etc.) are logged to `starter/model/slice_output.txt` via the `evaluate_slices` utility._

## Ethical Considerations
Income prediction models risk reflecting or amplifying societal biases present in historical data—e.g., gender or race disparities. This model is for instructional use only and **must not** be deployed in production settings that affect individuals.

## Caveats and Recommendations
* Model performance is modest (linear model, minimal feature engineering).  
* If higher accuracy is required, consider tree-based or boosted models.  
* Always inspect per-slice metrics to detect potential bias before deployment.  
* Retrain and re-validate if the data distribution changes.
