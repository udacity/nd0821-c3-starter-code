import datetime
import platform
from pathlib import Path

import sklearn


def generate_model_card(
    slices,
    overall,
    min_n,
    data,
    n_raw,
    train,
    test,
    cat_features,
    data_path,
    card_path=None,
    label="salary",
):
    if not card_path:
        data_path = Path(__file__).resolve().parents[1] / "data" / "census.csv"

    best = slices[slices.n >= min_n].sort_values("fbeta")

    worst, top = best.head(3), best.tail(3)

    def fmt(df_):
        return "\n".join(
            (
                f"- `{r.feature}` = **{r.value}** (n={r.n}): "
                f"precision {r.precision:.2f},"
                f"recall {r.recall:.2f},"
                f"F1 {r.fbeta:.2f}"
            )
            for r in df_.itertuples()
        )

    card = f"""# Model Card

## Model Details
- **Author:** Paweł Bojemski
- **Date:** {datetime.date.today().isoformat()}
- **Model type:** Random Forest classifier (`sklearn.ensemble.RandomForestClassifier`)
- **Hyperparameters:** n_estimators=100, random_state=42 (all others default)
- **Versions:** scikit-learn {sklearn.__version__}, Python {platform.python_version()}
- **Preprocessing:** `OneHotEncoder(handle_unknown="ignore")` on categorical features, `LabelBinarizer` on the label. Both are saved alongside the model in `model/`.

## Intended Use
Predict whether a person's annual income exceeds $50K from census attributes.
Intended for educational purposes (learning an ML deployment workflow). Not for decisions about real people.

## Training Data
- Source: clean census income dataset (`{data_path}`)
- Cleaning: stripped whitespace, treated `?` as missing (strategy: `drop`), normalised label text, removed duplicates ({n_raw:,} raw rows -> {len(data):,} clean rows)
- Rows: {len(data):,}; 80% used for training ({len(train):,} rows), stratified by label
- Features: {len(data.columns) - 1} total, of which {len(cat_features)} categorical: {", ".join(cat_features)}
- Label: `{label}` (`<=50K` vs `>50K`), positive-class rate {(data[label] == ">50K").mean():.1%}

## Evaluation Data
20% hold-out split ({len(test):,} rows), same preprocessing using encoders fitted on the training split only.

## Metrics
| Metric | Value |
|---|---|
| Precision | {overall["precision"]:.3f} |
| Recall | {overall["recall"]:.3f} |
| F1 | {overall["fbeta"]:.3f} |

Slice metrics for every categorical feature are in `slice_output.txt`
(slices with n >= {min_n}).

Weakest slices (by F1):
{fmt(worst)}

Strongest slices (by F1):
{fmt(top)}

## Ethical Considerations
The data contains sensitive attributes (race, sex, native country, marital status).
Performance differs across groups (see slices above), so predictions may be systematically less reliable for some groups.
Slices with few examples give noisy estimates. The data reflects historical societal biases around income, and the model can reproduce them.

## Caveats and Recommendations
- The data is old and may not reflect today's population or incomes.
- No hyperparameter tuning or class-imbalance handling was done. Precision is {overall["precision"]:.2f} vs recall {overall["recall"]:.2f} for the `>50K` class, so the model {"misses more high earners than it falsely flags" if overall["recall"] < overall["precision"] else "flags more people as high earners than it misses"}.
- Not suitable for hiring, lending, or any decision affecting real people without a fairness audit.
- Next steps: tune hyperparameters, try class weights, and evaluate fairness metrics across groups.
"""  # noqa: E501
    with open(card_path, "w") as f:
        f.write(card)
