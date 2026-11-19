import numpy as np
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder


def process_data(
    X, categorical_features=[], label=None, training=True, encoder=None, lb=None
):
    """Process the data used in the machine learning pipeline.

    Uses one hot encoding for categorical features and a label binarizer
    for the labels. This can be used in either training or
    inference/validation.

    Note: depending on the type of model used, you may want to add in
    functionality that scales the continuous data.

    Inputs
    ------
    X : pd.DataFrame
        Dataframe containing the features and label. Columns in
        `categorical_features`
    categorical_features: list[str]
        List containing the names of the categorical features (default=[])
    label : str
        Name of the label column in `X`. If None, then an empty array will be
        returned for y (default=None)
    training : bool
        Indicator if training mode or inference/validation mode.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder, only used if training=False.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer, only used if training=False.

    Returns
    -------
    X : np.array
        Processed data.
    y : np.array
        Processed labels if labeled=True, otherwise empty np.array.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained OneHotEncoder if training is True, otherwise returns the
        encoder passed in.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained LabelBinarizer if training is True, otherwise returns the
        binarizer passed in.
    """

    if label is not None:
        y = X[label]
        X = X.drop([label], axis=1)
    else:
        y = np.array([])

    X_categorical = X[categorical_features].values
    X_continuous = X.drop(categorical_features, axis=1)

    if training is True:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        lb = LabelBinarizer()
        X_categorical = encoder.fit_transform(X_categorical)
        y = lb.fit_transform(y.values).ravel()
    else:
        X_categorical = encoder.transform(X_categorical)
        try:
            y = lb.transform(y.values).ravel()
        # Catch the case where y is None because we're doing inference.
        except AttributeError:
            pass

    X = np.concatenate([X_continuous, X_categorical], axis=1)
    return X, y, encoder, lb


def clean_data(df, label="salary", missing="drop", missing_tokens=("?",)):
    """Clean the raw census data. Safe to run on already-clean data (idempotent).

    Steps
    -----
    1. Strip whitespace from column names and text values.
    2. Treat `missing_tokens` (default "?") and empty strings as missing.
    3. Drop rows with a missing label; normalise label text (e.g. "<=50K." -> "<=50K").
    4. Handle missing features: missing="drop" removes those rows,
       missing="unknown" fills text columns with "Unknown" (numeric NaNs are still dropped).
    5. Remove exact duplicate rows (so copies cannot end up in both train and test).
    """
    if missing not in ("drop", "unknown"):
        raise ValueError("missing must be 'drop' or 'unknown'")
    df = df.copy()
    df.columns = df.columns.str.strip()

    text_cols = df.select_dtypes(include=["object", "string"]).columns
    for c in text_cols:
        df[c] = df[c].str.strip()
    df = df.replace(list(missing_tokens) + [""], np.nan)

    df = df.dropna(subset=[label])
    df[label] = df[label].astype(str).str.strip().str.rstrip(".")

    if missing == "unknown":
        df[text_cols] = df[text_cols].fillna("Unknown")
    df = df.dropna()

    df = df.drop_duplicates().reset_index(drop=True)
    return df
