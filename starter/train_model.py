"""Script to train machine learning model."""

from sklearn.model_selection import train_test_split
from starter.ml.data import process_data, import_data
from starter.ml.model import train_model
from pickle import dump
from yaml import load, Loader


def train_ml_model():
    """ Train and save ML model create to predict the salary of a person based on their demographic features."""
    # Load the path file containing the paths to the data and model.
    paths_dict = load(open("path.yml"), Loader=Loader)
    MODEL_PATH = paths_dict["MODEL_PATH"]
    DATA_PATH = paths_dict["DATA_PATH"]
    ENCODER_PATH = paths_dict["ENCODER_PATH"]
    LB_PATH = paths_dict["LB_PATH"]

    # load in the data.
    data = import_data(DATA_PATH)

    train, test = train_test_split(data, test_size=0.20)

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    print("Processing data...")
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    # Proces the test data with the process_data function.
    X_test, y_test, _, _ = process_data(
        test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb)

    # Train and save a model.
    print("Training model...")
    model = train_model(X_train, y_train)

    with open(MODEL_PATH, "wb") as f:
        dump(model, f)

    # Save the encoder.
    with open(ENCODER_PATH, "wb") as f:
        dump(encoder, f)

    # Save the label encoder.
    with open(LB_PATH, "wb") as f:
        dump(lb, f)

    print("SUCCESS: Model trained and saved.")

    # print("Evaluating model...")
    # # Compute the model metrics by categorical value: Education feature.
    # preds = inference(model, X_test)
    # slice_data(preds, y_test, test, "education")

    # # Compute the model metrics.
    # precision, recall, fbeta = compute_model_metrics(preds, y_test)

    # return precision, recall, fbeta
