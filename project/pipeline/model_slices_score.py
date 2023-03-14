import pandas as pd
from ml.data import process_data
from ml.model import compute_model_metrics, inference
import joblib

df = pd.read_csv("../data/clean_census.csv")
print(df.head(5))

model = joblib.load("../model/rfc_model.pkl")
lb = joblib.load("../model/lb.pkl")
encoder = joblib.load("../model/encoder.pkl")

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

""" Function for calculating descriptive stats on slices of the Census dataset."""
with open('model_slices_score.txt', 'w') as f:
    for category in cat_features:
        for cls in df[category].unique():
            df_temp = df[df[category] == cls]

            X_test, y_test, _, _ = process_data(df_temp, categorical_features=cat_features,
                                      label="salary", training=False,
                                      encoder=encoder, lb=lb)

            preds = inference(model, X_test)
            precision, recall, fbeta = compute_model_metrics(y_test, preds)
            f.write(f"Category: {category}, Class: {cls}, Precision: {precision}, Recall: {recall}, Fbeta: {fbeta}\n")