# Script to train machine learning model.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pandas import read_csv
from jinja2 import Environment, FileSystemLoader

from ml.data import process_data
from ml.model import train_model
from ml.model import save_model
from ml.model import compute_model_metrics
from ml.model import inference
from ml.model import compute_model_performance_slice
from config import cat_features


# Add code to load in the data.
df = read_csv("./data/census.csv", skipinitialspace=True)

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(df, test_size=0.20)

# Process the train data with the process_data function.
X_train, y_train, encoder, lb = process_data(
    train,
    categorical_features=cat_features,
    label="salary",
    training=True
)

# Process the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb,
)

# Train and save a model.
model = train_model(X_train, y_train)
save_model(model, encoder, lb)


# Generate trained model metrics
predictions = inference(model, X_test)
print(
    pd.DataFrame.from_records(
        test,
        columns=[
            "age",
            "workclass",
            "fnlgt",
            "education",
            "education-num",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "capital-gain",
            "capital-loss",
            "hours-per-week",
            "native-country",
            "salary",
        ],
    )["salary"].value_counts()["<=50K"]
)
print(predictions[:15])
print(type(predictions))
print(classification_report(y_test, predictions))
precision, recal, fbeta = compute_model_metrics(y_test, predictions)

# Compute performance on model slices
with open('slice_ouput.txt', 'w') as file:

    for feature in cat_features:
        performance_df = compute_model_performance_slice(test, feature, y_test, predictions)
        print(f"\n{feature}\n"
              f"--------------\n"
              f"{performance_df}")
        file.write(f'{feature}\n')
        file.write('--------------\n')
        file.write(f'{performance_df}\n\n')

# Generate model card (using jinja)
environment = Environment(loader=FileSystemLoader("src/templates/"))
template = environment.get_template("model_card.md.jinja")
content = template.render(
    precision=precision,
    recal=recal,
    fbeta=fbeta
)

model_card_path = "model_card.md"
with open(model_card_path, 'w') as file:
    file.write(content)
