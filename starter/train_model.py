# Script to train machine learning model.
import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
from ml.data import make_dataset, process_data
from ml.model import train_model, compute_model_metrics, inference, performance_on_model_slices

# Fist we generate the processed dataset.csv from the raw census.csv
make_dataset('census.csv', 'dataset.csv')

# Then, we load the processed datset
data = pd.read_csv(os.path.join('data', 'dataset.csv'))

# Optional enhancement, use K-fold cross validation instead of a train-test
# split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
]

# Proces the train data with the process_data function.
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb,
)

# Save encoder to file
with open(os.path.join('model', 'encoder_dtc.pkl'), 'wb') as file:
    pickle.dump(encoder, file)

# Save lb to file
with open(os.path.join('model', 'lb_dtc.pkl'), 'wb') as file:
    pickle.dump(lb, file)

# Train model
model = train_model(X_train=X_train, y_train=y_train)

# Compute model metrics
precision, recall, fbeta = compute_model_metrics(
    y=y_test, preds=inference(model, X=X_test)
)
print(f'precision: {precision}')
print(f'recall: {recall}')
print(f'fbeta: {fbeta}')

# Computes performance on model slices
performance_on_model_slices(test,
                            cat_features=cat_features,
                            model=model,
                            encoder=encoder,
                            lb=lb)
