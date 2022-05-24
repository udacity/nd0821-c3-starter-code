# Script to train machine learning model.
from sklearn.model_selection import train_test_split
from ml.data import load_data, save_data, process_data
from ml.model import save_model, train_model

import os
import yaml

root_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

# Load data
data = load_data(root_path, "clean_census.csv")

# Read categorical_features
with open(os.path.join(root_path, "starter", "constants.yaml"), 'r') as f:
    categorical_features = yaml.safe_load(f)["categorical_features"]

# Split data
train_data, test_data = train_test_split(data, test_size=0.20)

# Save data
save_data(train_data, root_path, "train_census.csv")
save_data(test_data, root_path, "test_census.csv")

# Process training data
X_train, y_train, preprocessor, label_binarizer = process_data(
    train_data, categorical_features=categorical_features, label="salary", training=True
)

# Train model
model = train_model(X_train, y_train)

# Save model along with preprocessor and label_binarizer
save_model(model, root_path, "model.pkl")
save_model(preprocessor, root_path, "preprocessor.pkl")
save_model(label_binarizer, root_path, "label_binarizer.pkl")