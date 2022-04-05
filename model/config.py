TEST_SPLIT_SIZE = 0.2
TARGET = "salary"
DATA_PATH = "./data/census_cleaned.csv"
MODEL_PATH = "./model/classifier.pkl"
METRICS_OUTPUT_PATH = "./model/slice_output.txt"

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