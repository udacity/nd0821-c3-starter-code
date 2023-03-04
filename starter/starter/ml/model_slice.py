import pandas as pd
import logging

def slice_metric(df, cat_feature, num_feature):
    """ Function for calculating descriptive stats on slices of dataset."""
    logger.info("-------------------------------------------")
    logger.info(f"Calculate slice performance for {num_feature} with {cat_feature} held fixed value")
    for cls in df[cat_feature].unique():
        df_temp = df[df[cat_feature] == cls]
        mean = df_temp[num_feature].mean()
        stddev = df_temp[num_feature].std()
        logger.info(f"{cat_feature}: {cls}")
        logger.info(f"{num_feature} mean: {mean:.4f}")
        logger.info(f"{num_feature} stddev: {stddev:.4f}\n")

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s", filename="../../model/slice_output.txt")
logger = logging.getLogger()


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

num_features = [
    'age',
    'fnlgt',
    'education-num',
    'capital-gain',
    'capital-loss',
    'hours-per-week'
]

# Read data
data = pd.read_csv('../../data/census.csv')

# calculate slice performance
for cat_feature in cat_features:
    for num_feature in num_features:
        slice_metric(data, cat_feature, num_feature)
