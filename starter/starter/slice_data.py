"""This module include fuction to compute model metrics on data categorical feature held values"""
from aequitas.group import Group
import pandas as pd
import numpy as np


def slice_data(
        preds: np.ndarray,
        y_test: np.ndarray,
        data: pd.DataFrame,
        feature: str) -> None:
    """Slice data by feature and value and save the model metric by holding each value of the feature constant.
    Inputs
    ------
    preds: np.ndarray
        The model predictions.
    y_test: np.ndarray
        The true labels.
    data: pd.DataFrame
        The data used to make the predictions.
    feature: str
        The feature to slice the data by.
    """
    df = data.copy()
    df['score'] = preds
    df["label_value"] = y_test
    df = df[[feature, 'score', 'label_value']]

    # Create a Group() object
    group = Group()
    xtab, idxs = group.get_crosstabs(df)
    recall = np.round(xtab["tp"] / (xtab["tp"] + xtab["fn"]), 2)
    precision = np.round(xtab["tp"] / (xtab["tp"] + xtab["fp"]), 2)
    f1_score = np.round(2 * (precision * recall) / (precision + recall), 2)
    values = xtab["attribute_value"]

    with open('slice_output.txt', 'w') as f:
        f.write(f"Model performance by the feature: {feature}\n")
        for i in range(len(values)):
            f.write(
                f"Value: {values[i]} \t Recall: {recall[i]} \t Precision: {precision[i]} \t F1_score: {f1_score[i]}\n")
