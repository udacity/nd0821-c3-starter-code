import argparse
import json
import logging

from collections import defaultdict
import numpy
import numpy as np
from typing import List

import pandas as pd
from joblib import load
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer

from ml.model import compute_model_metrics, inference
from ml.data import process_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def go(args: argparse.Namespace):
    """
    Mostly a cut-n-past from the train_model.py script so that we can call the scoring functions.
    """
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

    test_df = pd.read_csv(args.input_artifact)
    ml_model = load(args.model_artifact)
    encoder = load(args.encoder_artifact)
    lb = load(args.lb_artifact)
    X_test, y_test, _, _ = process_data(test_df,
                                        categorical_features=cat_features,
                                        label=args.target_name,
                                        training=False,
                                        encoder=encoder,
                                        lb=lb
                                        )
    preds = inference(model=ml_model, X=X_test)
    precision, recall, fbeta = compute_model_metrics(y=y_test, preds=preds)
    logger.info('dump metrics to scores file: %s', args.scores_file)
    dump_metric_scores(args.scores_file, precision, recall, fbeta)

    logger.info('performance slicing to file: %s', args.slicing_scores_file)
    performance_slicing(scores_file=args.slicing_scores_file, model=ml_model, X=test_df, label=args.target_name,
                        cat_features=cat_features, encoder=encoder, lb=lb)


def dump_metric_scores(scores_file: str, precision: float, recall: float, fbeta: float):
    with open(scores_file, "w") as fd:
        json.dump(obj={"precision": precision, "recall": recall, "fbeta": fbeta},
                  fp=fd,
                  indent=4
                  )


def performance_slicing(scores_file: str, model, X: pd.DataFrame, label: str, cat_features: List[str],
                        encoder: OneHotEncoder, lb: LabelBinarizer):
    metrics = defaultdict(list)
    for feature in cat_features:
        for name in X[feature].unique():
            filtered_df = X[X[feature] == name]
            X_test, y_test, _, _ = process_data(X=filtered_df,
                                                categorical_features=cat_features,
                                                label=label,
                                                training=False,
                                                encoder=encoder,
                                                lb=lb
                                                )
            preds = inference(model, X_test)
            precision, recall, fbeta = compute_model_metrics(y=y_test, preds=preds)
            metrics[feature].append({"feature_value": name, "precision": precision, "recall": recall, "fbeta": fbeta})

    with open(scores_file, "w") as fd:
        json.dump(obj=metrics, fp=fd, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a model')

    parser.add_argument("--input_artifact",
                        type=str,
                        help="Train/Test data file",
                        default='./starter/data/census_clean_test.csv',
                        required=False)

    parser.add_argument("--scores_file",
                        type=str,
                        help="File to write scores to",
                        default='./starter/model/scores.json',
                        required=False)

    parser.add_argument("--slicing_scores_file",
                        type=str,
                        help="File to write slicing scores to",
                        default='./starter/model/slicing_scores.json',
                        required=False)

    parser.add_argument("--target_name",
                        type=str,
                        help="Target variable name",
                        default='salary',
                        required=False)

    parser.add_argument("--model_artifact",
                        type=str,
                        help="Model artifact",
                        default='./starter/model/census_classifier.joblib',
                        required=False)

    parser.add_argument("--encoder_artifact",
                        type=str,
                        help="Encoder artifact",
                        default='./starter/model/census_encoder.joblib',
                        required=False)

    parser.add_argument("--lb_artifact",
                        type=str,
                        help="Label encoder artifact",
                        default='./starter/model/census_lb.joblib',
                        required=False)
    arguments = parser.parse_args()
    go(arguments)
