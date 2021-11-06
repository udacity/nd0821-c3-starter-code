# Script to train machine learning model.
import logging
import argparse

from sklearn.model_selection import train_test_split
import pandas as pd
from joblib import dump

# Add the necessary imports for the starter code.
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger(__name__)


def go(args: argparse.Namespace):
    logger.info("Loading data... %s", args.input_artifact)
    data = pd.read_csv(args.input_artifact)

    # Optional enhancement, use K-fold cross validation instead of a train-test split.
    logger.info('train test split data with shape: %s, test_size: %.2f', data.shape, args.test_size)
    train, test = train_test_split(data, test_size=args.test_size)

    logger.info('save test data to: %s', args.test_output_artifact)
    test.to_csv(args.test_output_artifact, index=False)

    logger.info('Process the test data with the process_data function.')
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
    # Process the test data with the process_data function.
    target_name = args.target_name
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label=target_name, training=True
    )

    logger.info('Train and save a model.')
    # Train and save a model.
    ml_model = train_model(X_train, y_train)

    logger.info('Evaluate model')
    X_test, y_test, _, _ = process_data(test,
                                        categorical_features=cat_features,
                                        label=target_name,
                                        training=False,
                                        encoder=encoder,
                                        lb=lb
                                        )
    preds = inference(model=ml_model, X=X_test)
    precision, recall, fbeta = compute_model_metrics(y=y_test, preds=preds)
    logger.info('precision %.2f, recall: %.2f, fbeta: %.2f', precision, recall, fbeta)

    # Optional enhancement, save the model to a file.
    logger.info('Save the model to: %s', args.model_artifact)
    dump(ml_model, args.model_artifact)
    logger.info('Save encoder to: %s', args.encoder_artifact)
    dump(encoder, args.encoder_artifact)
    logger.info('Save label_binarizer to: %s', args.lb_artifact)
    dump(lb, args.lb_artifact)
    logger.info('Done!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model.")
    parser.add_argument("--input_artifact",
                        type=str,
                        help="Train/Test data file",
                        default='./starter/data/census_clean.csv',
                        required=False)
    parser.add_argument("--test_output_artifact",
                        type=str,
                        help="Test data file for evaluation",
                        default='./starter/data/census_clean_test.csv',
                        required=False)

    parser.add_argument("--test_size",
                        type=float,
                        help="Test size",
                        default=0.2,
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
