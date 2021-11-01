#!/usr/bin/env python
"""
Clean raw census data file and outputs artifact
"""
import os
import pandas as pd
import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args: argparse.Namespace):
    input_artifact = os.path.join(os.getcwd(), "starter", "data", args.input_artifact)
    logger.info("Cleaning artifact: %s", input_artifact)
    census_df = pd.read_csv(input_artifact)

    # remove whitespaces from column names
    census_df.columns = [col.strip() for col in census_df.columns]

    output_artifact = os.path.join(os.getcwd(), "starter", "data", args.output_artifact)
    logger.info("Saving cleaned artifact: %s", output_artifact)
    census_df.to_csv(output_artifact, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="A very basic data cleaning")

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="the input artifact to clean",
        required=False,
        default="census.csv"
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="the name for the output artifact",
        required=False,
        default="census_clean.csv"
    )

    arguments = parser.parse_args()
    go(arguments)
