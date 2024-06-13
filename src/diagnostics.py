import json
import subprocess
import sys

import pandas as pd

import logger as appLogger

# Load config.json and get environment variables
with open("config.json", "r") as f:
    config = json.load(f)

logger = appLogger.logger

# dataset_csv_path = os.path.join(config['output_folder_path'])
# output_model_path = os.path.join(config['output_model_path'])
# test_data_filepath = scoring.test_data_filepath

# ##################Function to get model predictions
# def model_predictions(model, dataset_filepath):
#     #read the deployed model and a test dataset, calculate predictions
#     logger.info(f"Calculating model predictions for dataset at {dataset_filepath}...")
#     return scoring.make_predictions(model, dataset_filepath)

# ##################Function to get summary statistics
# def dataframe_summary():
#     #calculate summary statistics here
#     training_df = training.load_training_data()
#     summary_stats = []
#     numeric_columns = training_df.select_dtypes(include=[float, int]).columns

#     for col in numeric_columns:
#         summary_stats.append({
#             'column': col,
#             'mean': training_df[col].mean(),
#             'median': training_df[col].median(),
#             'std': training_df[col].std()
#         })
#     # Formatting the summary
#     formatted_summary = "Diagnostic Dataframe Summary:\n"
#     for i, stats in enumerate(summary_stats):
#         formatted_summary += f"Column '{stats['column']}':\n"
#         formatted_summary += f"  Mean: {stats['mean']}\n"
#         formatted_summary += f"  Median: {stats['median']}\n"
#         formatted_summary += f"  Std: {stats['std']}\n"

#     # Logging the formatted summary
#     logger.info(formatted_summary)
#     return summary_stats

# ##################Function to check for missing data
# def check_missing_data():
#     # Load the configuration
#     training_df = training.load_training_data()

#     # Initialize a list to store the percentages of NA values for each column
#     missing_data_percentages = []

#     total_rows = training_df.shape[0]

#     # Calculate the percentage of NA values for each column
#     for col in training_df.columns:
#         num_missing = training_df[col].isna().sum()
#         percent_missing = (num_missing / total_rows) * 100
#         missing_data_percentages.append(percent_missing)
#     logger.info(f"Data missing data percentages by column:\n\t{missing_data_percentages}")
#     return missing_data_percentages

# ##################Function to get timings
# def execution_time():
#     #calculate timing of training.py and ingestion.py
#     ingestion_time = timeit.timeit(ingestion.ingest_data, number=1)
#     training_time = timeit.timeit(training.train_model, number=1)
#     logger.info(f"Execution Time:\n\tIngestion: {ingestion_time} seconds\n\tTraining: {training_time} seconds")
#     return ingestion_time, training_time

# Function to check dependencies


def outdated_packages_list():
    def get_required_packages(requirements_path):
        logger.info(f"Reading required packages from {requirements_path}")
        with open(requirements_path, "r") as file:
            packages = [line.strip().split("=")[0]
                        for line in file if "=" in line]
        logger.info(f"Found {len(packages)} required packages.")
        return packages

    def get_outdated_packages():
        logger.info("Checking for outdated packages using pip list")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "list", "--outdated"],
            stdout=subprocess.PIPE,
            text=True,
            check=True,
        )
        lines = result.stdout.splitlines()
        outdated_packages = {}
        for line in lines[2:]:  # Skip the header lines
            package, current_version, _arrow, latest_version = line.split()
            outdated_packages[package] = (current_version, latest_version)
        logger.info(
            f"Found {len(outdated_packages)} outdated packages {':' if outdated_packages else ''} \
             {', '.join(outdated_packages.keys())}")
        return outdated_packages

    required_packages = get_required_packages("requirements.txt")
    conda_required_packages = get_required_packages("conda_requirements.txt")
    required_packages = set(required_packages).union(conda_required_packages)
    outdated_packages = get_outdated_packages()

    data = []
    for package in required_packages:
        if package in outdated_packages:
            current_version, latest_version = outdated_packages[package]
            data.append([package, current_version, latest_version])
    active_outdated_packages = [package_info[0] for package_info in data]
    not_used_outdated_packages = set(
        outdated_packages) - set(active_outdated_packages)

    df = pd.DataFrame(
        data,
        columns=[
            "Module",
            "Current Version",
            "Latest Version"])

    if not outdated_packages:
        logger.info("No outdated packages found, returning empty dataframe.")
    else:
        logger.info(
            f"To update packages, run:\npip install --upgrade {' '.join(active_outdated_packages)}\n\
    Then run:\npip freeze > requirements.txt\nThen run:\nconda list --export > conda_requirements.txt\nTo remove \
    unused packages, run:\npip uninstall -y {' '.join(not_used_outdated_packages)}")
        logger.info(
            f"Generated dataframe with {len(df)} outdated packages from requirements.txt"
        )
        logger.info(f"Dataframe preview:\n{df.head()}")
    return df


def run_diagnostics():
    # model = scoring.load_production_model()
    # model_predictions(model, test_data_filepath)
    # dataframe_summary()
    # execution_time()
    # check_missing_data()
    outdated_packages_list()


if __name__ == "__main__":
    run_diagnostics()
