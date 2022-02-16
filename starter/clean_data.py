'''
Cleaning the dataset

Author: Oliver
Date: February 2022

'''
import pandas as pd
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def remove_duplicates(file_pth):
    # Add code to load in the data.
    file_dir, file_name = os.path.split(file_pth)
    df = pd.read_csv(file_pth)

    if (df.duplicated().any()):
        df.drop_duplicates(keep='first', inplace=True)
        df.reset_index(drop=True, inplace=True)
        logger.info("Dropped duplicates...")
        cleaned_pth = str(file_dir+'/cleaned_'+file_name)
        df.to_csv(cleaned_pth)
        logger.info(f"Saved to: {cleaned_pth}")

if __name__ == "__main__":
    try:
        remove_duplicates("./data/census.csv")

    except (Exception) as error:
        print("Main error: %s", error)
    


