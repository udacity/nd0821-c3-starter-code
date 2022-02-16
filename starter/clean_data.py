'''
Cleaning the dataset

Author: Oliver
Date: February 2022

'''
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

# Add code to load in the data.
df = pd.read_csv("./data/census.csv")

if (df.duplicated().any()):
    df.drop_duplicates(keep='first', inplace=True)
    df.reset_index(drop=True, inplace=True)
    logger.info("dropped duplicates")
    df.to_csv("./data/cleaned_census.csv")
    


