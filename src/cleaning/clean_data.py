"""
Author: Eric Koch
Date Created: 2024-05-30

This modules loads raw census data and cleans it for further processing
"""
import pandas as pd
import os

def load_raw_census_data():
    data_path = os.path.join(os.path.dirname(os.path.abspath(__name__)),"data/census.csv")
    return pd.read_csv(data_path)

def clean_data(df):
    # Drop duplicates
    df = df.drop_duplicates()
    # Remove l/r whitespace in columns, replace spaces
    df = df.rename(mapper=lambda x: x.lstrip().rstrip().lower().replace(' ', '_'), axis=1)
    # Remove all spaces
    df = df.replace(' ', '', regex=True)
    return df

def save_new_data(df: pd.DataFrame):
    output_path = os.path.join(os.path.dirname(os.path.abspath(__name__)),"src/cleaning/clean_data.csv")
    df.to_csv(output_path)

if __name__ == "__main__":
    df = load_raw_census_data()
    df = clean_data(df)
    print(df.head())
    save_new_data(df)