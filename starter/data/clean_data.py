import pandas as pd

# Load data
data = pd.read_csv("starter/data/census.csv")
print(data.head())
print(data.info())

# Remove white spaces in column names
data.columns = [col.replace(" ", "") for col in data.columns]

# Remove white spaces in categorical columns
data = data.replace(" ", "")

# Save cleaned data
data.to_csv("starter/data/census_cleaned.csv")
print(data.head())
print(data.info())
