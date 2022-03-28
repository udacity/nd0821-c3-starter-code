import pytest
import pandas as pd


@pytest.fixture(scope="session")
def data() -> pd.DataFrame:
    return pd.read_csv("data/census_test.csv")
