import pytest
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def pytest_addoption(parser):
    parser.addoption("--raw_data", action="store")

    
@pytest.fixture(scope="session")
def data(request):

    file_path = request.config.option.raw_data
    logger.info(f"MyPara: {file_path}")

    if file_path is None:
        pytest.fail("--file missing on command line")

    raw_data = pd.read_csv(file_path)

    return raw_data