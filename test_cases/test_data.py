import os
from starter.ml.data import make_dataset

def test_raw_data():
    assert os.path.isfile(os.path.join('data', 'census.csv'))

def test_processed_data():
    assert os.path.isfile(os.path.join('data', 'dataset.csv'))

def test_make_dataset():
    # make datset and verify files
    df_processed = make_dataset('census.csv', 'dataset.csv')

    # test shape
    assert df_processed.shape[1] != 12

    # test '-' names
    hyphen_test = [col for col in df_processed.columns if '-' in col]
    assert len(hyphen_test) == 0

