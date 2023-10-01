#!/usr/bin/env -S python3 -i

###################
# Imports
###################
import numpy as np
import pandas as pd
import logging

from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    LabelBinarizer
)

###################
# Coding
###################

# get logging properties
# info see: https://realpython.com/python-logging-source-code/
logger = logging.getLogger(__name__)


def load_data(path):
    """
    Returns read-in csv data from path string param as dataframe, if available.
    If not raises FileNotFoundError exception.
    """
    assert isinstance(path, str), 'load_data() path param must be a string'
    assert Path(path).exists(), f'dataset of given path does not exist: {path}'
    logger.info('Census data path exists: %s', path)
    try:
        df = pd.read_csv(path)
        logger.info('Census dataset: %s samples with %s features each', df.shape[0], df.shape[1])
        return df
    except Exception as e:
        logger.exception("Census data not loaded")
        raise FileNotFoundError(path) from e


def get_x_data(df, label) -> pd.DataFrame:
    """
    Returns the X dataframe without target label.
    """
    logger.info("Extracting target label column from dataframe")
    return df.drop(columns=[label])


def get_y_target(df, label) -> pd.DataFrame:
    """
    Returns the target column of the dataframe.
    """
    logger.info("Returns dataframes target label column only")
    return df[label]


def get_cat_features(df_X_data) -> pd.DataFrame:
    """
    Returns categorical features list of dataframe not including target label.
    """
    cat_cols_selector = selector(dtype_include=object)
    cat_cols = cat_cols_selector(df_X_data)
    logger.info("Get categorical features: %s", cat_cols)

    return cat_cols


def get_num_features(df_X_data) -> pd.DataFrame:
    """
    Returns binary features list of dataframe not including target label.
    """
    num_cols_selector = selector(dtype_exclude=object)
    num_cols = num_cols_selector(df_X_data)
    logger.info("Get numeric features: %s", num_cols)

    return num_cols


def get_bin_features(df_X_data) -> pd.DataFrame:
    """
    Returns bineric, numerical features list of dataframe not including target label.
    """
    bin_cols = [col for col, val in df_X_data.nunique().items() if val == 2]
    logger.info("Get binary features with values like 0/1 or Yes/No or False/True: %s", bin_cols)

    return bin_cols


def get_column_transformer(cat_cols, num_cols, bin_cols, scaling=None) -> ColumnTransformer:
    """
    Returns a column transformer for categorical, numerical and binary
    features doing One-Hot-Encoding for cat ones,
    standard scaling to numerical if scaling is True
    and imputation to each at the beginning of the transformer pipeline.
    This concept is used later on for the ML pipeline.

    Regarding the OHE:
    Using a dummy variable encoding with drop set to 'first' and dtype 'int'.
    Unknown values are handled with 'ignore'. Sparse is set to False.
    """

    # So, we do the simple imputation and scaling is set to True.
    # In general, it is proposed to scale datasets num features. For cat, there is ohe.

    logger.info("Create column transformer for cat, num and bin preprocessing with %s", scaling)

    if scaling:
        num_transformer = Pipeline(steps=[
            ('imp', SimpleImputer(missing_values=np.nan, strategy='median')),
            ('scale', StandardScaler()),
        ])
    else:
        num_transformer = Pipeline(steps=[
            ('imp', SimpleImputer(missing_values=np.nan, strategy='median')),
        ])

    cat_transformer = Pipeline(steps=[
        ('imp', SimpleImputer(missing_values=np.nan, strategy='most_frequent')),
        ('ohe', OneHotEncoder(dtype='int', drop='first', handle_unknown="ignore", sparse=False)),
    ])

    # transformer without scaling has to be at the end of the process, or parts will be scaled too
    bin_transformer = Pipeline(steps=[
        ('imp', SimpleImputer(missing_values=np.nan, strategy='most_frequent')),
    ])

    # modification of the dataframe columns
    preprocessor = ColumnTransformer(
        transformers = [
            ('num_transf', num_transformer, num_cols),
            ('cat_transf', cat_transformer, cat_cols),
            ('bin_transf', bin_transformer, bin_cols),
        ],
        # remainder='passthrough', # unknown columns
        n_jobs=-1
    )

    return preprocessor


def map_country_to_continent(cntry, config_file) -> str:
    '''
    Converts the given country name to an associated continent if available,
    items being part of the non_converting_countries list are returned unchanged,
    'Other' is returned if no matching group has been found.
    '''
    str_continent = 'Other'
    if cntry in list(config_file['etl']['non_converting_countries']):
        return cntry
    if cntry in list(config_file['etl']['north_west_south_europe']):
        return 'North_West_South_Europe'
    if cntry in list(config_file['etl']['south_east_europe']):
        return 'South_East_Europe'
    if cntry in list(config_file['etl']['caribbean']):
        return 'Caribbean'
    if cntry in list(config_file['etl']['central_america']):
        return 'Central_America'
    if cntry in list(config_file['etl']['south_america']):
        return 'South_America'
    if cntry in list(config_file['etl']['south_east_asia']):
        return 'South_East_Asia'
    if cntry in list(config_file['etl']['australia']):
        return 'Australia'
    if cntry in list(config_file['etl']['south_middle_asia']):
        return 'South_Middle_Asia'
    if cntry in list(config_file['etl']['east_asia']):
        return 'East_Asia'
    # nothing found, so use 'Other'
    return str_continent


def clean_data(df, config_file) -> pd.DataFrame:
    """
    Returns partly preprocessed dataframe including target label
    with general modifications as identified during EDA,
    e.g.
    column label and value changes regarding existing spaces
    or wrong identified column values of specific features
    or removing dupliate rows
    or remove redundant 'education' column
    or remove the 'fnlgt' feature (not having an added value)
    or change target 'salary' column to binary values of
    ' >50K': 1, ' <=50K': 0
    and other replacement actions are listed.

    Out-of-scope:
    regarding the specific column value types,
    imputation, ohe or scaling activities are handled with
    model specific column transformers.

    Additionally done:
    the 'native_country' feature changes of the EDA have been
    implemented for training and inference;
    the 'fnlgt' feature has been removed even its meaning is not
    clear, because it seems to be a kind of index.

    Inputs
    ------
    df : pd.DataFrame
        Dataframe containing the features and target label from original dataset.
    config_file : configuration yaml file content, can be None

    Returns
    -------
    df : pd.DataFrame
        Generally cleaned dataframe with binary, numerical target label.
    """

    # change column labels
    columns = df.columns
    df.columns = [col.strip().replace('-', '_') for col in columns]

    # replacement actions
    df = df.replace(to_replace=[' ?', '?', '? ', ' ? '], value='Other')
    df = df.replace(to_replace=['Others', 'others', 'other', 'Other values (5)'], value='Other')
    df = df.applymap(lambda x: x.strip().replace(' ', '-') if isinstance(x, str) else x)
    df['sex'] = df['sex'].map({'Male': 0, 'Female': 1})

    # native_country modification, creation of new bins
    if config_file:
        df['mod_native_country'] = df['native_country'].map(
            lambda cntry: map_country_to_continent(cntry=cntry, config_file=config_file))
    else:
        df['mod_native_country'] = df['native_country'].copy()

    # change target column 'salary' (remember: spaces are stripped) if available in df
    if 'salary' in df.columns:
        df['salary'] = df['salary'].map({'>50K': 1, '<=50K': 0})

    # remove original columns being modified or unnecessary,
    # only available ones will be removed
    df = df.drop(columns=['native_country', 'education', 'fnlgt'], errors='ignore')

    # remove duplicate rows
    df = df.drop_duplicates(ignore_index=False)

    logger.info("Census dataframe is cleaned; sex and salary are binary numbers")

    return df


def process_data_udacity(X, categorical_features=[], label=None, training=True, encoder=None, lb=None):
    """ Process the data used in the machine learning pipeline.

    Processes the data using one-hot-encoding for the categorical features and a
    label binarizer for the labels. This can be used in either training or
    inference/validation.

    Note: depending on the type of model used, you may want to add in functionality that
    scales the continuous data.
    This function coding from Udacity partly modified by script author.

    Inputs
    ------
    X : pd.DataFrame
        Dataframe containing the features and label. Data must be cleaned already!
    categorical_features: list[str]
        List containing the names of the categorical features (default=[])
    label : str
        Name of the label column in `X`. If None, then an empty array will be returned
        for y (default=None)
    training : bool
        Indicator if training mode or inference/validation mode.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder, only used if training=False.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer, only used if training=False.
        Note: The LabelBinarizer is the right choice for encoding string columns -
        first it translates strings to integers, then it binarizes those integers to bit vectors.

    Returns
    -------
    X : np.array
        Processed data.
    y : np.array
        Processed labels if labeled=True, otherwise empty np.array.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained OneHotEncoder if training is True, otherwise returns the encoder passed in.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained LabelBinarizer if training is True, otherwise returns the binarizer
        passed in.
    """
    logger.info("You called Udacity function: process_data()")

    if label is not None:
        logger.info("Extracting label from data.")
        logger.info("columns: %s", X.columns)
        y = X[label]
        X = X.drop([label], axis=1)
    else:
        y = np.array([])

    X_categorical = X[categorical_features].values
    X_continuous = X.drop(*[categorical_features], axis=1)

    if training:
        logger.info("Training phase: process data.")
        encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        lb = LabelBinarizer()
        X_categorical = encoder.fit_transform(X_categorical)
        y = lb.fit_transform(y.values).ravel()
    else:
        X_categorical = encoder.transform(X_categorical)
        try:
            y = lb.transform(y.values).ravel()
        # Catch the case where y is None because we're doing inference.
        except AttributeError:
            logger.info("Census data: no y labels available, because doing inference")

    X = np.concatenate([X_continuous, X_categorical], axis=1)
    logger.info("Data are processed ...")

    return X, y, encoder, lb


def process_data(df, label=None, scaling=True, config_file=None):
    """ Process the data used for the machine learning classification pipeline.
        Own modified coding for preprocessing compared to basic given one.

    Inputs
    ------
    df : pd.DataFrame
        General dataframe containing the features and target label.
    label : None or string
        Name of target label column in dataframe cleaned_df.
        If None, then an empty array will be returned for y (default=None).
        Regarding our business use case, for training 'salary' is used,
        see config file configuration settings.
    scaling : None or bool True if necessary
        Indicator to know which kind of model condition is used for prediction.
        For some model types scaling of continous features is necessary, for some not.
        Therefore a different column transformer is used for cleaned dataset.
        Note:
        By now, only XGBClassifier is used, there scaling is necessary,
        therefore as default it is set to True;
        could be changed to None, if other appropriate estimators are used as well.
    config_file : configuration yaml file content, can be None

    Returns
    -------
    preproc : scikit-learn ColumnTransformer
        Preprocessor of cleaned dataframe as input for classification pipeline.
    X : np.array
        Processed data.
    y : np.array
        Processed labels if label string is given, otherwise empty np.array.
    """
    logger.info("--- You called script authors function: process_data()")

    # start with general cleaning of dataframe if config file is given
    if config_file:
        df = clean_data(df, config_file)

    # for supervised classification task:
    # separation of target column y and data X
    if label:   # is not None
        logger.info("Extracting target label from cleaned dataframe: %s", label)
        logger.info("Dataframe columns: %s", df.columns)
        y = get_y_target(df, label)
        X = get_x_data(df, label)
    else:
        logger.info("Census data: no y target labels available, because doing inference")
        X = df
        y = np.array([])

    # training / inference workflow depending on classification model type
    # by now: only XGB classifier is implemented;
    # depending on used classifier model we need ohe, scaling etc. or not
    # future toDo: implement a more OO like style with behavioural strategy pattern
    # for more than one classifier type

    cat_cols = get_cat_features(X)
    num_cols = get_num_features(X)
    bin_cols = get_bin_features(X)

    # create preprocessing transformer for ML pipeline
    preproc = get_column_transformer(cat_cols, num_cols, bin_cols, scaling)

    logger.info("Return df preproc column transformer and X, y for classification pipeline.")
    return preproc, X, y
