import numpy as np
import pandas as pd


# MinMaxScaler
def min_max_scaler(cols):
    """
    MinMaxScaler
    :param cols:
    :return:
    """
    for col in cols:
        min_val = np.min(col)
        max_val = np.max(col)
        col = (col - min_val) / (max_val - min_val)
    return cols


# StandardScaler
def standard_scaler(cols):
    """
    StandardScaler
    :param cols:
    :return:
    """
    for col in cols:
        mean = np.mean(col)
        std = np.std(col)
        col = (col - mean) / std
    return cols


# RobustScaler
def robust_scaler(cols):
    """
    RobustScaler
    :param cols:
    :return:
    """
    for col in cols:
        median = np.median(col)
        q1 = np.percentile(col, 25)
        q3 = np.percentile(col, 75)
        col = (col - median) / (q3 - q1)
    return cols


# MaxAbsScaler
def max_abs_scaler(cols):
    """
    MaxAbsScaler
    :param cols:
    :return:
    """
    for col in cols:
        max_val = np.max(col)
        col = col / max_val
    return cols


# PowerTransformer
def power_transformer(cols):
    """
    PowerTransformer
    :param cols:
    :return:
    """
    for col in cols:
        mean = np.mean(col)
        std = np.std(col)
        col = (col - mean) / std
    return cols


# QuantileTransformer
def quantile_transformer(cols):
    """
    QuantileTransformer
    :param cols:
    :return:
    """
    for col in cols:
        median = np.median(col)
        q1 = np.percentile(col, 25)
        q3 = np.percentile(col, 75)
        col = (col - median) / (q3 - q1)
    return cols


# Normalizer
def normalizer(cols):
    """
    Normalizer
    :param cols:
    :return:
    """
    for col in cols:
        norm = np.linalg.norm(col)
        col = col / norm
    return cols


# Encoding categorical features by mapping
def encode_with_dictionary(df, columns, dictionary):
    '''
    :param df: dataframe
    :param columns: columns to be encoded
    :param dictionary: dictionary of mapping
    :return: encoding categorical features by mapping
    '''
    for column in columns:
        df[column] = df[column].map(dictionary)
    return df


# Encode categorical features by one-hot encoding
def encode_with_one_hot(df, columns):
    '''
    :param df: dataframe
    :param columns: columns to be encoded
    :return: encoding categorical features by one-hot encoding
    '''
    for column in columns:
        df = pd.concat([df, pd.get_dummies(df[column], prefix=column)], axis=1)
        df.drop(column, axis=1, inplace=True)
    return df


# Encode categorical features by label encoding
def encode_with_label(df, columns):
    '''
    :param df: dataframe
    :param columns: columns to be encoded
    :return: encoding categorical features by label encoding
    '''
    for column in columns:
        df[column] = df[column].astype('category')
        df[column] = df[column].cat.codes
    return df


# Normalize the data
def normalize(df, columns):
    '''
    :param df: dataframe
    :param columns: columns to be normalized
    :return: normalize the data
    '''
    for column in columns:
        df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())
    return df


# Standardize the data
def standardize(df, columns):
    '''
    :param df: dataframe
    :param columns: columns to be standardized
    :return: standardize the data
    '''
    for column in columns:
        df[column] = (df[column] - df[column].mean()) / df[column].std()
    return df
