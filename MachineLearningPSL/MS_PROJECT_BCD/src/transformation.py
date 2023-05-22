import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import missingno as ms
import matplotlib.pyplot as plt
import warnings

from scipy.stats import shapiro

warnings.filterwarnings('ignore')


# Log transformation for each column
def log_transformation(df, columns):
    '''
    Log transformation is a transformation that is used to make highly skewed distributions less skewed.
    :param df: dataframe
    :param columns: columns to be transformed
    :return: log transformation for each column
    '''
    # Check that the columns are in the DataFrame
    if not set(columns).issubset(df.columns):
        raise ValueError("One or more columns not found in the dataframe.")
    
    # Check that the columns are numeric
    if not all(df[col].dtype in (int, float) for col in columns):
        raise ValueError("One or more columns are not numeric.")
    
    # Log transform the columns
    for column in columns:
        df[column + '_log'] = np.log(df[column])
        df.drop(column, axis=1, inplace=True)
    return df


def test_log_transform(df):
    for column in df.columns:
        data_log = np.log(np.abs(df[column]))
        if np.isnan(data_log).any() or np.isinf(data_log).any():
            print(column, 'has inf values, cannot be plotted')
        else:
            fig, axs = plt.subplots(nrows=1, ncols=2)
            axs[0].hist(df[column], edgecolor='black')
            axs[1].hist(data_log, edgecolor='black')
            axs[0].set_title('Original Data')
            axs[1].set_title('Log-Transformed Data')
            plt.title('Log Transformation of ' + column)
            plt.show()
        stat, p = shapiro(data_log)
        print('Statistics=%.3f, p=%.3f' % (stat, p))
        if p > 0.05:
            print('Probably Gaussian')
        else:
            print('Probably not Gaussian')


# Square root transformation for each column
def square_root_transformation(df, columns):
    '''
    Square root transformation is a transformation that is used to make highly skewed distributions less skewed.
    :param df: dataframe
    :param columns: columns to be transformed
    :return: square root transformation for each column
    '''
    try:
        for column in columns:
            # Add a new column to the dataframe
            df[column + '_sqrt'] = np.sqrt(df[column])
            # Drop the original column
            df.drop(column, axis=1, inplace=True)
        return df
    except Exception as e:
        print(e)
        return df


def test_square_root_transform(df):
    for column in df.columns:
        data_sqrt = np.sqrt(np.abs(df[column]))
        if np.isnan(data_sqrt).any() or np.isinf(data_sqrt).any():
            print(column, 'has inf values, cannot be plotted')
        else:
            fig, axs = plt.subplots(nrows=1, ncols=2)
            axs[0].hist(df[column], edgecolor='black')
            axs[1].hist(data_sqrt, edgecolor='black')
            axs[0].set_title('Original Data')
            axs[1].set_title('Square Root-Transformed Data')
            plt.title('Square Root Transformation of ' + column)
            plt.show()
        stat, p = shapiro(data_sqrt)
        print('Statistics=%.3f, p=%.3f' % (stat, p))
        if p > 0.05:
            print('Probably Gaussian')
        else:
            print('Probably not Gaussian')


# Cube root transformation for each column
def cube_root_transformation(df, columns):
    '''
    Cube root transformation is a transformation that is used to make highly skewed distributions less skewed.
    :param df: dataframe
    :param columns: columns to be transformed
    :return: cube root transformation for each column
    '''
    if not isinstance(df, pd.DataFrame):
        raise TypeError('df must be a pandas dataframe')
    if not isinstance(columns, list):
        raise TypeError('columns must be a list')
    if columns == []:
        raise ValueError('columns must not be an empty list')
    for column in columns:
        # Add a new column to the dataframe
        df[column + '_cbrt'] = np.cbrt(df[column])
        # Drop the original column
        df.drop(column, axis=1, inplace=True)
    return df


# Test cube root transformation
def test_cube_root_transform(df):
    for column in df.columns:
        data_cbrt = np.cbrt(np.abs(df[column]))
        if np.isnan(data_cbrt).any() or np.isinf(data_cbrt).any():
            print(column, 'has inf values, cannot be plotted')
        else:
            fig, axs = plt.subplots(nrows=1, ncols=2)
            axs[0].hist(df[column], edgecolor='black')
            axs[1].hist(data_cbrt, edgecolor='black')
            axs[0].set_title('Original Data')
            axs[1].set_title('Cube Root-Transformed Data')
            plt.title('Cube Root Transformation of ' + column)
            plt.show()
        stat, p = shapiro(data_cbrt)
        print('Statistics=%.3f, p=%.3f' % (stat, p))
        if p > 0.05:
            print('Probably Gaussian')
        else:
            print('Probably not Gaussian')

