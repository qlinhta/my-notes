import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
import warnings

from scipy.stats import shapiro

warnings.filterwarnings('ignore')


# Correlation between features of X and y, ascending order
def correlation_with_target(df, target):
    """
    Correlation is a statistical measure that expresses the extent to which two variables are linearly related.
    :param df: dataframe
    :param target: target column
    :return: correlation between features of X and y, ascending order
    """
    corr = df.corrwith(target).sort_values(ascending=False)
    sns.barplot(x=corr, y=corr.index)
    plt.show()


# Check skewness of each column in dataframe
def check_skewness(df):
    """
    Skewness is a measure of the asymmetry of the probability distribution of a real-valued random variable about its mean.
    The skewness value can be positive or negative, or even undefined.
    :param df: dataframe
    :return: skewness of each column in dataframe
    """
    for i in df.columns:
        # If skewness is positive, then it is right skewed
        if df[i].skew() > 0:
            print(i, 'is right skewed')
        # If skewness is negative, then it is left skewed
        elif df[i].skew() < 0:
            print(i, 'is left skewed')
        # If skewness is 0, then it is normally distributed
        else:
            print(i, 'is normally distributed')
    df.skew().plot(kind='bar', figsize=(10, 5))
    plt.show()
    return df.skew()


# Check normality of each column in dataframe
def check_normality(X):
    """
    Normality is a property of a probability distribution of a random variable whereby it may be assumed that the variable
    :param X: dataframe
    :return: normality of each column in dataframe
    """
    normality = []
    for i in X.columns:
        stat, p = shapiro(X[i])
        normality.append(p)
    result = pd.DataFrame({'p-value': normality}, index=X.columns)
    result['normality'] = result['p-value'].apply(lambda x: 'Gaussian' if x > 0.05 else 'Not Gaussian')
    return result


# Chebyshev's inequality for each feature of X
def chebyshev_inequality(df):
    # Chebyshev's inequality for each feature of X
    # Chebyshev's inequality: P(|X - μ| ≥ kσ) ≤ 1/k^2 for k > 1
    # P(|X - μ| ≥ kσ) is the probability that the random variable X is
    # at least k standard deviations away from the mean μ

    for i in df.columns:
        print(i, 'mean:', df[i].mean(), 'std:', df[i].std())
        print('Chebyshev\'s inequality:', df[i].mean() - 2 * df[i].std(), '<=', i, '<=', df[i].mean() + 2 * df[i].std())
        print('P(|X - μ| ≥ kσ):', stats.norm.cdf(df[i].mean() - 2 * df[i].std(), df[i].mean(), df[i].std()))


# Estimate interval of confidence for each feature of X

def interval_confidence(data, alpha):
    if len(data) <= 1:
        raise ValueError("data must have at least two values")
    if alpha < 0 or alpha > 1:
        raise ValueError("alpha must be between 0 and 1")
    z = stats.norm.ppf(1 - alpha / 2)
    mean = data.mean()
    std = data.std()
    n = len(data)
    return mean - z * std / np.sqrt(n), mean + z * np.sqrt(n)


def estimate_interval_of_confidence(df):
    """
    :param df: dataframe
    :return: interval of confidence for each feature of X
    """

    # Estimate interval confidence for each feature of X with alpha = 0.05
    for i in df.columns:
        print(i, 'mean:', df[i].mean(), 'std:', df[i].std())
        print('Interval confidence:', interval_confidence(df[i], 0.05))


# Local Outlier Factor
def local_outlier_factor(df, n_neighbors, contamination, threshold):
    """
    Local Outlier Factor (LOF) is an unsupervised anomaly detection method which computes the local density deviation of a given data point with respect to its neighbors.
    It considers as outlier the samples that have a substantially lower density than their neighbors.
    :param df: dataframe
    :param n_neighbors: Number of neighbors to use by default for k_neighbors queries.
    :param contamination: The amount of contamination of the data set, i.e. the proportion of outliers in the data set.
    :param threshold: The threshold to use when converting raw outlier scores to binary labels.
    :return: threshold, scores, index_of_outliers: The local outlier factor of each input samples. The lower, the more normal
    """
    from sklearn.neighbors import LocalOutlierFactor
    try:
        lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
        y_pred = lof.fit_predict(df)
        scores = lof.negative_outlier_factor_
        # with threshold
        index_of_outliers = np.where(scores < threshold)
        index_of_outliers = list(index_of_outliers[0])
    except Exception as e:
        print('Error in Local Outlier Factor: ', e)
        return None, None, None
    return threshold, scores, index_of_outliers
