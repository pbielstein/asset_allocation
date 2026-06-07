# utility functions

import pandas as pd
import numpy as np
from scipy.stats import norm


def check_data_quality(df: pd.DataFrame) -> None:
    """Check for missing values, stale prices, and outliers."""
    THRESHHOLD_OUTLIERS = 4

    print("Data Quality Check:")
    print("Missing Values:")
    print(df.isnull().sum())
    print("Stale prices indicated by zero returns:")
    print((df == 0).sum())
    print(f"\nOutliers (using {THRESHHOLD_OUTLIERS} standard deviations):")
    z_scores = (df - df.mean()) / df.std()
    outliers = (z_scores.abs() > THRESHHOLD_OUTLIERS).sum()
    print(outliers)
    # print the outliers for each column
    for col in df.columns:
        outlier_indices = z_scores[col].abs() > THRESHHOLD_OUTLIERS
        if outlier_indices.any():
            print(f"\nOutliers in {col}:")
            print(df.loc[outlier_indices, col])


def compute_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate summary statistics for a DataFrame of returns."""
    # determine frequency of data (daily, monthly, etc.) based on index to determine the annualisation factor
    if isinstance(df.index, pd.DatetimeIndex):
        average_difference_days = df.index.diff().mean().days
        if average_difference_days > 0.5 and average_difference_days < 1.5:
            ann_factor = 252
            frequency = 'D'
        elif average_difference_days > 28 and average_difference_days < 31:
            ann_factor = 12
            frequency = 'M'
        elif average_difference_days > 89 and average_difference_days < 92:
            ann_factor = 4
            frequency = 'Q'
        elif average_difference_days > 364 and average_difference_days < 366:
            ann_factor = 1
            frequency = 'Y'
    else:
        raise ValueError("Index must be a DatetimeIndex to determine frequency and annualisation factor.")
    
    # calculate summary statistics
    stats = pd.DataFrame(index=df.columns)
    stats['mean_arithmetic_annualised_pct'] = round(df.mean() * ann_factor * 100, 1)
    stats['mean_geo_annualised_pct'] = round(df.apply(lambda x:np.exp(np.mean(np.log(1 + x))) ** 12 - 1) * 100, 1)
    stats['std_annualised_pct'] = round(df.std() * np.sqrt(ann_factor) * 100, 1)
    stats['skew'] = round(df.skew(), 2)
    stats['kurtosis'] = round(df.kurtosis(), 2)
    stats['risk_reward_ratio'] = round((stats['mean_arithmetic_annualised_pct']/100) / (stats['std_annualised_pct']/100), 2)
    stats['VaR_95_pct'] = round(df.quantile(0.05) * 100, 1)
    stats['ES_95_pct'] = round(df[df < df.quantile(0.05)].mean() * 100, 1)
    stats['max_drawdown_pct'] = round((df.cummax() - df).max() * 100, 1)
    stats['number_of_observations'] = df.count()
    stats['frequency'] = frequency
    
    return stats.T


def calculate_value_at_risk(x: np.ndarray, confidence_level: float=0.95, method: str='empirical') -> float:
    """Calculate value at risk (VaR) at a given confidence level.
    Args:
        x (np.ndarray): vector of monthly returns.
        confidence_level (float): the confidence level for VaR calculation. Default is 95%.
        method (str): the method for VaR calculation (empirical or parametric). Default is 'empirical'. The parametric method assumes that returns are normally distributed, i.e. most likely somewhat underestimates the tail risk.
    Returns:
        float: value at risk without flipping the sign, i.e. usually as a negative number.
    """
    # check that x is a numpy array or something that can be converted to a numpy array
    if isinstance(x, np.ndarray):
        pass
    else:
        try:
            x = np.asarray(x)
        except Exception as e:
            raise ValueError("Input x must be a numpy array.")
    # check that confidence level is between 0 and 1
    if confidence_level <= 0 or confidence_level >= 1:
        raise ValueError("Confidence level must be between 0 and 1.")
    # confidence levels are usually 90% or higher
    if confidence_level < 0.9:
        print(f"Warning: confidence level is {confidence_level}, usually it's a number above 0.9")
    
    # calculate var depending on the specified method
    if method == 'empirical':
        var = np.percentile(x, (1 - confidence_level) * 100)
    elif method == 'parametric':
        # mean minus the standard deviation times the z-score corresponding to the confidence level
        var = np.mean(x) - np.std(x) * norm.ppf(confidence_level)
    else:
        raise ValueError("Method must be either 'empirical' or 'parametric'.")

    return var


def calculate_expected_shortfall(x: np.ndarray, confidence_level: float=0.95, method: str='empirical') -> float:
    """Calculate expected shortfall (ES) at a given confidence level.
    Args:
        x (np.ndarray): vector of monthly returns.
        confidence_level (float): the confidence level for ES calculation. Default is 95%.
        method (str): the method for ES calculation (empirical or parametric). Default is 'empirical'. The parametric method assumes that returns are normally distributed, i.e. most likely somewhat underestimates the tail risk.
    Returns:
        float: expected shortfall without flipping the sign, i.e. usually as a negative number.
    """
    # check that x is a numpy array or something that can be converted to a numpy array
    if isinstance(x, np.ndarray):
        pass
    else:
        try:
            x = np.asarray(x)
        except Exception as e:
            raise ValueError("Input x must be a numpy array.")
    # check that confidence level is between 0 and 1
    if confidence_level <= 0 or confidence_level >= 1:
        raise ValueError("Confidence level must be between 0 and 1.")
    # confidence levels are usually 90% or higher
    if confidence_level < 0.9:
        print(f"Warning: confidence level is {confidence_level}, usually it's a number above 0.9")

    # calculate ES
    #  depending on the specified method
    if method == 'empirical':
        var_threshold = np.percentile(x, (1 - confidence_level) * 100)
        es = x[x < var_threshold].mean()
    elif method == 'parametric':
        # mean minus the standard deviation times the z-score corresponding to the confidence level, multiplied by the ratio of the pdf and cdf of a normal distribution at the z-score corresponding to the confidence level
        es = np.mean(x) - np.std(x) * norm.pdf(norm.ppf(confidence_level)) / (1 - confidence_level)
    else:
        raise ValueError("Method must be either 'empirical' or 'parametric'.")

    return es


def calculate_total_return(x: np.ndarray, method: str='cumulative') -> float:
    """ Calculate the cumulative or average return for a given return vector. 
    Args:
        x (np.ndarray): vector of monthly returns.
        method (str): method for the total return calculation (cumulative or geometric_average). The default is "cumulative" which returns the cumulative return. "geometric_average" calculates the annualised geometric average return assuming monthly return frequency.
    Returns:
        float: total return in percent
    """
    # check that x is a numpy array or something that can be converted to a numpy array
    if isinstance(x, np.ndarray):
        pass
    else:
        try:
            x = np.asarray(x)
        except Exception as e:
            raise ValueError("Input x must be a numpy array.")

    # calculate total return    
    if method == 'cumulative':
        total_return = (np.prod(1 + x) - 1) * 100
    elif method == 'geometric_average':
        total_return = (np.exp(np.log(1 + x).mean() * 12) - 1) * 100
    else:
        raise ValueError("Method must be either 'cumulative' or 'geometric_average'.")
    
    return total_return
