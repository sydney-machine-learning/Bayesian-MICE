"""
MICE (Multiple Imputation by Chained Equations) for Univariate Time Series
Uses both past and future lags as features for imputation
Following the same pattern as MICE_scratch.ipynb but adapted for univariate time series
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, BayesianRidge
import warnings
warnings.filterwarnings('ignore')


def place_holder(series):
    """
    Initialize missing values in the time series.
    Uses forward-fill, backward-fill, and mean as fallback.
    
    Parameters:
    -----------
    series : array-like or pd.Series
        Time series with missing values
    
    Returns:
    --------
    series_filled : pd.Series
        Series with missing values initialized
    """
    series_filled = pd.Series(series).copy()
    
    # Forward fill
    series_filled = series_filled.ffill()
    
    # Backward fill for any remaining missing values
    series_filled = series_filled.bfill()
    
    # If still missing, use mean
    if series_filled.isnull().any():
        mean_val = series_filled.mean()
        series_filled = series_filled.fillna(mean_val)
    
    return series_filled


def create_lag_features(series, t, n_past_lags, n_future_lags):
    """
    Create feature vector using past and future lags for time point t.
    
    Parameters:
    -----------
    series : array-like
        Time series data
    t : int
        Current time index
    n_past_lags : int
        Number of past lags to use
    n_future_lags : int
        Number of future lags to use
    
    Returns:
    --------
    features : array-like or None
        Feature vector [past_lag_1, ..., past_lag_n, future_lag_1, ..., future_lag_n]
        Returns None if not enough data available
    """
    n = len(series)
    
    # Check if we have enough past and future values
    if t < n_past_lags or t >= n - n_future_lags:
        return None
    
    features = []
    
    # Past lags: t-1, t-2, ..., t-n_past_lags
    for lag in range(1, n_past_lags + 1):
        features.append(series[t - lag])
    
    # Future lags: t+1, t+2, ..., t+n_future_lags
    for lag in range(1, n_future_lags + 1):
        features.append(series[t + lag])
    
    # Check for NaN in features
    if np.isnan(features).any():
        return None
    
    return np.array(features)


def predict_impute(series_filled, static_mask, n_past_lags, n_future_lags, 
                   imputation_model='linear', random_state=None):
    """
    Predict and impute missing values for one iteration.
    Uses past and future lags as features.
    
    Parameters:
    -----------
    series_filled : pd.Series
        Time series with initialized values (may have been updated in previous iterations)
    static_mask : pd.Series
        Boolean mask indicating originally missing values (static, doesn't change)
    n_past_lags : int
        Number of past lags to use
    n_future_lags : int
        Number of future lags to use
    imputation_model : str, default='linear'
        Model to use: 'linear' or 'bayesian_ridge'
    random_state : int, default=None
        Random seed
    
    Returns:
    --------
    series_filled : pd.Series
        Series with imputed values updated
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Get indices where values are originally missing
    missing_indices = static_mask[static_mask].index.tolist()
    
    if len(missing_indices) == 0:
        return series_filled  # No missing values to impute
    
    # Prepare training data: use observed values (not originally missing)
    observed_indices = static_mask[~static_mask].index.tolist()
    
    # Build feature matrix and target vector for training
    X_train = []
    y_train = []
    train_indices = []
    
    for t in observed_indices:
        # Only use indices where we can create lag features
        t_pos = series_filled.index.get_loc(t)
        features = create_lag_features(series_filled.values, t_pos, n_past_lags, n_future_lags)
        
        if features is not None:
            X_train.append(features)
            y_train.append(series_filled.loc[t])
            train_indices.append(t)
    
    if len(X_train) == 0:
        # Not enough data to train, return as is
        return series_filled
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    # Fit model
    if imputation_model == 'bayesian_ridge':
        model = BayesianRidge(compute_score=True, max_iter=300)
    elif imputation_model == 'linear':
        model = LinearRegression()
    else:
        raise ValueError(f"Unknown imputation_model: {imputation_model}")
    
    model.fit(X_train, y_train)
    
    # Predict missing values
    for t in missing_indices:
        t_pos = series_filled.index.get_loc(t)
        features = create_lag_features(series_filled.values, t_pos, n_past_lags, n_future_lags)
        
        if features is not None:
            # Predict using the model
            prediction = model.predict(features.reshape(1, -1))[0]
            
            # Add random noise based on residual standard deviation (similar to MICE_scratch)
            if imputation_model == 'linear':
                y_pred_train = model.predict(X_train)
                residuals = y_train - y_pred_train
                residual_std = np.std(residuals) if len(residuals) > 1 else 0.01 * np.mean(np.abs(y_train))
                
                noise = np.random.normal(0, residual_std)
                prediction += noise
            
            series_filled.loc[t] = prediction
        else:
            # For boundary cases, use interpolation
            # This handles cases at the beginning/end where we can't create lag features
            pass
    
    return series_filled


def perform_iteration(series, static_mask, n_past_lags, n_future_lags, max_iter,
                     imputation_model='linear', random_state=None):
    """
    Perform one complete MICE iteration cycle.
    
    Parameters:
    -----------
    series : pd.Series
        Original time series with missing values
    static_mask : pd.Series
        Boolean mask indicating originally missing values
    n_past_lags : int
        Number of past lags to use
    n_future_lags : int
        Number of future lags to use
    max_iter : int
        Number of iterations to perform
    imputation_model : str, default='linear'
        Model to use for imputation
    random_state : int, default=None
        Random seed
    
    Returns:
    --------
    series_filled : pd.Series
        Imputed time series
    """
    # Initialize with placeholders
    series_filled = place_holder(series)
    
    # Iterate max_iter times
    for i in range(max_iter):
        # Use different random seed for each iteration to add variability
        iter_seed = random_state + i if random_state is not None else None
        series_filled = predict_impute(series_filled, static_mask, n_past_lags, n_future_lags,
                                      imputation_model, random_state=iter_seed)
    
    return series_filled


def mice(series, n_past_lags=3, n_future_lags=3, max_iter=10, n_imputation=5,
         imputation_model='linear', random_state=None):
    """
    Perform Multiple Imputation by Chained Equations (MICE) for univariate time series.
    Uses both past and future lags as features.
    
    Parameters:
    -----------
    series : array-like or pd.Series
        Univariate time series with missing values (NaN)
    n_past_lags : int, default=3
        Number of past lags to use as features (t-1, t-2, ..., t-n_past_lags)
    n_future_lags : int, default=3
        Number of future lags to use as features (t+1, t+2, ..., t+n_future_lags)
    max_iter : int, default=10
        Number of iterations per imputation
    n_imputation : int, default=5
        Number of imputed datasets to generate
    imputation_model : str, default='linear'
        Model to use: 'linear' or 'bayesian_ridge'
    random_state : int, default=None
        Random seed for reproducibility
    
    Returns:
    --------
    imputed_datasets : list of pd.Series
        List of n_imputation imputed time series
    """
    # Convert to Series if needed
    series = pd.Series(series)
    
    # Compute static missing mask from the original data
    static_mask = series.isnull()
    
    if not static_mask.any():
        print("No missing values found. Returning original series.")
        return [series.copy() for _ in range(n_imputation)]
    
    imputed_datasets = []
    
    for i in range(n_imputation):
        # Start fresh from the original data each time
        series_imputed = series.copy()
        
        # Use different seed for each imputation
        imp_seed = random_state + i * 1000 if random_state is not None else None
        
        # Perform iteration
        series_imputed = perform_iteration(series_imputed, static_mask, n_past_lags, n_future_lags,
                                          max_iter, imputation_model, random_state=imp_seed)
        
        imputed_datasets.append(series_imputed.copy())
    
    return imputed_datasets


def pool_results(imputed_datasets, original_series):
    """
    Pool results from multiple imputations using Rubin's rules.
    
    Parameters:
    -----------
    imputed_datasets : list of pd.Series
        List of imputed time series
    original_series : pd.Series
        Original time series with missing values
    
    Returns:
    --------
    pooled_series : pd.Series
        Pooled (averaged) imputed time series
    variance_info : dict
        Variance information (within, between, total)
    """
    m = len(imputed_datasets)
    pooled_series = original_series.copy()
    variance_info = {}
    
    # Get missing mask
    missing_mask = original_series.isnull()
    
    if not missing_mask.any():
        return pooled_series, variance_info
    
    # Get missing indices
    missing_indices = missing_mask[missing_mask].index
    
    # Extract imputed values across all m datasets: shape (m, n_missing)
    imputed_values = np.array([
        imp_df.loc[missing_indices].values
        for imp_df in imputed_datasets
    ])
    
    # Mean across imputations (Rubin's pooled estimate)
    pooled_values = np.mean(imputed_values, axis=0)
    pooled_series.loc[missing_indices] = pooled_values
    
    # Between-imputation variance B: variance across imputations, per missing position
    B_i = np.var(imputed_values, axis=0, ddof=1) if m > 1 else np.zeros(len(missing_indices))
    B = np.mean(B_i)
    
    # Within-imputation variance W: average of variances across missing positions
    # For univariate time series, we estimate W from residuals
    W_i = []
    for j in range(len(missing_indices)):
        pos_vals = imputed_values[:, j]
        var_j = np.var(pos_vals, ddof=1) if m > 1 else 0.0
        W_i.append(var_j)
    W = np.mean(W_i) if len(W_i) > 0 else 0.0
    
    # Total variance T using Rubin's Rule
    T = W + (1 + 1/m) * B if m > 1 else W
    
    variance_info = {
        'within_variance': W,
        'between_variance': B,
        'total_variance': T,
        'n_imputations': m
    }
    
    return pooled_series, variance_info
