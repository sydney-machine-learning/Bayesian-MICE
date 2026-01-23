"""
Univariate Time Series MICE using the same format as MICE_MCMC_V2
Uses past and future lags to prepare datasets
No MCMC - uses simple regression models (LinearRegression or BayesianRidge)
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


def place_holder(series):
    """
    Initialize missing values in the time series.
    Uses forward-fill, backward-fill, and mean as fallback.
    Following the same pattern as MICE_MCMC_V2.
    
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


class UnivariateTimeSeriesDataPreparation:
    """
    Data preparation for univariate time series MICE.
    Creates features using past and future lags.
    Following the same structure as ForwardOnlyDataPreparation in MICE_MCMC_V2.
    """
    
    @staticmethod
    def prepare_lag_features(series, t, n_past_lags, n_future_lags):
        """
        Prepare feature vector using past and future lags for time point t.
        Similar to _prepare_lags_only but for univariate time series with both past and future.
        
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
    
    @staticmethod
    def prepare_training_data(series, n_past_lags, n_future_lags):
        """
        Prepare training data using past and future lags.
        Similar to _prepare_lags_only in MICE_MCMC_V2.
        
        Parameters:
        -----------
        series : pd.Series or array-like
            Time series data
        n_past_lags : int
            Number of past lags
        n_future_lags : int
            Number of future lags
        
        Returns:
        --------
        X : np.array, shape (n_samples, n_past_lags + n_future_lags)
            Feature matrix
        y : np.array, shape (n_samples,)
            Target values
        valid_indices : np.array
            Indices where features could be created
        """
        series = pd.Series(series)
        n = len(series)
        
        X = []
        y = []
        valid_indices = []
        
        # We can only create features for indices that have enough past and future values
        min_idx = n_past_lags
        max_idx = n - n_future_lags
        
        for t in range(min_idx, max_idx):
            features = UnivariateTimeSeriesDataPreparation.prepare_lag_features(
                series.values, t, n_past_lags, n_future_lags
            )
            
            if features is not None and not np.isnan(series.iloc[t]):
                X.append(features)
                y.append(series.iloc[t])
                valid_indices.append(t)
        
        if len(X) == 0:
            return np.array([]), np.array([]), np.array([])
        
        return np.array(X), np.array(y), np.array(valid_indices)
    
    @staticmethod
    def prepare_prediction_features(series, t, n_past_lags, n_future_lags):
        """
        Prepare features for predicting at time t.
        Similar to prepare_prediction_features in MICE_MCMC_V2.
        
        Parameters:
        -----------
        series : pd.Series or array-like
            Time series data
        t : int
            Time index to predict
        n_past_lags : int
            Number of past lags
        n_future_lags : int
            Number of future lags
        
        Returns:
        --------
        features : np.array or None
            Feature vector for prediction, or None if insufficient data
        """
        features = UnivariateTimeSeriesDataPreparation.prepare_lag_features(
            series, t, n_past_lags, n_future_lags
        )
        
        if features is None:
            return None
        
        return features.reshape(1, -1)


def predict_impute(series_filled, static_mask, n_past_lags, n_future_lags,
                  imputation_model='linear', random_state=None):
    """
    Predict and impute missing values for one iteration.
    Following the same pattern as predict_impute in MICE_scratch but adapted for univariate time series.
    
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
    
    # Prepare training data using observed values (not originally missing)
    # Similar to how MICE_MCMC_V2 prepares training data
    X_train, y_train, train_indices = UnivariateTimeSeriesDataPreparation.prepare_training_data(
        series_filled, n_past_lags, n_future_lags
    )
    
    if len(X_train) == 0:
        # Not enough data to train, return as is
        return series_filled
    
    # Filter training data to only use observed values (not originally missing)
    # This ensures we train only on originally observed data
    observed_mask = np.array([
        idx not in missing_indices for idx in train_indices
    ])
    
    if observed_mask.sum() < n_past_lags + n_future_lags + 1:
        # Not enough observed data
        return series_filled
    
    X_obs = X_train[observed_mask]
    y_obs = y_train[observed_mask]
    
    # Scale features
    scaler = StandardScaler()
    X_obs_scaled = scaler.fit_transform(X_obs)
    
    # Fit model
    if imputation_model == 'bayesian_ridge':
        model = BayesianRidge(compute_score=True, max_iter=300)
    elif imputation_model == 'linear':
        model = LinearRegression()
    else:
        raise ValueError(f"Unknown imputation_model: {imputation_model}")
    
    model.fit(X_obs_scaled, y_obs)
    
    # Predict missing values
    for t in missing_indices:
        t_pos = series_filled.index.get_loc(t)
        features = UnivariateTimeSeriesDataPreparation.prepare_prediction_features(
            series_filled.values, t_pos, n_past_lags, n_future_lags
        )
        
        if features is not None:
            # Scale features
            features_scaled = scaler.transform(features)
            
            # Predict using the model
            prediction = model.predict(features_scaled)[0]
            
            # Add random noise based on residual standard deviation (similar to MICE_scratch)
            if imputation_model == 'linear':
                y_pred_train = model.predict(X_obs_scaled)
                residuals = y_obs - y_pred_train
                residual_std = np.std(residuals) if len(residuals) > 1 else 0.01 * np.mean(np.abs(y_obs))
                
                noise = np.random.normal(0, residual_std)
                prediction += noise
            
            series_filled.loc[t] = prediction
    
    return series_filled


def perform_iteration(series, static_mask, n_past_lags, n_future_lags, max_iter,
                     imputation_model='linear', random_state=None):
    """
    Perform one complete MICE iteration cycle.
    Following the same pattern as perform_iteration in MICE_scratch.
    
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
    Following the same pattern as mice() in MICE_scratch.
    
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
    Following the same pattern as pool_results in MICE_scratch.
    
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


def create_missing_data_pattern(series, missing_rate=0.1, pattern='random', random_state=None):
    """
    Create missing data patterns for testing.
    
    Parameters:
    -----------
    series : array-like
        Complete time series
    missing_rate : float, default=0.1
        Proportion of data to make missing
    pattern : str, default='random'
        Missing pattern: 'random', 'block', or 'mcar'
    random_state : int, default=None
        Random seed
    
    Returns:
    --------
    series_with_missing : pd.Series
        Series with missing values
    missing_mask : pd.Series
        Boolean mask indicating missing positions
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    series = pd.Series(series)
    n = len(series)
    n_missing = int(n * missing_rate)
    missing_mask = pd.Series([False] * n, index=series.index)
    
    if pattern == 'random' or pattern == 'mcar':
        # Missing completely at random
        missing_indices = np.random.choice(n, size=n_missing, replace=False)
        missing_mask.iloc[missing_indices] = True
    
    elif pattern == 'block':
        # Missing in blocks (more realistic for time series)
        block_size = max(1, n_missing // 5)  # 5 blocks
        n_blocks = n_missing // block_size
        
        for _ in range(n_blocks):
            start_idx = np.random.randint(0, n - block_size)
            missing_mask.iloc[start_idx:start_idx + block_size] = True
        
        # Fill remaining missing spots randomly
        remaining = n_missing - missing_mask.sum()
        if remaining > 0:
            available = np.where(~missing_mask)[0]
            if len(available) >= remaining:
                additional = np.random.choice(available, size=remaining, replace=False)
                missing_mask.iloc[additional] = True
    
    else:
        raise ValueError(f"Unknown pattern: {pattern}")
    
    series_with_missing = series.copy()
    series_with_missing[missing_mask] = np.nan
    
    return series_with_missing, missing_mask

