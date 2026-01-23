"""
Compare Univariate Time Series MICE with other imputation methods
Methods: MICE, Interpolation, Mean Imputation, ARMA
Metrics: RMSE, NMAE (Normalized Mean Absolute Error)
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# Import MICE functions
from UnivariateTimeSeriesMICE import mice, pool_results

# Try to import statsmodels for ARMA, but handle if not available
try:
    from statsmodels.tsa.arima.model import ARIMA
    ARMA_AVAILABLE = True
except ImportError:
    ARMA_AVAILABLE = False
    print("Warning: statsmodels not available. ARMA imputation will be skipped.")


def calculate_rmse(true_values, predicted_values):
    """
    Calculate Root Mean Square Error.
    
    Parameters:
    -----------
    true_values : array-like
        True values
    predicted_values : array-like
        Predicted/imputed values
    
    Returns:
    --------
    rmse : float
        Root Mean Square Error
    """
    true_values = np.asarray(true_values).flatten()
    predicted_values = np.asarray(predicted_values).flatten()
    
    # Remove NaN values
    valid_mask = ~(np.isnan(true_values) | np.isnan(predicted_values))
    if not np.any(valid_mask):
        return np.inf
    
    true_clean = true_values[valid_mask]
    pred_clean = predicted_values[valid_mask]
    
    rmse = np.sqrt(np.mean((true_clean - pred_clean)**2))
    return rmse


def calculate_nmae(true_values, predicted_values, method='range'):
    """
    Calculate Normalized Mean Absolute Error.
    
    Parameters:
    -----------
    true_values : array-like
        True values
    predicted_values : array-like
        Predicted/imputed values
    method : str, default='range'
        Normalization method: 'range' (by max-min) or 'mean' (by mean)
    
    Returns:
    --------
    nmae : float
        Normalized Mean Absolute Error
    """
    true_values = np.asarray(true_values).flatten()
    predicted_values = np.asarray(predicted_values).flatten()
    
    # Remove NaN values
    valid_mask = ~(np.isnan(true_values) | np.isnan(predicted_values))
    if not np.any(valid_mask):
        return np.inf
    
    true_clean = true_values[valid_mask]
    pred_clean = predicted_values[valid_mask]
    
    # Calculate MAE
    mae = np.mean(np.abs(true_clean - pred_clean))
    
    # Normalize
    if method == 'range':
        denom = np.max(true_clean) - np.min(true_clean)
        if denom < 1e-8:
            return mae  # Return unnormalized MAE for constant values
        nmae = mae / denom
    elif method == 'mean':
        denom = np.mean(np.abs(true_clean))
        if denom < 1e-8:
            return mae  # Return unnormalized MAE for near-zero values
        nmae = mae / denom
    else:
        raise ValueError("method must be 'range' or 'mean'")
    
    return nmae


def mean_imputation(series):
    """
    Simple mean imputation.
    
    Parameters:
    -----------
    series : pd.Series
        Time series with missing values
    
    Returns:
    --------
    imputed_series : pd.Series
        Series with missing values filled with mean
    """
    imputed_series = series.copy()
    mean_val = series.mean()
    imputed_series = imputed_series.fillna(mean_val)
    return imputed_series


def interpolation_imputation(series, method='linear'):
    """
    Interpolation-based imputation.
    
    Parameters:
    -----------
    series : pd.Series
        Time series with missing values
    method : str, default='linear'
        Interpolation method: 'linear', 'polynomial', 'spline', etc.
    
    Returns:
    --------
    imputed_series : pd.Series
        Series with missing values interpolated
    """
    imputed_series = series.copy()
    
    # Interpolate
    imputed_series = imputed_series.interpolate(method=method)
    
    # Fill any remaining NaN at boundaries with forward/backward fill
    imputed_series = imputed_series.ffill().bfill()
    
    # If still missing, use mean
    if imputed_series.isnull().any():
        mean_val = series.mean()
        imputed_series = imputed_series.fillna(mean_val)
    
    return imputed_series


def arma_imputation(series, order=(1, 0, 1), max_iter=5):
    """
    ARMA model-based imputation.
    
    Parameters:
    -----------
    series : pd.Series
        Time series with missing values
    order : tuple, default=(1, 0, 1)
        ARIMA order (p, d, q)
    max_iter : int, default=5
        Maximum iterations for ARIMA fitting
    
    Returns:
    --------
    imputed_series : pd.Series
        Series with missing values imputed using ARMA
    """
    if not ARMA_AVAILABLE:
        raise ImportError("statsmodels is required for ARMA imputation")
    
    imputed_series = series.copy()
    
    # Initial imputation for ARIMA fitting (use forward/backward fill)
    initial_imputed = imputed_series.ffill().bfill()
    if initial_imputed.isnull().any():
        initial_imputed = initial_imputed.fillna(initial_imputed.mean())
    
    try:
        # Fit ARIMA model
        model = ARIMA(initial_imputed, order=order)
        fitted_model = model.fit()
        
        # Get missing indices
        missing_mask = series.isnull()
        missing_indices = missing_mask[missing_mask].index
        
        if len(missing_indices) > 0:
            # Predict missing values
            # For ARIMA, we can use the fitted model to predict
            # Get predictions for all time points
            try:
                predictions = fitted_model.predict(start=0, end=len(series)-1)
                
                # Convert to numpy array if needed
                if hasattr(predictions, 'values'):
                    predictions = predictions.values
                elif hasattr(predictions, 'iloc'):
                    predictions = predictions.values
                else:
                    predictions = np.array(predictions)
                
                # Update missing values
                for idx in missing_indices:
                    pos = series.index.get_loc(idx)
                    if 0 <= pos < len(predictions):
                        imputed_series.loc[idx] = predictions[pos]
            except Exception as e:
                print(f"   Warning: ARMA prediction failed: {e}")
                # Fall back to using initial imputation
                pass
        
        # Fill any remaining NaN
        if imputed_series.isnull().any():
            imputed_series = imputed_series.fillna(initial_imputed)
            
    except Exception as e:
        print(f"Warning: ARMA fitting failed: {e}. Using interpolation fallback.")
        imputed_series = interpolation_imputation(series)
    
    return imputed_series


def evaluate_imputation_method(true_series, imputed_series, missing_mask):
    """
    Evaluate an imputation method using RMSE and NMAE.
    
    Parameters:
    -----------
    true_series : array-like
        True complete time series
    imputed_series : array-like
        Imputed time series
    missing_mask : array-like
        Boolean mask indicating missing positions
    
    Returns:
    --------
    metrics : dict
        Dictionary with 'rmse' and 'nmae' keys
    """
    # Only evaluate on imputed positions
    true_imputed = true_series[missing_mask]
    pred_imputed = imputed_series[missing_mask]
    
    rmse = calculate_rmse(true_imputed, pred_imputed)
    nmae = calculate_nmae(true_imputed, pred_imputed, method='range')
    
    return {
        'rmse': rmse,
        'nmae': nmae
    }


def compare_all_methods(original_series, series_with_missing, missing_mask,
                       n_past_lags=3, n_future_lags=3, max_iter=10, n_imputation=5,
                       random_state=42):
    """
    Compare all imputation methods.
    
    Parameters:
    -----------
    original_series : array-like
        Complete time series (ground truth)
    series_with_missing : pd.Series
        Time series with missing values
    missing_mask : array-like
        Boolean mask indicating missing positions
    n_past_lags : int, default=3
        Number of past lags for MICE
    n_future_lags : int, default=3
        Number of future lags for MICE
    max_iter : int, default=10
        Max iterations for MICE
    n_imputation : int, default=5
        Number of imputations for MICE
    random_state : int, default=42
        Random seed
    
    Returns:
    --------
    results : dict
        Dictionary with results for each method
    """
    results = {}
    
    # Convert to Series if needed
    original_series = pd.Series(original_series)
    series_with_missing = pd.Series(series_with_missing)
    missing_mask = pd.Series(missing_mask) if not isinstance(missing_mask, pd.Series) else missing_mask
    
    print("=" * 70)
    print("COMPARING IMPUTATION METHODS")
    print("=" * 70)
    print(f"Series length: {len(original_series)}")
    print(f"Missing values: {missing_mask.sum()} ({100*missing_mask.sum()/len(original_series):.2f}%)")
    print()
    
    # 1. Mean Imputation
    print("1. Mean Imputation...")
    mean_imputed = mean_imputation(series_with_missing)
    mean_metrics = evaluate_imputation_method(original_series, mean_imputed, missing_mask)
    results['Mean Imputation'] = {
        'imputed_series': mean_imputed,
        'rmse': mean_metrics['rmse'],
        'nmae': mean_metrics['nmae']
    }
    print(f"   RMSE: {mean_metrics['rmse']:.6f}")
    print(f"   NMAE: {mean_metrics['nmae']:.6f}")
    print()
    
    # 2. Interpolation
    print("2. Linear Interpolation...")
    interp_imputed = interpolation_imputation(series_with_missing, method='linear')
    interp_metrics = evaluate_imputation_method(original_series, interp_imputed, missing_mask)
    results['Interpolation'] = {
        'imputed_series': interp_imputed,
        'rmse': interp_metrics['rmse'],
        'nmae': interp_metrics['nmae']
    }
    print(f"   RMSE: {interp_metrics['rmse']:.6f}")
    print(f"   NMAE: {interp_metrics['nmae']:.6f}")
    print()
    
    # 3. ARMA Imputation
    if ARMA_AVAILABLE:
        print("3. ARMA Imputation...")
        try:
            arma_imputed = arma_imputation(series_with_missing, order=(1, 0, 1))
            arma_metrics = evaluate_imputation_method(original_series, arma_imputed, missing_mask)
            results['ARMA'] = {
                'imputed_series': arma_imputed,
                'rmse': arma_metrics['rmse'],
                'nmae': arma_metrics['nmae']
            }
            print(f"   RMSE: {arma_metrics['rmse']:.6f}")
            print(f"   NMAE: {arma_metrics['nmae']:.6f}")
        except Exception as e:
            print(f"   Failed: {e}")
            results['ARMA'] = None
        print()
    else:
        print("3. ARMA Imputation... Skipped (statsmodels not available)")
        results['ARMA'] = None
        print()
    
    # 4. MICE
    print("4. MICE (Multiple Imputation by Chained Equations)...")
    print(f"   Using {n_past_lags} past lags and {n_future_lags} future lags")
    print(f"   Running {n_imputation} imputations with {max_iter} iterations each...")
    
    imputed_datasets = mice(
        series_with_missing,
        n_past_lags=n_past_lags,
        n_future_lags=n_future_lags,
        max_iter=max_iter,
        n_imputation=n_imputation,
        imputation_model='bayesian_ridge',
        random_state=random_state
    )
    
    # Pool results
    pooled_series, variance_info = pool_results(imputed_datasets, series_with_missing)
    mice_metrics = evaluate_imputation_method(original_series, pooled_series, missing_mask)
    
    results['MICE'] = {
        'imputed_series': pooled_series,
        'imputed_datasets': imputed_datasets,
        'variance_info': variance_info,
        'rmse': mice_metrics['rmse'],
        'nmae': mice_metrics['nmae']
    }
    print(f"   RMSE: {mice_metrics['rmse']:.6f}")
    print(f"   NMAE: {mice_metrics['nmae']:.6f}")
    print(f"   Within variance: {variance_info.get('within_variance', 0):.6f}")
    print(f"   Between variance: {variance_info.get('between_variance', 0):.6f}")
    print()
    
    # Summary
    print("=" * 70)
    print("SUMMARY - Performance Comparison")
    print("=" * 70)
    
    summary_data = []
    for method_name, method_results in results.items():
        if method_results is not None:
            summary_data.append({
                'Method': method_name,
                'RMSE': method_results['rmse'],
                'NMAE': method_results['nmae']
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('RMSE')  # Sort by RMSE (lower is better)
    
    print(summary_df.to_string(index=False))
    print()
    
    # Find best method
    best_rmse = summary_df.loc[summary_df['RMSE'].idxmin(), 'Method']
    best_nmae = summary_df.loc[summary_df['NMAE'].idxmin(), 'Method']
    
    print(f"Best RMSE: {best_rmse}")
    print(f"Best NMAE: {best_nmae}")
    print("=" * 70)
    
    return results, summary_df


def plot_comparison(original_series, series_with_missing, results, missing_mask, save_path=None):
    """
    Plot comparison of all imputation methods.
    
    Parameters:
    -----------
    original_series : array-like
        Complete time series
    series_with_missing : pd.Series
        Time series with missing values
    results : dict
        Results dictionary from compare_all_methods
    missing_mask : array-like
        Boolean mask for missing positions
    save_path : str, optional
        Path to save the plot
    """
    import matplotlib.pyplot as plt
    
    n_methods = len([r for r in results.values() if r is not None])
    fig, axes = plt.subplots(n_methods + 1, 1, figsize=(14, 4 * (n_methods + 1)))
    
    if n_methods == 0:
        axes = [axes]
    
    # Plot 1: Original with missing
    axes[0].plot(original_series.values, label='Original (True)', color='blue', linewidth=2, alpha=0.8)
    axes[0].scatter(np.where(missing_mask)[0], original_series.values[missing_mask], 
                   color='red', s=50, zorder=5, label='Missing Values', marker='x')
    axes[0].set_title('Original Time Series with Missing Values', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Value')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot each method
    plot_idx = 1
    for method_name, method_results in results.items():
        if method_results is not None:
            imputed = method_results['imputed_series']
            rmse = method_results['rmse']
            nmae = method_results['nmae']
            
            axes[plot_idx].plot(original_series.values, label='Original (True)', 
                               color='blue', linewidth=1.5, alpha=0.7)
            axes[plot_idx].plot(imputed.values, label=f'{method_name} (Imputed)', 
                               color='orange', linewidth=1.5, alpha=0.7, linestyle='--')
            axes[plot_idx].scatter(np.where(missing_mask)[0], imputed.values[missing_mask], 
                                  color='red', s=30, zorder=5, label='Imputed Values', alpha=0.8)
            axes[plot_idx].set_title(f'{method_name} - RMSE: {rmse:.4f}, NMAE: {nmae:.4f}', 
                                    fontsize=11, fontweight='bold')
            axes[plot_idx].set_xlabel('Time')
            axes[plot_idx].set_ylabel('Value')
            axes[plot_idx].legend()
            axes[plot_idx].grid(True, alpha=0.3)
            plot_idx += 1
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()

