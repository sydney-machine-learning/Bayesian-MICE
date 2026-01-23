"""
Example: Compare Univariate Time Series MICE with other imputation methods
"""

import numpy as np
import pandas as pd
from compare_imputation_methods import (
    compare_all_methods, 
    plot_comparison
)
from UnivariateTimeSeriesMICE import create_missing_data_pattern


def generate_sample_time_series(n=200, trend=True, seasonality=True, noise_level=0.1, random_state=42):
    """
    Generate a sample univariate time series.
    """
    np.random.seed(random_state)
    t = np.arange(n)
    
    # Base signal
    series = np.zeros(n)
    
    # Add trend
    if trend:
        series += 0.01 * t
    
    # Add seasonality
    if seasonality:
        period = 20  # Seasonal period
        series += 2 * np.sin(2 * np.pi * t / period)
        series += 1 * np.cos(4 * np.pi * t / period)
    
    # Add noise
    series += np.random.normal(0, noise_level, n)
    
    return series


def main():
    """
    Main function to run the comparison.
    """
    print("=" * 70)
    print("UNIVARIATE TIME SERIES IMPUTATION METHODS COMPARISON")
    print("=" * 70)
    print()
    
    # Generate sample time series
    print("Generating sample time series...")
    original_series = generate_sample_time_series(
        n=200, 
        trend=True, 
        seasonality=True, 
        noise_level=0.5, 
        random_state=42
    )
    print(f"Generated series of length {len(original_series)}")
    print()
    
    # Create missing data
    print("Creating missing data pattern...")
    missing_rate = 0.15  # 15% missing
    series_with_missing, missing_mask = create_missing_data_pattern(
        original_series, 
        missing_rate=missing_rate, 
        pattern='random', 
        random_state=42
    )
    print(f"Missing rate: {missing_rate*100:.1f}% ({missing_mask.sum()} values)")
    print()
    
    # Compare all methods
    results, summary_df = compare_all_methods(
        original_series=original_series,
        series_with_missing=series_with_missing,
        missing_mask=missing_mask,
        n_past_lags=3,
        n_future_lags=3,
        max_iter=10,
        n_imputation=5,
        random_state=42
    )
    
    # Plot comparison
    print("\nGenerating comparison plots...")
    plot_comparison(
        original_series=original_series,
        series_with_missing=series_with_missing,
        results=results,
        missing_mask=missing_mask,
        save_path='imputation_methods_comparison.png'
    )
    
    # Save summary to CSV
    summary_df.to_csv('imputation_comparison_summary.csv', index=False)
    print("Summary saved to: imputation_comparison_summary.csv")
    
    print("\n" + "=" * 70)
    print("Comparison completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()

