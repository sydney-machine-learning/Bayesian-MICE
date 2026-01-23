# Univariate Time Series Imputation Methods Comparison

This module compares different imputation methods for univariate time series data, including MICE, interpolation, mean imputation, and ARMA models.

## Methods Compared

1. **Mean Imputation**: Simple mean imputation - fills missing values with the mean of observed values
2. **Interpolation**: Linear interpolation - uses linear interpolation between observed values
3. **ARMA**: ARMA (AutoRegressive Moving Average) model - uses time series modeling to predict missing values
4. **MICE**: Multiple Imputation by Chained Equations - uses past and future lags as features for iterative imputation

## Metrics

- **RMSE**: Root Mean Square Error - measures the average magnitude of errors
- **NMAE**: Normalized Mean Absolute Error - MAE normalized by the range of true values (makes it scale-independent)

## Usage

### Basic Usage

```python
from compare_imputation_methods import compare_all_methods
from UnivariateTimeSeriesMICE import create_missing_data_pattern
import numpy as np
import pandas as pd

# Your time series with missing values
original_series = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
series_with_missing, missing_mask = create_missing_data_pattern(
    original_series, missing_rate=0.2, pattern='random', random_state=42
)

# Compare all methods
results, summary_df = compare_all_methods(
    original_series=original_series,
    series_with_missing=series_with_missing,
    missing_mask=missing_mask,
    n_past_lags=3,      # For MICE
    n_future_lags=3,    # For MICE
    max_iter=10,        # For MICE
    n_imputation=5,     # For MICE
    random_state=42
)

# Print summary
print(summary_df)
```

### Run Example

```bash
cd MCMC_MICE_codes
python example_comparison.py
```

This will:
1. Generate a sample time series
2. Create missing data
3. Compare all methods
4. Generate comparison plots
5. Save results to CSV

## Output

The comparison function returns:
- `results`: Dictionary with imputed series and metrics for each method
- `summary_df`: DataFrame with RMSE and NMAE for all methods, sorted by RMSE

## Dependencies

- numpy
- pandas
- scikit-learn
- matplotlib (for plotting)
- statsmodels (optional, for ARMA - will skip if not available)

## Notes

- ARMA imputation requires `statsmodels` package. If not available, ARMA will be skipped.
- MICE uses both past and future lags, making it suitable for time series with temporal dependencies.
- The comparison evaluates only on the originally missing positions.

