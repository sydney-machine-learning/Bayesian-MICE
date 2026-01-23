from packages import *

def time_series_pattern(data, time_col = None, value_col = None):
    all_results = {}
    
    # Sort by time if time_col is given
    if time_col is not None:
        data = data.sort_values(by=time_col)

    # Choose which columns to analyze
    if value_col is not None:
        columns = [value_col]
    else:
        columns = [col for col in data.columns if col != time_col]
    
    # Process each column
    for col in columns:
        # Skip the time column and object type columns
        if col == time_col or data[col].dtype == np.dtype('O'):
            if data[col].dtype == np.dtype('O'):
                warnings.warn(f"Skipping column '{col}' because it is object type and not suitable for time series analysis.")
            continue
            
        results = {
            'has_trend': False,
            'trend_strength': 0.0,
            'has_seasonality': False,
            'seasonal_period': None,
            'seasonal_strength': 0.0
        }
        
        series = data[col].dropna()
        
        if len(series) < 10:
            warnings.warn(f"Skipping column '{col}' because it has fewer than 10 non-NA values.")
            all_results[col] = results
            continue  # Not enough data for analysis
        
        # Mann-Kendall test for trend using pymannkendall
        try:
            mk_result = mk.original_test(series)
            results['has_trend'] = mk_result.p < 0.05  # FIXED: Corrected p-value access
            results['trend_strength'] = abs(mk_result.tau)  # Strength of the trend (Tau)
            print(f"Column '{col}' - Mann-Kendall test result: {mk_result.trend}, p-value: {mk_result.p}, tau: {mk_result.tau}")
        except Exception as e:
            warnings.warn(f"Mann-Kendall test failed: {str(e)}")
        
        # STL decomposition for seasonality (requires at least 2 full periods)
        try:
            # Try to determine an appropriate seasonal period
            # First check if the time data has a regular frequency we can use
            if time_col is not None and pd.api.types.is_datetime64_any_dtype(data[time_col]):
                # Try to infer frequency from time data
                time_diff = pd.Series(data[time_col]).diff().dropna()
                most_common_diff = time_diff.value_counts().idxmax()
                if isinstance(most_common_diff, pd.Timedelta):
                    if most_common_diff.days == 1:
                        # Daily data - check common periods
                        potential_periods = [7, 14, 30, 365]  # weekly, bi-weekly, monthly, yearly
                    elif most_common_diff.days == 7:
                        # Weekly data
                        potential_periods = [4, 12, 52]  # monthly, quarterly, yearly
                    elif most_common_diff.days >= 28 and most_common_diff.days <= 31:
                        # Monthly data
                        potential_periods = [3, 6, 12]  # quarterly, bi-annual, yearly
                    else:
                        # Use default periods
                        potential_periods = [7, 12, 24, 30, 52, 365]
                else:
                    # Fall back to default periods if we can't determine frequency
                    potential_periods = [7, 12, 24, 30, 52, 365]
            else:
                # Without time data, use some common periods
                potential_periods = [7, 12, 24, 30, 52, 365]
            
            best_period = None
            best_strength = 0
            best_stl = None  # Save the best STL fit

            for period in potential_periods:
                if len(series) >= 2 * period:
                    try:
                        stl = STL(series, period=period, robust=True).fit()
                        seasonal = stl.seasonal
                        resid = stl.resid
                        strength = 1 - np.var(resid) / np.var(seasonal + resid)

                        if strength > best_strength and strength > 0.1:
                            best_strength = strength
                            best_period = period
                            best_stl = stl
                    except Exception as e:
                        warnings.warn(f"STL failed for period {period}: {str(e)}")

            if best_period is not None:
                results['has_seasonality'] = True
                results['seasonal_period'] = best_period
                results['seasonal_strength'] = best_strength
                print(f"Column '{col}' - Detected seasonality with period {best_period}, strength: {best_strength:.4f}")
                
                # Improved plotting
                plt.figure(figsize=(12, 8))  # Larger figure size
                
                # If data is very large, sample it for plotting
                if len(series) > 1000:
                    # For visualization only - use a sample
                    sample_size = min(1000, len(series))
                    step = len(series) // sample_size
                    
                    plt.subplot(4, 1, 1)
                    plt.plot(np.arange(0, len(series), step), series[::step], 'b-', linewidth=1)
                    plt.title(f'Observed: {col}')
                    
                    plt.subplot(4, 1, 2)
                    plt.plot(np.arange(0, len(series), step), best_stl.trend[::step], 'b-', linewidth=1)
                    plt.title('Trend')
                    
                    plt.subplot(4, 1, 3)
                    plt.plot(np.arange(0, len(series), step), best_stl.seasonal[::step], 'b-', linewidth=1)
                    plt.title(f'Seasonal (period={best_period})')
                    
                    plt.subplot(4, 1, 4)
                    plt.scatter(np.arange(0, len(series), step), best_stl.resid[::step], s=3, alpha=0.5)
                    plt.title('Residual')
                else:
                    # If data is small enough, plot everything
                    best_stl.plot()
                
                plt.tight_layout()
                plt.show()
                plt.close()  # Close the figure to free memory

        except Exception as e:
            warnings.warn(f"STL decomposition failed: {str(e)}")
        
        all_results[col] = results
    
    return all_results

def handle_seasonality(X_filled, column, missing_mask, non_missing, period):
    result = X_filled[column].copy()
    # Calculate seasonal averages for each position in the cycle
    seasonal_positions = pd.Series(np.arange(len(X_filled)) % period, index=X_filled.index)
    seasonal_means = {}
    
    # For each position in the seasonal cycle, calculate the average
    for pos in range(period):
        pos_mask = (seasonal_positions == pos) & non_missing
        if pos_mask.any():
            seasonal_means[pos] = X_filled.loc[pos_mask, column].mean()
    
    # If we have any seasonal means, use them
    if seasonal_means:
        # Fill missing values with the average from their seasonal position
        for idx in X_filled.index[missing_mask]:
            pos = seasonal_positions[idx]
            if pos in seasonal_means:
                
                result.loc[idx] = seasonal_means[pos]  # ← Fix: don't use column indexer here
            else:
                # If no data for this specific position, use average of all seasonal means
                result.loc[idx] = sum(seasonal_means.values()) / len(seasonal_means)  # ← Fix: don't use column indexer
    else:
        # No seasonal means available, fall back to mean imputation
        mean_value = X_filled.loc[non_missing, column].mean()
        result.loc[missing_mask] = mean_value  # ← Fix: correctly apply mask
    
    return result
               
def place_holder(data, ts_analysis=None):
    X_filled = data.copy()
    for column in X_filled.columns:
        missing_mask = X_filled[column].isnull()
        if not missing_mask.any():
            continue  # No missing values in this column

        # Get time series analysis for this column if available
        col_ts_analysis = ts_analysis.get(column, {}) if ts_analysis else {}
        has_trend = col_ts_analysis.get('has_trend', False)
        has_seasonality = col_ts_analysis.get('has_seasonality', False)

        if X_filled[column].dtype == np.dtype('O'):  # Categorical variable
            mode_value = X_filled[column].mode()[0]
            X_filled.loc[missing_mask, column] = mode_value
        else:  # Numerical data
            non_missing = ~missing_mask  # Get non-missing values
            
            if has_trend and has_seasonality:
                # Handle both trend and seasonality together using STL decomposition
                period = col_ts_analysis.get('seasonal_period', 7)
                
                # Create a clean series of the non-missing values
                temp_series = X_filled.loc[non_missing, column].copy()
                
                # Check if we have enough data for decomposition
                if len(temp_series) >= 2 * period and len(temp_series) >= 4:
                    try:             
                        # Perform STL decomposition on non-missing values
                        stl = STL(temp_series, period=period, robust=True).fit()
                        trend = stl.trend
                        seasonal = stl.seasonal
                        residual = stl.resid
                        
                        # Store decomposition components
                        trend_dict = {idx: val for idx, val in zip(temp_series.index, trend)}
                        seasonal_dict = {}
                        
                        # Calculate average seasonal component for each position in the cycle
                        seasonal_positions = np.arange(len(temp_series)) % period
                        for pos in range(period):
                            pos_indices = seasonal_positions == pos
                            if np.any(pos_indices):
                                seasonal_values = seasonal[pos_indices]
                                seasonal_dict[pos] = np.mean(seasonal_values)
                        
                        # Extrapolate trend to all indices by fitting a simple model to the trend component
                        # Use trend dictionary indices and values for regression
                        trend_indices = np.array([X_filled.index.get_loc(idx) for idx in trend_dict.keys()])
                        trend_values = np.array(list(trend_dict.values()))
                        
                        if len(trend_indices) >= 2:  # Need at least 2 points for regression
                            # Fit trend model
                            slope, intercept, _, _, _ = stats.linregress(trend_indices, trend_values)
                            
                            # Extrapolate trend to all points
                            all_indices = np.arange(len(X_filled))
                            trend_all = intercept + slope * all_indices
                            
                            # For each missing value, combine trend and seasonal components
                            for i, idx in enumerate(X_filled.index):
                                if missing_mask.loc[idx]:  # ← Fix: use .loc[] to access the boolean value
                                    # Get position in seasonal cycle
                                    pos = i % period
                                    # Get trend component
                                    trend_component = trend_all[i]
                                    # Get seasonal component
                                    seasonal_component = seasonal_dict.get(pos, 0)
                                    if pos not in seasonal_dict and seasonal_dict:
                                        # If specific position not found but others are, use average
                                        seasonal_component = sum(seasonal_dict.values()) / len(seasonal_dict)               
                                    # Combine components
                                    X_filled.loc[idx, column] = trend_component + seasonal_component
                        else:
                            # Not enough points for trend analysis, fall back to seasonal-only approach
                            warnings.warn("Too few points for trend component analysis. Using seasonal-only imputation.")
                            for i, idx in enumerate(X_filled.index):
                                if missing_mask.loc[idx]:  # ← Fix: use .loc[] to access the boolean value
                                    pos = i % period
                                    if pos in seasonal_dict:
                                        X_filled.loc[idx, column] = X_filled[column].mean() + seasonal_dict[pos]
                                    elif seasonal_dict:
                                        X_filled.loc[idx, column] = X_filled[column].mean() + sum(seasonal_dict.values()) / len(seasonal_dict)
                                    else:
                                        X_filled.loc[idx, column] = X_filled.loc[non_missing, column].mean()
                    
                    except Exception as e:
                        warnings.warn(f"STL decomposition failed: {str(e)}. Using simpler methods.")
                        # Fall back to separate handling of trend and seasonality
                        if has_seasonality:
                            # Handle seasonality (use existing seasonal averaging code)
                            tmp_filled = handle_seasonality(X_filled, column, missing_mask, non_missing, period)
                            X_filled[column] = tmp_filled
                        
                        # Handle any remaining missing values with interpolation for trend
                        still_missing = X_filled[column].isnull()
                        if still_missing.any():
                            X_filled[column] = X_filled[column].interpolate(method='linear')
                            
                            # Fill any final gaps with mean
                            final_missing = X_filled[column].isnull()
                            if final_missing.any():
                                X_filled.loc[final_missing, column] = X_filled.loc[non_missing, column].mean()
                else:
                    # Not enough data for STL, use separate handling
                    if has_seasonality:
                        # Handle seasonality first
                        tmp_filled = handle_seasonality(X_filled, column, missing_mask, non_missing, period)
                        X_filled[column] = tmp_filled
                    
                    # Handle any remaining missing values with interpolation for trend
                    still_missing = X_filled[column].isnull()
                    if still_missing.any():
                        X_filled[column] = X_filled[column].interpolate(method='linear')
                        
                        # Fill any final gaps with mean
                        final_missing = X_filled[column].isnull()
                        if final_missing.any():
                            X_filled.loc[final_missing, column] = X_filled.loc[non_missing, column].mean()
            elif has_seasonality:
                # Handle seasonality only using seasonal averaging
                period = col_ts_analysis.get('seasonal_period', 7)
                tmp_filled = handle_seasonality(X_filled, column, missing_mask, non_missing, period)
                X_filled[column] = tmp_filled
                        
            elif has_trend:
                # Handle trend only using interpolation
                X_filled[column] = X_filled[column].interpolate(method='linear')
                
                # Fill any remaining NAs with mean
                still_missing = X_filled[column].isnull()
                if still_missing.any():
                    X_filled.loc[still_missing, column] = X_filled.loc[non_missing, column].mean()
            else:
                # No patterns, use mean imputation
                mean_value = X_filled.loc[non_missing, column].mean()
                X_filled.loc[missing_mask, column] = mean_value
    
    return X_filled           