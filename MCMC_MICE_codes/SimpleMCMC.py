from packages import *
from MCMC_CHAIN import *
from Visualisation import *
from placeholder import *


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('enhanced_mcmc_mice')
class SelectiveDataPreparation:
    @staticmethod
    def prepare_selective_data(df, target, features, time_col='Date_Time', max_lags=2, data_type ='air'):
        """
        Simple wrapper that calls _prepare_lags_only
        """
        return SelectiveDataPreparation._prepare_lags_only(df, target, features, time_col, max_lags, data_type)
    
    @staticmethod
    def _prepare_lags_only(data, target, features, time_col, max_lags, data_type ='air'):
        """
        Conservative baseline: Just lags with proper multiple lag support
        """
        print(f"  Using LAGS ONLY strategy with {max_lags} lags")
        
        # Only hour for minimal time awareness
        time_features = []
        if data_type == 'air' or data_type == 'simulated':
            if time_col in data.columns and pd.api.types.is_datetime64_any_dtype(data[time_col]):
                data['hour'] = data[time_col].dt.hour
                data['dayofweek'] = data[time_col].dt.dayofweek
                time_features = ['hour', 'dayofweek']

        elif data_type == 'physionet':
            if time_col in data.columns:
                print(f"🔍 PHYSIONET simple time processing...")
                
                try:
                    # Extract just the numeric patterns from time strings
                    def extract_time_features(time_str):
                        """Extract hour and minute from physionet time format"""
                        try:
                            if pd.isna(time_str):
                                return 0, 0
                            
                            time_str = str(time_str).strip()
                            
                            if ':' in time_str:
                                parts = time_str.split(':')
                                hours = int(parts[0]) % 24  # Wrap around 24 hours
                                minutes = int(parts[1]) if len(parts) > 1 else 0
                                return hours, minutes
                            else:
                                # If no colon, treat as total minutes
                                total_min = int(float(time_str))
                                hours = (total_min // 60) % 24
                                minutes = total_min % 60
                                return hours, minutes
                                
                        except:
                            return 0, 0  # Default fallback
                    
                    # Apply extraction
                    time_data = data[time_col].apply(extract_time_features)
                    data['hour'] = [t[0] for t in time_data]
                    data['minutes'] = [t[1] for t in time_data]
                    
                    time_features = ['hour', 'minutes']
                    
                    print(f"   ✅ Extracted time features successfully")
                    print(f"   Sample hours: {data['hour'].head().tolist()}")
                    print(f"   Sample minutes: {data['minutes'].head().tolist()}")
                    
                except Exception as e:
                    print(f"   ❌ Time feature extraction failed: {e}")
                    # Fallback: create dummy time features
                    data['hour'] = 12  # Constant hour
                    data['minutes'] = 0  # Constant minutes
                    time_features = ['hour', 'minutes']
                    print(f"   Using constant time features as fallback")
                   
        
        all_features = features + time_features
        feature_matrix = []
        target_values = []
        used_indices = []
        
        # Start from max_lags to preserve more data
        for t in range(max_lags, len(data)):
            # Current features
            current_features = data[all_features].iloc[t].values
            
            # Add multiple lags of target to capture temporal dependency
            if max_lags > 0:
                # FIXED: Create multiple lags properly
                target_lags = []
                for lag in range(1, max_lags + 1):
                    lag_value = data[target].iloc[t - lag]
                    target_lags.append(lag_value)
                
                # Combine current features with all lag features
                features_row = np.concatenate([current_features, target_lags])
            else:
                features_row = current_features
            
            feature_matrix.append(features_row)
            target_values.append(data[target].iloc[t])
            used_indices.append(data.index[t])
        
        X = np.array(feature_matrix)
        y = np.array(target_values)
        
        # Calculate actual number of lag features added
        n_lag_features = max_lags if max_lags > 0 else 0
        
        print(f"  Final feature matrix: {X.shape}")
        print(f"  Features: {len(features)} predictors + {len(time_features)} time + {n_lag_features} lags")
        
        return X, y, np.array(used_indices)


class SimpleMCMCWithPlaceholder:
    """
    Simple MCMC-MICE using lags strategy + place_holder initialization + time_series_pattern
    """
    def __init__(self, time_col='Date_Time', n_samples=12000, burn_in=None, initialization='mean'):
        self.time_col = time_col
        self.n_samples = n_samples
        self.burn_in = burn_in or max(4000, int(n_samples * 0.6))
        self.scalers = {}
        self.data_prep = SelectiveDataPreparation()
        self.initialization = initialization
        # Store time series analysis for reuse
        self.ts_analysis = None
        self._ts_analysis_computed = False

    def analyze_time_series_patterns(self, data_with_time, verbose=True):
        """Debug version of time series analysis"""
        if verbose:
            print("🔍 DEBUG: Analyzing time series patterns...")
            print(f"   Data shape: {data_with_time.shape}")
            print(f"   Time column: {self.time_col}")
            print(f"   Time column exists: {self.time_col in data_with_time.columns}")
        
        try:
            # Check if time_series_pattern function exists
            if 'time_series_pattern' not in globals():
                print("❌ time_series_pattern function not found!")
                self.ts_analysis = {}
                self._ts_analysis_computed = True
                return self.ts_analysis
                
            self.ts_analysis = time_series_pattern(data_with_time, time_col=self.time_col, value_col=None)
            self._ts_analysis_computed = True
            
            if verbose:
                print("✅ Time series analysis completed:")
                for col, analysis in self.ts_analysis.items():
                    if col != self.time_col:
                        trend_str = "✓" if analysis.get('has_trend', False) else "✗"
                        seasonal_str = "✓" if analysis.get('has_seasonality', False) else "✗"
                        period = analysis.get('seasonal_period', 'None')
                        strength = analysis.get('seasonal_strength', 0)
                        print(f"     {col}: Trend {trend_str}, Seasonality {seasonal_str} "
                            f"(period: {period}, strength: {strength:.3f})")
        
        except Exception as e:
            print(f"❌ Time series analysis failed: {type(e).__name__}: {str(e)}")
            self.ts_analysis = {}
            self._ts_analysis_computed = True
        
        return self.ts_analysis
    
        # Also add debug wrapper around place_holder function:
    def debug_place_holder(self, data, ts_analysis=None):
        """Debug wrapper for place_holder function"""
        print(f"  🎯 place_holder called:")
        print(f"     Data shape: {data.shape}")
        print(f"     TS analysis: {ts_analysis is not None}")
        print(f"     Input nulls: {data.isnull().sum().sum()}")
        
        try:
            result = place_holder(data, ts_analysis=ts_analysis)
            print(f"     Output nulls: {result.isnull().sum().sum()}")
            print(f"     Data changed: {not data.equals(result)}")
            return result
        except Exception as e:
            print(f"     ERROR: {type(e).__name__}: {str(e)}")
            raise

    def _initialize_missing_values(self, data, ts_analysis=None):
        """
        Initialize missing values using different strategies with comprehensive debugging
        """
        print(f"\n🔍 DEBUG INITIALIZATION:")
        print(f"   Method: '{self.initialization}'")
        print(f"   Data shape: {data.shape}")
        print(f"   Missing values: {data.isnull().sum().sum()}")
        print(f"   TS analysis available: {self.ts_analysis is not None}")
        
        if self.ts_analysis:
            print(f"   TS analysis keys: {list(self.ts_analysis.keys())}")
            for col, analysis in self.ts_analysis.items():
                if col != self.time_col:
                    print(f"   {col}: trend={analysis.get('has_trend', False)}, "
                        f"seasonal={analysis.get('has_seasonality', False)}")
        
        if self.initialization == 'mean':
            print("📊 Executing MEAN initialization...")
            result = self._mean_initialization(data)
            print(f"✅ Mean initialization completed - remaining nulls: {result.isnull().sum().sum()}")
            return result
            
        elif self.initialization == 'placeholder':
            print("🔄 Executing PLACEHOLDER initialization...")
            print(f"   Calling place_holder with ts_analysis: {self.ts_analysis is not None}")
            
            try:
                # Call place_holder and capture result
                result = self.debug_place_holder(data, ts_analysis=self.ts_analysis)
                remaining_nulls = result.isnull().sum().sum()
                print(f"✅ Placeholder completed successfully - remaining nulls: {remaining_nulls}")
                
                # Check if results are identical to input (would indicate no processing)
                changes_made = not data.equals(result)
                print(f"   Changes made to data: {changes_made}")
                
                if changes_made:
                    # Show which columns were modified
                    for col in data.columns:
                        if col != self.time_col:
                            orig_nulls = data[col].isnull().sum()
                            new_nulls = result[col].isnull().sum()
                            if orig_nulls != new_nulls:
                                print(f"   📈 {col}: {orig_nulls} → {new_nulls} nulls")
                
                return result
                
            except ImportError as e:
                print(f"❌ Import error in placeholder: {e}")
                print("🔧 Falling back to mean initialization...")
                return self._mean_initialization(data)
                
            except Exception as e:
                print(f"❌ Placeholder initialization FAILED: {type(e).__name__}: {str(e)}")
                print("🔧 Falling back to mean initialization...")
                return self._mean_initialization(data)
        
        else:
            raise ValueError(f"Unknown initialization method: {self.initialization}")

    def _mean_initialization(self, data):
        """Simple mean/mode imputation"""
        print(f"  Using MEAN initialization")
        imputed_data = data.copy()
        
        # Numeric columns: use mean
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != self.time_col and data[col].isnull().any():
                mean_val = data[col].mean()
                imputed_data[col] = imputed_data[col].fillna(mean_val)
        
        # Categorical columns: use mode
        cat_cols = data.select_dtypes(include=['object']).columns
        for col in cat_cols:
            if col != self.time_col and data[col].isnull().any():
                mode_val = data[col].mode()[0] if len(data[col].mode()) > 0 else 'missing'
                imputed_data[col] = imputed_data[col].fillna(mode_val)
        
        return imputed_data
    def _scale_data(self, X, y, target):
        """Scale features and target"""
        if target not in self.scalers:
            self.scalers[target] = {'X': StandardScaler(), 'y': StandardScaler()}
        Xs = self.scalers[target]['X'].fit_transform(X)
        ys = self.scalers[target]['y'].fit_transform(y.reshape(-1,1)).flatten()
        return Xs, ys

    def _unscale_predictions(self, y_pred, target):
        """Unscale predictions back to original scale"""
        return self.scalers[target]['y'].inverse_transform(y_pred.reshape(-1,1)).flatten() \
               if target in self.scalers else y_pred

    @staticmethod
    def validate_inputs(true_values, predicted_values):
        """Validate input arrays and return clean versions"""
        true_values = np.asarray(true_values).flatten()
        predicted_values = np.asarray(predicted_values).flatten()
        
        if len(true_values) != len(predicted_values):
            raise ValueError(f"Array length mismatch: {len(true_values)} vs {len(predicted_values)}")
        
        if len(true_values) == 0:
            raise ValueError("Empty arrays provided")
        
        # Remove NaN values
        valid_mask = ~(np.isnan(true_values) | np.isnan(predicted_values) | 
                      np.isinf(true_values) | np.isinf(predicted_values))
        
        if not np.any(valid_mask):
            raise ValueError("No valid (non-NaN, non-inf) values found")
        
        return true_values[valid_mask], predicted_values[valid_mask]

    @staticmethod
    def calculate_rmse(true_values, predicted_values):
        """Calculate Root Mean Square Error with robust error handling"""
        try:
            true_clean, pred_clean = SimpleMCMCWithPlaceholder.validate_inputs(true_values, predicted_values)
            return np.sqrt(np.mean((pred_clean - true_clean)**2))
        except (ValueError, ZeroDivisionError) as e:
            print(f"⚠️  RMSE calculation failed: {e}")
            return np.inf
    
    @staticmethod
    def calculate_mae(true_values, predicted_values):
        """Calculate Mean Absolute Error with robust error handling"""
        try:
            true_clean, pred_clean = SimpleMCMCWithPlaceholder.validate_inputs(true_values, predicted_values)
            return np.mean(np.abs(pred_clean - true_clean))
        except (ValueError, ZeroDivisionError) as e:
            print(f"⚠️  MAE calculation failed: {e}")
            return np.inf
    
    @staticmethod
    def calculate_mre(true_values, predicted_values, min_threshold=1e-8):
        """Calculate Mean Relative Error with robust handling of zero values"""
        try:
            true_clean, pred_clean = SimpleMCMCWithPlaceholder.validate_inputs(true_values, predicted_values)
            
            # Handle near-zero values by using a minimum threshold
            abs_true = np.abs(true_clean)
            
            # If all values are below threshold, return a reasonable MRE
            if np.all(abs_true < min_threshold):
                print(f"⚠️  All true values near zero (< {min_threshold}), using MAE instead")
                # For near-zero true values, MRE doesn't make sense, so use normalized MAE
                mae = np.mean(np.abs(pred_clean - true_clean))
                return mae * 100  # Convert to percentage-like scale
            
            # Use maximum of absolute value and threshold to avoid division by very small numbers
            denominator = np.maximum(abs_true, min_threshold)
            relative_errors = np.abs(pred_clean - true_clean) / denominator
            
            # Remove any potential infinite or very large values
            finite_mask = np.isfinite(relative_errors) & (relative_errors < 1000)  # Cap at 100,000%
            
            if not np.any(finite_mask):
                print("⚠️  No finite relative errors found")
                return np.inf
            
            return np.mean(relative_errors[finite_mask]) * 100
            
        except (ValueError, ZeroDivisionError) as e:
            print(f"⚠️  MRE calculation failed: {e}")
            return np.inf
    
    @staticmethod
    def calculate_nrmse(true_values, predicted_values, method='std', min_std=1e-8):
        """Calculate Normalized RMSE with robust handling of zero variance"""
        try:
            true_clean, pred_clean = SimpleMCMCWithPlaceholder.validate_inputs(true_values, predicted_values)
            
            rmse = np.sqrt(np.mean((true_clean - pred_clean)**2))
            
            if method == 'range':
                denom = np.max(true_clean) - np.min(true_clean)
                if denom < min_std:
                    print(f"⚠️  True values have no range (constant), using RMSE: {rmse:.6f}")
                    return rmse  # Return unnormalized RMSE for constant values
            elif method == 'std':
                denom = np.std(true_clean)
                if denom < min_std:
                    print(f"⚠️  True values have zero variance, using RMSE: {rmse:.6f}")
                    return rmse  # Return unnormalized RMSE for constant values
            else:
                raise ValueError("method must be 'range' or 'std'")
            
            return rmse / denom
            
        except (ValueError, ZeroDivisionError) as e:
            print(f"⚠️  NRMSE calculation failed: {e}")
            return np.inf
    
    
    def calculate_all_metrics(self, true_values, predicted_values):
        """Calculate all metrics at once with robust error handling"""
        metrics = {
            'rmse': self.calculate_rmse(true_values, predicted_values),
            'mae': self.calculate_mae(true_values, predicted_values),
            'mre': self.calculate_mre(true_values, predicted_values),
            'nrmse': self.calculate_nrmse(true_values, predicted_values)
        }
        
        # Log any infinite metrics
        infinite_metrics = [k for k, v in metrics.items() if np.isinf(v)]
        if infinite_metrics:
            print(f"⚠️  Infinite metrics detected: {infinite_metrics}")
        
        return metrics

    def run_mcmc_with_separated_phases(mcmc_mice, X_obs_scaled, y_obs_scaled, X_miss_scaled, y_miss, 
                           target_name, mcmc_seed=None, verbose=False, show_convergence_plots=False,
                           output_dir="./plots_RWM", run_number=None):
        """
        UPDATED: Three-phase approach - dual chains for diagnostics + fresh chain for predictions
        PHASE 1: Run dual chains for convergence diagnostics only
        PHASE 2: Evaluate diagnostics (but don't fail on poor convergence)  
        PHASE 3: Run fresh single chain SPECIFICALLY for predictions
        """
        def save_summary_as_image(summary_df, filepath):
            fig, ax = plt.subplots(figsize=(12, len(summary_df)*0.4 + 1))
            ax.axis('off')
            table = ax.table(
                cellText=summary_df.round(4).values,
                colLabels=summary_df.columns,
                rowLabels=summary_df.index,
                loc='center',
                cellLoc='center'
            )
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.2)
            plt.savefig(filepath, bbox_inches='tight', dpi=300)
            plt.close()
            print(f"📷 Saved convergence summary image: {filepath}")

        if verbose:
            print(f"    🔬 Two-phase MCMC for {target_name}...")
            print(f"    Phase 1: Convergence check (dual-chain)")
            print(f"    Phase 2: Diagnostics evaluation") 
        
        # ========================================
        # PHASE 1: CONVERGENCE DIAGNOSTICS ONLY
        # ========================================
        if verbose:
            print(f"    📊 Phase 1: Running convergence diagnostics...")
        
        # Set seeds for reproducibility
        if mcmc_seed is not None:
            chain1_seed = mcmc_seed
            chain2_seed = mcmc_seed + 50000
            
        else:
            chain1_seed = None
            chain2_seed = None
        
        convergence_summary = {'convergence_status': 'UNKNOWN', 'rhat_max': np.inf}
        visualizer = MCMCMICEVisualizer()

            # Run two chains ONLY for diagnostics (don't use their predictions)
        mcmc_chain1 = MCMC_CHAIN(
            n_samples=mcmc_mice.n_samples,
            n_burnin=mcmc_mice.burn_in,
            x_data=X_obs_scaled,
            y_data=y_obs_scaled,
            x_test=X_miss_scaled,
            y_test=y_miss,
            seed=chain1_seed,
            variable_name=target_name,
            verbose=False,
            use_adaptive=True,
            sampler_type="RWM"
        )
        results_chain1, predict_chain1 = mcmc_chain1.sampler()  # DISCARD predictions
        
        # Show convergence plots if requested
        if show_convergence_plots:
            print(f"    📊 Showing convergence plots for {target_name} (Chain 1)...")
            visualizer.convergence_plots(mcmc_chain1, chain_label="chain1", target_col=target_name,
                    run_number=1)
        for param in ['tau','rmse', 'w0', 'w1']:
            if show_convergence_plots and param in results_chain1.columns:
                visualizer.plot_credible_interval_trace(
                    results_chain1[param].values,
                    param_name=param,
                    run_number=1
                )

        mcmc_chain2 = MCMC_CHAIN(
            n_samples=mcmc_mice.n_samples,
            n_burnin=mcmc_mice.burn_in,
            x_data=X_obs_scaled,
            y_data=y_obs_scaled,
            x_test=X_miss_scaled,
            y_test=y_miss,
            seed=chain2_seed,
            variable_name=target_name,
            verbose=False,
            use_adaptive=True,
            sampler_type="RWM"
        )
        results_chain2, predict_chain2 = mcmc_chain2.sampler()  # DISCARD predictions
        
        if show_convergence_plots:
            print(f"    📊 Showing convergence plots for {target_name} (Chain 2)...")
            visualizer.convergence_plots(mcmc_chain2, chain_label="chain2", target_col=target_name,
                    run_number=1)
        for param in ['tau','rmse', 'w0', 'w1']:
            if show_convergence_plots and param in results_chain2.columns:
                visualizer.plot_credible_interval_trace(
                    results_chain2[param].values,
                    param_name=param,
                    run_number=1
                )

  

        # ========================================
        # PHASE 2: DIAGNOSTICS EVALUATION (NON-BLOCKING)
        # ========================================
        try:
            # Convert to ArviZ format
            res_dict_chain1 = results_chain1.to_dict(orient='list')
            res_dict_chain2 = results_chain2.to_dict(orient='list')
            
            az_results = az.from_dict({
                par: np.vstack([res_dict_chain1[par], res_dict_chain2[par]]) 
                for par in res_dict_chain1
            })
            
            summary = az.summary(az_results)
            if verbose:
                print(f"    Convergence diagnostics summary: \n{summary}")
            
            convergence_summary = {
                'rhat_values': summary['r_hat'].to_dict(),
                'rhat_max': summary['r_hat'].max(),
                'rhat_mean': summary['r_hat'].mean(),
                'ess_bulk_min': summary['ess_bulk'].min() if 'ess_bulk' in summary.columns else None,
                'ess_tail_min': summary['ess_tail'].min() if 'ess_tail' in summary.columns else None,
                'convergence_status': 'GOOD' if summary['r_hat'].max() < 1.05 else 
                                    'MODERATE' if summary['r_hat'].max() < 1.1 else 'POOR',
                'summary_table': summary
            }
            
            if verbose:
                print(f"    📊 Convergence Results:")
                print(f"       Max R-hat: {convergence_summary['rhat_max']:.4f}")
                print(f"       Status: {convergence_summary['convergence_status']}")

                # Show detailed R-hat for each parameter
                for param, rhat in convergence_summary['rhat_values'].items():
                    status = "✅" if rhat < 1.05 else "⚠️" if rhat < 1.1 else "❌"
                    print(f"        {param}: {rhat:.4f} {status}")

            # ✅ Save convergence summary as image
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                fname_parts = ["convergence_summary"]
                if target_name: fname_parts.append(target_name)
                if run_number is not None: fname_parts.append(f"run{run_number}")
                fname_parts.append("dualchain")
                filename = "_".join(fname_parts) + ".png"
                filepath = os.path.join(output_dir, filename)
                save_summary_as_image(summary, filepath)
                
        except Exception as e:
            if verbose:
                print(f"    ❌ Convergence diagnostics failed: {e}")
            convergence_summary = {
                'rhat_values': np.inf,
                'convergence_status': 'ERROR',
                'error': str(e)
            }
        if verbose:         
            if convergence_summary['convergence_status'] == 'POOR':
                print(f"    ⚠️  Poor convergence detected, but continuing with predictions anyway")
            else:
                print(f"    ✅ Convergence: {convergence_summary['convergence_status']}")

        # ✅ Return diagnostic chain results and predictions
        return results_chain1, predict_chain1, convergence_summary