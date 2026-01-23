from packages import *
from SimpleMCMC import *
from MCMC_CHAIN import *
from Visualisation import *

def run_single_mcmc(mcmc_mice, data_with_time, missing_data, subdata, target_col, missing_indices, true_values, 
                    n_posterior_samples=5, max_iter=5, run_seed=None, verbose=False, 
                    show_convergence_plots=False, output_dir="./plots_RWM", run_number=None):
    """
    Run MCMC with separated convergence check and fresh prediction phase.
    """
   #check if MCMC_MICE is initialised
    if verbose:
        print(f"Running separated-phase MCMC for {target_col}...")
        print(f"Phase approach: Convergence check → Fresh prediction chain")
        if show_convergence_plots:
            print(f"Convergence plots will be displayed for this run")
    

    try:
        # Set seed for reproducibility
        if run_seed is not None:
            np.random.seed(run_seed)
        
        # Step 1: Initialize missing values (same as before)
        if verbose:
            print(f"        Step 1: Initializing with {mcmc_mice.initialization}...")
        
        if mcmc_mice.initialization == 'placeholder':
            if not mcmc_mice._ts_analysis_computed:
                mcmc_mice.analyze_time_series_patterns(data_with_time, verbose=False)
        
        imputed_data = mcmc_mice._initialize_missing_values(missing_data, ts_analysis=mcmc_mice.ts_analysis)
        
        # Step 2: Run MCMC-MICE iterations (same as before)
        if verbose:
            print(f"        Step 2: Running {max_iter} MCMC-MICE iterations...")
        
        missing_mask = missing_data.isnull()
        missing_cols = [col for col in missing_data.columns if col != mcmc_mice.time_col and missing_mask[col].any()]
        all_vars = [col for col in missing_data.columns if col != mcmc_mice.time_col]
        
        final_mcmc_results = None
        
        # MICE-style iterations
        for iteration in range(max_iter):
            if verbose and iteration == 0:
                print(f"          MCMC-MICE iteration {iteration + 1}/{max_iter}...")
            
            vars_to_impute = missing_cols.copy()
            if iteration > 0:
                np.random.shuffle(vars_to_impute)
            
            for var_idx, current_var in enumerate(vars_to_impute):
                observed = ~missing_mask[current_var]
                if observed.sum() < 10:
                    continue
                
                predictors = [col for col in all_vars if col != current_var]
                
                try:
                    # Prepare data
                    X, y, used_indices = mcmc_mice.data_prep.prepare_selective_data(
                        imputed_data, current_var, predictors, mcmc_mice.time_col, max_lags=2, data_type='air'
                    )
                    
                    if len(X) == 0 or len(y) == 0:
                        continue
                    
                    # Identify missing values in prepared data
                    used_indices_set = set(used_indices)
                    original_missing_set = set(missing_mask[current_var][missing_mask[current_var]].index)
                    matching_indices = sorted(original_missing_set.intersection(used_indices_set))
                    true_values = subdata.loc[matching_indices, target_col].values
                    print("Total missing (artificial):", len(original_missing_set))  # Or however you created it
                    print("Used in MCMC prediction:", len(matching_indices))
                    print("True values retrieved for comparison:", len(true_values))
                                    
                    aligned_missing_mask = np.array([idx in original_missing_set for idx in used_indices])
                    aligned_observed_mask = ~aligned_missing_mask
                    
                    X_obs = X[aligned_observed_mask]
                    y_obs = y[aligned_observed_mask]
                    
                    if len(y_obs) < 5:
                        continue

                    should_show_plots = (show_convergence_plots and current_var == target_col and iteration == 0)
                    mcmc_seed_val = (run_seed + iteration * 1000 + var_idx * 100 + 50000) if run_seed else None
                    
                    visualizer = MCMCMICEVisualizer()
                    validation_passed = True
                    # First: Validate model quality using observed data split
                    if len(y_obs) > 20:
                        print(f"        📊 Validating model quality for {current_var}...")
                        
                        split_idx = int(0.8 * len(y_obs))
                        X_train, X_val = X_obs[:split_idx], X_obs[split_idx:]
                        y_train, y_val = y_obs[:split_idx], y_obs[split_idx:]
                        
                        X_train_scaled, y_train_scaled = mcmc_mice._scale_data(X_train, y_train, current_var)
                        X_val_scaled = mcmc_mice.scalers[current_var]['X'].transform(X_val)
                        
                        # Validation run
                        val_results, val_pred, val_convergence = mcmc_mice.run_mcmc_with_separated_phases(
                            X_obs_scaled=X_train_scaled,
                            y_obs_scaled=y_train_scaled,
                            X_miss_scaled=X_val_scaled,
                            y_miss=y_val,  # Known values for validation
                            target_name=f"{current_var}_validation",
                            mcmc_seed=mcmc_seed_val,
                            verbose=False,
                            show_convergence_plots=False,
                            run_number=1
                        )                 
                          # Check validation performance
                        if val_pred and 'test_pred' in val_pred:
                            val_predictions = np.mean(val_pred['test_pred'], axis=0)
                            print(f"y_val shape: {y_val.shape}")
                            # IMPORTANT: Unscale predictions before comparison
                            val_predictions_unscaled = mcmc_mice._unscale_predictions(val_predictions, current_var)
                            print(f"val_predictions_unscaled shape: {val_predictions_unscaled.shape}")
                            val_rmse = np.sqrt(np.mean((y_val - val_predictions_unscaled)**2))
                            print(f"  Validation RMSE for {current_var}: {val_rmse:.4f}")

                            test_pred_samples_unscaled = np.array([
                                mcmc_mice._unscale_predictions(sample, current_var)
                                for sample in val_pred['test_pred']
                            ])
         
                            # Plot validation predictions with CI
                            visualizer.plot_prediction_with_ci(
                                    y_true=y_val,  # original scale
                                    train_sims=test_pred_samples_unscaled,
                                    title=f"{current_var} Test",
                                    save_path=os.path.join(output_dir, f"trainpred_fit_ci_{current_var}_run{run_number}.png"),
                                    run_number=1
                                )


                            if 'train_sim' in val_pred:
                                # Unscale each row in train_sim
                                print(f"🔁 Generating train_sim CI plot for {current_var}")
                                train_sim_unscaled = np.array([
                                    mcmc_mice._unscale_predictions(row, current_var)
                                    for row in val_pred['train_sim']
                                ])


                                visualizer.plot_prediction_with_ci(
                                    y_true=y_train,  # original scale
                                    train_sims=train_sim_unscaled,
                                    title=f"{current_var} Train",
                                    save_path=os.path.join(output_dir, f"train_fit_ci_{current_var}_run{run_number}.png"),
                                    run_number=1
                                )

                            if 'test_sim' in val_pred:
                                # Unscale each row in test_sim
                                print(f"🔁 Generating test_sim CI plot for {current_var}")
                                test_sim_unscaled = np.array([
                                    mcmc_mice._unscale_predictions(row, current_var)
                                    for row in val_pred['test_sim']
                                ])

                                visualizer.plot_prediction_with_ci(
                                    y_true=y_val,  # original scale
                                    train_sims=test_sim_unscaled,
                                    title=f"{current_var} Test Fit with 95% CI",
                                    save_path=os.path.join(output_dir, f"test_fit_ci_{current_var}_run{run_number}.png"),
                                    run_number=1
                                )
                            # Set validation threshold (adjust as needed)
                            validation_threshold = np.std(y_val) * 2.0
                            
                            if val_rmse > validation_threshold:
                                if verbose:
                                    print(f"        ❌ Validation FAILED: RMSE {val_rmse:.4f} > {validation_threshold:.4f}")
                                validation_passed = False
                            else:
                                if verbose:
                                    print(f"        ✅ Validation PASSED: RMSE {val_rmse:.4f}")
                        else:
                            if verbose:
                                print(f"        ❌ Validation FAILED: No predictions generated")
                            validation_passed = False
                    
                    # Skip this variable if validation failed
                    if not validation_passed:
                        if verbose:
                            print(f"        ⏭️  Skipping {current_var} due to validation failure")
                        continue    
                    
                    # Scale data
                    X_obs_scaled, y_obs_scaled = mcmc_mice._scale_data(X_obs, y_obs, current_var)

                    X_miss = X[aligned_missing_mask]
                    if len(X_miss) == 0:
                        continue
                    X_miss_scaled = mcmc_mice.scalers[current_var]['X'].transform(X_miss)
                    y_miss = np.zeros(X_miss.shape[0])
                    
                    mcmc_seed = run_seed + iteration * 1000 + var_idx * 100 if run_seed else None
                    # Only show convergence plots for target variable in first iteration of first run
                    results_df, pred_dict, convergence_diag = mcmc_mice.run_mcmc_with_separated_phases(
                        X_obs_scaled=X_obs_scaled,
                        y_obs_scaled=y_obs_scaled,
                        X_miss_scaled=X_miss_scaled,
                        y_miss=y_miss,
                        target_name=current_var,
                        mcmc_seed=mcmc_seed,
                        verbose=(verbose and iteration == 0 and var_idx == 0),
                        show_convergence_plots=should_show_plots,
                        run_number=1
                    )#(verbose and iteration == 0 and var_idx == 0),
                    
                    # Check if MCMC succeeded
                    if results_df is None or pred_dict is None:
                        if verbose:
                            print(f"❌ {current_var}: MCMC failed (poor convergence)")
                        continue
                    
                    # Store MCMC results for target variable
                    if current_var == target_col:
                        final_mcmc_results = {
                            'results_df': results_df,
                            'pred_dict': pred_dict,
                            'X_miss_scaled': X_miss_scaled,
                            'missing_indices_in_prepared': np.where(aligned_missing_mask)[0],
                            'used_indices': used_indices,
                            'target_col': current_var,
                            'convergence_diagnostics': convergence_diag,
                            'original_missing_indices': missing_indices,
                            'aligned_missing_mask': aligned_missing_mask,
                            'matching_indices': matching_indices
                        }
                        
                        if verbose:
                            conv_status = convergence_diag['convergence_status']
                            rhat_max = convergence_diag.get('rhat_max', 'N/A')
                            print(f"        ✅ Good convergence for {current_var} (R-hat: {rhat_max:.4f})")
                    
                    # Update imputed data for next iteration
                    if 'test_pred' in pred_dict and pred_dict['test_pred'] is not None:
                        pred_y_missing = pred_dict['test_pred']
                        if not (np.any(np.isnan(pred_y_missing)) or np.any(np.isinf(pred_y_missing))):
                            mean_pred = pred_y_missing.mean(axis=0)
                            unscaled_pred = mcmc_mice._unscale_predictions(mean_pred, current_var)
                            for i, idx in enumerate(matching_indices):
                                imputed_data.loc[idx, current_var] = unscaled_pred[i]

                except Exception as e:
                    if verbose:
                        print(f"        Error in MCMC for {current_var}: {str(e)}")
                    continue

        if final_mcmc_results is None:
            if verbose:
                print(f"        No MCMC results available for {target_col}")
            return np.inf, np.inf, np.inf, np.inf, np.inf

        if verbose:
            print(f"        Step 3: Generating {n_posterior_samples} imputations from fresh chain...")

        pred_dict = final_mcmc_results['pred_dict']
        matching_indices = final_mcmc_results['matching_indices']

        if 'test_pred' not in pred_dict or pred_dict['test_pred'] is None:
            if verbose:
                print(f"          No predictions available from fresh chain")
            return np.inf, np.inf, np.inf, np.inf, np.inf

        posterior_predictions = pred_dict['test_pred']

        rmse_list = []
        mae_list = []
        mre_list = []
        nrmse_list = []
        imputed_datasets = []

        for imputation_idx in range(n_posterior_samples):
            sample_idx = np.random.randint(0, posterior_predictions.shape[0])
            sampled_predictions = posterior_predictions[sample_idx]

            unscaled_pred = mcmc_mice._unscale_predictions(sampled_predictions, target_col)

            if verbose and imputation_idx == 0:
                print(f"        📊 Prediction shapes:")
                print(f"           True values: {len(true_values)}")
                print(f"           Predictions: {len(unscaled_pred)}")

            if len(matching_indices) != len(unscaled_pred):
                print(f"❌ Mismatch: true={len(matching_indices)} vs pred={len(unscaled_pred)}")
                return np.inf, np.inf, np.inf, np.inf, np.inf

            true_values_aligned = subdata.loc[matching_indices, target_col].values
            metrics = mcmc_mice.calculate_all_metrics(true_values_aligned, unscaled_pred)

            rmse_list.append(metrics['rmse'])
            mae_list.append(metrics['mae'])
            mre_list.append(metrics['mre'])
            nrmse_list.append(metrics['nrmse'])

            full_imputed_dataset = imputed_data.copy()
            for i, idx in enumerate(matching_indices):
                full_imputed_dataset.loc[idx, target_col] = unscaled_pred[i]

            imputed_datasets.append(full_imputed_dataset.copy())

        if len(rmse_list) > 0:
            avg_rmse = np.mean(rmse_list)
            avg_mae = np.mean(mae_list)
            avg_mre = np.mean(mre_list)
            avg_nrmse = np.mean(nrmse_list)

            if verbose:
                print(f"✅ Separated-phase MCMC completed successfully")
                print(f"📈 Final metrics - RMSE: {avg_rmse:.4f}")

            return imputed_datasets, avg_rmse, avg_mae, avg_mre, avg_nrmse
        else:
            return np.inf, np.inf, np.inf, np.inf, np.inf

    except Exception as e:
        if verbose:
            print(f"❌ Separated-phase MCMC failed: {str(e)}")
            import traceback
            traceback.print_exc()
        return np.inf, np.inf, np.inf, np.inf, np.inf
