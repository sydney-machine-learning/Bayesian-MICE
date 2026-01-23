from packages import *
from SimpleMCMC import *
from MCMC_CHAIN import *
from Visualisation import *
from Run_Single_MCMC import * 

def enhanced_comparison_with_runs(complete_data, missing_data, data_with_time, 
                                        time_col='Date_Time', n_runs=30, n_imputations=5, 
                                        max_iter=5, visualize_runs=[1, 15, 30], 
                                        save_plots=False, output_dir='./plots'):
    """
    ENHANCED VERSION: 30 independent experimental runs with comprehensive timing tracking
    - Tracks execution time for each method and overall experiment
    - Provides detailed timing analysis and performance/cost trade-offs
    - Maintains original experimental design with added efficiency metrics
    """
    
    if save_plots:
        import os
        os.makedirs(output_dir, exist_ok=True)
    
    print("="*100)
    print(f"ENHANCED 30 INDEPENDENT EXPERIMENTAL RUNS WITH TIMING ANALYSIS")
    print("Each run: MICE vs MCMC_MICE_V1 vs MCMC_MICE_V2 (5 imputations each)")
    print("="*100)
    
    # Initialize timing tracking
    experiment_start_time = time.time()
    
    # Timing storage
    timing_results = {
        'total_experiment_time': 0,
        'per_run_times': [],
        'method_times': {
            'MICE': {'per_run': [], 'per_column': defaultdict(list), 'total': 0},
            'MCMC_MICE_V1': {'per_run': [], 'per_column': defaultdict(list), 'total': 0},
            'MCMC_MICE_V2': {'per_run': [], 'per_column': defaultdict(list), 'total': 0}
        },
        'setup_times': [],
        'cleanup_times': []
    }
    
    # Find missing values
    missing_mask = missing_data.isnull() & ~complete_data.isnull()
    numeric_cols = missing_data.select_dtypes(include=[np.number]).columns.tolist()
    if time_col in numeric_cols:
        numeric_cols.remove(time_col)
    
    cols_with_missing = [col for col in numeric_cols if missing_mask[col].sum() > 0]
    print(f"Testing {len(cols_with_missing)} columns: {cols_with_missing}")
    print(f"Estimated total operations: {n_runs} runs × {len(cols_with_missing)} columns × 3 methods = {n_runs * len(cols_with_missing) * 3}")
    
    # Storage for results (same as original)
    all_results = {col: {'MICE_rmse': [], 'MCMC_MICE_V1_rmse': [], 'MCMC_MICE_V2_rmse': [],
                        'MICE_mae': [], 'MCMC_MICE_V1_mae': [], 'MCMC_MICE_V2_mae': [],
                        'MICE_mre': [], 'MCMC_MICE_V1_mre': [], 'MCMC_MICE_V2_mre': [],
                        'MICE_nrmse': [], 'MCMC_MICE_V1_nrmse': [], 'MCMC_MICE_V2_nrmse': []} 
                  for col in cols_with_missing}
    
    # Initialize visualizer
    setup_start = time.time()
    visualizer = MCMCMICEVisualizer(time_col=time_col)
    setup_time = time.time() - setup_start
    timing_results['setup_times'].append(setup_time)
    
    # Run 30 independent experiments
    for run in range(n_runs):
        run_start_time = time.time()
        
        print(f"\n{'='*80}")
        print(f"INDEPENDENT EXPERIMENTAL RUN {run+1}/{n_runs}")
        print(f"Estimated completion: {((time.time() - experiment_start_time) / (run + 1e-6)) * (n_runs - run) / 60:.1f} minutes remaining")
        print('='*80)
        
        # 🔧 KEY FIX: Create fresh MCMC instances for EACH run
        instance_creation_start = time.time()
        print(f"🔄 Creating fresh MCMC instances for run {run+1}...")
        mcmc_mice_mean = SimpleMCMCWithPlaceholder(
            time_col=time_col,
            n_samples=30000,
            initialization='mean'
        )
        
        mcmc_mice_placeholder = SimpleMCMCWithPlaceholder(
            time_col=time_col,
            n_samples=30000,
            initialization='placeholder'
        )
        instance_creation_time = time.time() - instance_creation_start
        
        # 🔧 Reset random state for each independent run
        base_seed = 1000 + run * 100000  # Large separation between runs
        np.random.seed(base_seed)
        
        # Storage for this run's visualization and timing
        run_imputed_datasets = {}
        run_predictions = {}
        run_method_times = {'MICE': 0, 'MCMC_MICE_V1': 0, 'MCMC_MICE_V2': 0}
        
        for col_idx, target_col in enumerate(cols_with_missing):
            print(f"\nProcessing {target_col} (Column {col_idx+1}/{len(cols_with_missing)})")
            
            # Get missing positions and true values
            col_missing_mask = missing_mask[target_col]
            missing_indices = missing_data.index[col_missing_mask]
            true_values = complete_data.loc[missing_indices, target_col]
            
            if len(missing_indices) == 0:
                continue
            
            print(f"  Missing values to predict: {len(missing_indices)}")
            
            # Storage for this column's results
            col_imputed_datasets = {}
            col_predictions = {}
            
            # ==========================================
            # MICE APPROACH (with timing)
            # ==========================================
            print(f"  🐭 Running MICE (5 imputations)...")
            mice_start_time = time.time()
            
            mice_rmse_list = []
            mice_mae_list = []
            mice_mre_list = []
            mice_nrmse_list = []
            mice_imputed_datasets = []
            mice_predictions = []
            
            for imp in range(n_imputations):
                imp_start = time.time()
                mice_seed = base_seed + col_idx * 1000 + imp * 100
                imputer = IterativeImputer(
                    max_iter=max_iter,
                    random_state=mice_seed,
                    sample_posterior=True,
                    n_nearest_features=min(10, len(numeric_cols)-1),
                    verbose=0
                )
                
                mice_imputed = imputer.fit_transform(missing_data[numeric_cols])
                mice_df = pd.DataFrame(mice_imputed, columns=numeric_cols, index=missing_data.index)
                mice_pred = mice_df.loc[missing_indices, target_col].values
                
                mice_imputed_datasets.append(mice_df)
                mice_predictions.append(mice_pred)
                
                # Calculate metrics
                metrics = mcmc_mice_mean.calculate_all_metrics(true_values.values, mice_pred)
                mice_rmse_list.append(metrics['rmse'])
                mice_mae_list.append(metrics['mae'])
                mice_mre_list.append(metrics['mre'])
                mice_nrmse_list.append(metrics['nrmse'])
                
                if imp == 0:  # Log timing for first imputation
                    print(f"    First MICE imputation: {time.time() - imp_start:.2f}s")
            
            mice_total_time = time.time() - mice_start_time
            run_method_times['MICE'] += mice_total_time
            timing_results['method_times']['MICE']['per_column'][target_col].append(mice_total_time)
            
            # Store averaged MICE results for visualization
            mice_avg_imputed = mice_imputed_datasets[0].copy()
            mice_avg_pred = np.mean(mice_predictions, axis=0)
            col_imputed_datasets['MICE'] = mice_avg_imputed
            col_predictions['MICE'] = mice_avg_pred
            
            # Average metrics
            mice_avg_rmse = np.mean(mice_rmse_list)
            mice_avg_mae = np.mean(mice_mae_list)
            mice_avg_mre = np.mean(mice_mre_list)
            mice_avg_nrmse = np.mean(mice_nrmse_list)
            
            print(f"    ✅ MICE completed in {mice_total_time:.2f}s (avg: {mice_total_time/n_imputations:.2f}s/imputation)")
            
            # ==========================================
            # MCMC-MICE WITH MEAN INITIALIZATION (FIXED + TIMING)
            # ==========================================
            print(f"  🔬 Running MCMC-MICE with MEAN initialization...")
            mcmc_mean_start_time = time.time()
            
            try:
                # 🔧 CRITICAL FIX: Clear any accumulated state before each variable
                mcmc_mice_mean.scalers = {}  # Clear accumulated scalers
                mcmc_mice_mean.ts_analysis = None  # Clear cached time series analysis
                mcmc_mice_mean._ts_analysis_computed = False
                
                run_seed_mean = base_seed + 10000 + col_idx * 1000
                
                # 🔧 Use fresh data copies to avoid mutation
                data_with_time_copy = data_with_time.copy()
                missing_data_copy = missing_data.copy()
                complete_data_copy = complete_data.copy()
                
                # Use your existing run_single_mcmc function
                mcmc_mean_result = run_single_mcmc(
                    mcmc_mice=mcmc_mice_mean,
                    data_with_time=data_with_time_copy,
                    missing_data=missing_data_copy,
                    subdata=complete_data_copy,
                    target_col=target_col,
                    missing_indices=missing_indices,
                    true_values=true_values,
                    n_posterior_samples=n_imputations,
                    max_iter=max_iter,
                    run_seed=run_seed_mean,
                    verbose=(run == 0 and col_idx == 0),
                    show_convergence_plots=(run == 0 and col_idx == 0),
                    run_number=1  # Only show plots for first run, first column
                )
                
                mcmc_mean_total_time = time.time() - mcmc_mean_start_time
                run_method_times['MCMC_MICE_V1'] += mcmc_mean_total_time
                timing_results['method_times']['MCMC_MICE_V1']['per_column'][target_col].append(mcmc_mean_total_time)
                
                if len(mcmc_mean_result) == 5:  # Successful run
                    mcmc_mean_imputed_datasets, mcmc_mean_avg_rmse, mcmc_mean_avg_mae, mcmc_mean_avg_mre, mcmc_mean_avg_nrmse = mcmc_mean_result
                    
                    if mcmc_mean_avg_rmse != np.inf and len(mcmc_mean_imputed_datasets) > 0:
                        col_imputed_datasets['MCMC_MICE_V1'] = mcmc_mean_imputed_datasets[0]
                        mcmc_mean_pred = mcmc_mean_imputed_datasets[0].loc[missing_indices, target_col].values
                        col_predictions['MCMC_MICE_V1'] = mcmc_mean_pred
                        print(f"    ✅ MCMC_MICE_V1 completed successfully in {mcmc_mean_total_time:.2f}s")
                    else:
                        print(f"    ❌ MCMC_MICE_V1 failed - invalid results (took {mcmc_mean_total_time:.2f}s)")
                        mcmc_mean_avg_rmse = np.inf
                        mcmc_mean_avg_mae = np.inf
                        mcmc_mean_avg_mre = np.inf
                        mcmc_mean_avg_nrmse = np.inf
                else:
                    print(f"    ❌ MCMC_MICE_V1 failed - wrong return format (took {mcmc_mean_total_time:.2f}s)")
                    mcmc_mean_avg_rmse = np.inf
                    mcmc_mean_avg_mae = np.inf
                    mcmc_mean_avg_mre = np.inf
                    mcmc_mean_avg_nrmse = np.inf
                
            except Exception as e:
                mcmc_mean_total_time = time.time() - mcmc_mean_start_time
                run_method_times['MCMC_MICE_V1'] += mcmc_mean_total_time
                timing_results['method_times']['MCMC_MICE_V1']['per_column'][target_col].append(mcmc_mean_total_time)
                print(f"    ❌ MCMC_MICE_V1 failed for {target_col}: {str(e)} (took {mcmc_mean_total_time:.2f}s)")
                mcmc_mean_avg_rmse = np.inf
                mcmc_mean_avg_mae = np.inf
                mcmc_mean_avg_mre = np.inf
                mcmc_mean_avg_nrmse = np.inf
            
            # ==========================================
            # MCMC-MICE WITH PLACEHOLDER INITIALIZATION (FIXED + TIMING)
            # ==========================================
            print(f"  🎯 Running MCMC_MICE_V2 with PLACEHOLDER initialization...")
            mcmc_placeholder_start_time = time.time()
            
            try:
                # 🔧 CRITICAL FIX: Clear any accumulated state before each variable
                mcmc_mice_placeholder.scalers = {}  # Clear accumulated scalers
                mcmc_mice_placeholder.ts_analysis = None  # Clear cached time series analysis
                mcmc_mice_placeholder._ts_analysis_computed = False
                
                run_seed_placeholder = base_seed + 20000 + col_idx * 1000
                
                # 🔧 Use fresh data copies to avoid mutation
                data_with_time_copy2 = data_with_time.copy()
                missing_data_copy2 = missing_data.copy()
                complete_data_copy2 = complete_data.copy()
                # Use your existing run_single_mcmc function
                mcmc_placeholder_result = run_single_mcmc(
                    mcmc_mice=mcmc_mice_placeholder,
                    data_with_time=data_with_time_copy2,
                    missing_data=missing_data_copy2,
                    subdata=complete_data_copy2,
                    target_col=target_col,
                    missing_indices=missing_indices,
                    true_values=true_values,
                    n_posterior_samples=n_imputations,
                    max_iter=max_iter,
                    run_seed=run_seed_placeholder,
                    verbose=(run == 0 and col_idx == 0),
                    show_convergence_plots=(run == 0 and col_idx == 0),
                    run_number=1  # Only show plots for first run, first column
                )
                
                mcmc_placeholder_total_time = time.time() - mcmc_placeholder_start_time
                run_method_times['MCMC_MICE_V2'] += mcmc_placeholder_total_time
                timing_results['method_times']['MCMC_MICE_V2']['per_column'][target_col].append(mcmc_placeholder_total_time)
                
                if len(mcmc_placeholder_result) == 5:  # Successful run
                    mcmc_placeholder_imputed_datasets, mcmc_placeholder_avg_rmse, mcmc_placeholder_avg_mae, mcmc_placeholder_avg_mre, mcmc_placeholder_avg_nrmse = mcmc_placeholder_result
                    
                    if mcmc_placeholder_avg_rmse != np.inf and len(mcmc_placeholder_imputed_datasets) > 0:
                        col_imputed_datasets['MCMC_MICE_V2'] = mcmc_placeholder_imputed_datasets[0]
                        mcmc_placeholder_pred = mcmc_placeholder_imputed_datasets[0].loc[missing_indices, target_col].values
                        col_predictions['MCMC_MICE_V2'] = mcmc_placeholder_pred
                        print(f"    ✅ MCMC_MICE_V2 completed successfully in {mcmc_placeholder_total_time:.2f}s")
                    else:
                        print(f"    ❌ MCMC_MICE_V2 failed - invalid results (took {mcmc_placeholder_total_time:.2f}s)")
                        mcmc_placeholder_avg_rmse = np.inf
                        mcmc_placeholder_avg_mae = np.inf
                        mcmc_placeholder_avg_mre = np.inf
                        mcmc_placeholder_avg_nrmse = np.inf
                else:
                    print(f"    ❌ MCMC_MICE_V2 failed - wrong return format (took {mcmc_placeholder_total_time:.2f}s)")
                    mcmc_placeholder_avg_rmse = np.inf
                    mcmc_placeholder_avg_mae = np.inf
                    mcmc_placeholder_avg_mre = np.inf
                    mcmc_placeholder_avg_nrmse = np.inf
                
            except Exception as e:
                mcmc_placeholder_total_time = time.time() - mcmc_placeholder_start_time
                run_method_times['MCMC_MICE_V2'] += mcmc_placeholder_total_time
                timing_results['method_times']['MCMC_MICE_V2']['per_column'][target_col].append(mcmc_placeholder_total_time)
                print(f"    ❌ MCMC_MICE_V2 failed for {target_col}: {str(e)} (took {mcmc_placeholder_total_time:.2f}s)")
                mcmc_placeholder_avg_rmse = np.inf
                mcmc_placeholder_avg_mae = np.inf
                mcmc_placeholder_avg_mre = np.inf
                mcmc_placeholder_avg_nrmse = np.inf
            
            # Store results for this run (same as original)
            all_results[target_col]['MICE_rmse'].append(mice_avg_rmse)
            all_results[target_col]['MCMC_MICE_V1_rmse'].append(mcmc_mean_avg_rmse)
            all_results[target_col]['MCMC_MICE_V2_rmse'].append(mcmc_placeholder_avg_rmse)
            
            all_results[target_col]['MICE_mae'].append(mice_avg_mae)
            all_results[target_col]['MCMC_MICE_V1_mae'].append(mcmc_mean_avg_mae)
            all_results[target_col]['MCMC_MICE_V2_mae'].append(mcmc_placeholder_avg_mae)
            
            all_results[target_col]['MICE_mre'].append(mice_avg_mre)
            all_results[target_col]['MCMC_MICE_V1_mre'].append(mcmc_mean_avg_mre)
            all_results[target_col]['MCMC_MICE_V2_mre'].append(mcmc_placeholder_avg_mre)

            all_results[target_col]['MICE_nrmse'].append(mice_avg_nrmse)
            all_results[target_col]['MCMC_MICE_V1_nrmse'].append(mcmc_mean_avg_nrmse)
            all_results[target_col]['MCMC_MICE_V2_nrmse'].append(mcmc_placeholder_avg_nrmse)
            
            # Print results with timing comparison
            print(f"    Run {run+1} Results for {target_col}:")
            print(f"      MICE              - RMSE: {mice_avg_rmse:.4f}, MAE: {mice_avg_mae:.4f}, MRE: {mice_avg_mre:.4f}, NRMSE: {mice_avg_nrmse:.4f}, Time: {mice_total_time:.2f}s")
            print(f"      MCMC_MICE_V1         - RMSE: {mcmc_mean_avg_rmse:.4f}, MAE: {mcmc_mean_avg_mae:.4f}, MRE: {mcmc_mean_avg_mre:.4f}, NRMSE: {mcmc_mean_avg_nrmse:.4f}, Time: {mcmc_mean_total_time:.2f}s")
            print(f"      MCMC_MICE_V2  - RMSE: {mcmc_placeholder_avg_rmse:.4f}, MAE: {mcmc_placeholder_avg_mae:.4f}, MRE: {mcmc_placeholder_avg_mre:.4f}, NRMSE: {mcmc_placeholder_avg_nrmse:.4f}, Time: {mcmc_placeholder_total_time:.2f}s")
                        
            # Calculate speed ratios
            if mice_total_time > 0:
                mean_speed_ratio = mcmc_mean_total_time / mice_total_time
                placeholder_speed_ratio = mcmc_placeholder_total_time / mice_total_time
                print(f"      Speed vs MICE     - MCMC_MICE_V1: {mean_speed_ratio:.1f}x slower, MCMC_MICE_V2: {placeholder_speed_ratio:.1f}x slower")
            
            # Store data for visualization
            run_imputed_datasets[target_col] = col_imputed_datasets
            run_predictions[target_col] = col_predictions
            
            # Create visualizations for selected runs (same as original)
            if (run + 1) in visualize_runs:
                print(f"    Creating visualizations for Run {run+1}...")
                
                try:
                    fname1 = f'{output_dir}/imputation_comparison_run{run+1}_{target_col}.png' if save_plots else None
                    visualizer.plot_imputed_datasets_comparison(
                        complete_data=complete_data,
                        missing_data=missing_data,
                        imputed_datasets_dict=col_imputed_datasets,
                        target_col=target_col,
                        missing_indices=missing_indices.tolist(),
                        dataset_name=f'Run {run+1}',
                        fname=fname1
                    )
                    
                    fname2 = f'{output_dir}/prediction_accuracy_run{run+1}_{target_col}.png' if save_plots else None
                    visualizer.plot_prediction_accuracy_comparison(
                        true_values=true_values.values,
                        predictions_dict=col_predictions,
                        target_col=target_col,
                        dataset_name=f'Run {run+1}',
                        fname=fname2
                    )
                    
                    print(f"    ✅ Visualizations created for {target_col}")
                    
                except Exception as e:
                    print(f"    ⚠️  Visualization failed for {target_col}: {str(e)}")
        
        # Calculate run timing
        run_total_time = time.time() - run_start_time
        timing_results['per_run_times'].append(run_total_time)
        
        # Store method times for this run
        timing_results['method_times']['MICE']['per_run'].append(run_method_times['MICE'])
        timing_results['method_times']['MCMC_MICE_V1']['per_run'].append(run_method_times['MCMC_MICE_V1'])
        timing_results['method_times']['MCMC_MICE_V2']['per_run'].append(run_method_times['MCMC_MICE_V2'])
        
        print(f"\n⏱️  Run {run+1} Timing Summary:")
        print(f"   Total run time: {run_total_time:.2f}s ({run_total_time/60:.1f} minutes)")
        print(f"   MICE total: {run_method_times['MICE']:.2f}s ({run_method_times['MICE']/run_total_time*100:.1f}%)")
        print(f"   MCMC_MICE_V1 total: {run_method_times['MCMC_MICE_V1']:.2f}s ({run_method_times['MCMC_MICE_V1']/run_total_time*100:.1f}%)")
        print(f"   MCMC_MICE_V2 total: {run_method_times['MCMC_MICE_V2']:.2f}s ({run_method_times['MCMC_MICE_V2']/run_total_time*100:.1f}%)")
        print(f"   Setup/cleanup: {run_total_time - sum(run_method_times.values()):.2f}s")
        
        # 🔧 Cleanup after each run to prevent memory accumulation
        cleanup_start = time.time()
        print(f"🧹 Cleaning up after run {run+1}...")
        del mcmc_mice_mean
        del mcmc_mice_placeholder
        gc.collect()
        cleanup_time = time.time() - cleanup_start
        timing_results['cleanup_times'].append(cleanup_time)
    
    # Calculate total experiment time
    experiment_total_time = time.time() - experiment_start_time
    timing_results['total_experiment_time'] = experiment_total_time
    
    # Update method totals
    for method in timing_results['method_times']:
        timing_results['method_times'][method]['total'] = sum(timing_results['method_times'][method]['per_run'])
    
    # [Original final analysis code remains the same...]
    print(f"\n{'='*100}")
    print("FINAL RESULTS ACROSS ALL 30 RUNS")
    print('='*100)
    
    summary_results = {}
    
    for col in cols_with_missing:
        mice_rmse_array = np.array(all_results[col]['MICE_rmse'])
        mcmc_mean_rmse_array = np.array([x for x in all_results[col]['MCMC_MICE_V1_rmse'] if x != np.inf])
        mcmc_placeholder_rmse_array = np.array([x for x in all_results[col]['MCMC_MICE_V2_rmse'] if x != np.inf])
        
        mice_mae_array = np.array(all_results[col]['MICE_mae'])
        mcmc_mean_mae_array = np.array([x for x in all_results[col]['MCMC_MICE_V1_mae'] if x != np.inf])
        mcmc_placeholder_mae_array = np.array([x for x in all_results[col]['MCMC_MICE_V2_mae'] if x != np.inf])
        
        mice_mre_array = np.array(all_results[col]['MICE_mre'])
        mcmc_mean_mre_array = np.array([x for x in all_results[col]['MCMC_MICE_V1_mre'] if x != np.inf])
        mcmc_placeholder_mre_array = np.array([x for x in all_results[col]['MCMC_MICE_V2_mre'] if x != np.inf])

        mice_nrmse_array = np.array(all_results[col]['MICE_nrmse'])
        mcmc_mean_nrmse_array = np.array([x for x in all_results[col]['MCMC_MICE_V1_nrmse'] if x != np.inf])
        mcmc_placeholder_nrmse_array = np.array([x for x in all_results[col]['MCMC_MICE_V2_nrmse'] if x != np.inf])
        
        success_rate_mean = len(mcmc_mean_rmse_array) / n_runs
        success_rate_placeholder = len(mcmc_placeholder_rmse_array) / n_runs
        
        summary_results[col] = {
            'MICE_RMSE_mean': np.mean(mice_rmse_array),
            'MICE_RMSE_std': np.std(mice_rmse_array),
            'MCMC_MICE_V1_RMSE_mean': np.mean(mcmc_mean_rmse_array) if len(mcmc_mean_rmse_array) > 0 else np.inf,
            'MCMC_MICE_V1_RMSE_std': np.std(mcmc_mean_rmse_array) if len(mcmc_mean_rmse_array) > 0 else 0,
            'MCMC_MICE_V2_RMSE_mean': np.mean(mcmc_placeholder_rmse_array) if len(mcmc_placeholder_rmse_array) > 0 else np.inf,
            'MCMC_MICE_V2_RMSE_std': np.std(mcmc_placeholder_rmse_array) if len(mcmc_placeholder_rmse_array) > 0 else 0,
            'MICE_MAE_mean': np.mean(mice_mae_array),
            'MICE_MAE_std': np.std(mice_mae_array),
            'MCMC_MICE_V1_MAE_mean': np.mean(mcmc_mean_mae_array) if len(mcmc_mean_mae_array) > 0 else np.inf,
            'MCMC_MICE_V1_MAE_std': np.std(mcmc_mean_mae_array) if len(mcmc_mean_mae_array) > 0 else 0,
            'MCMC_MICE_V2_MAE_mean': np.mean(mcmc_placeholder_mae_array) if len(mcmc_placeholder_mae_array) > 0 else np.inf,
            'MCMC_MICE_V2_MAE_std': np.std(mcmc_placeholder_mae_array) if len(mcmc_placeholder_mae_array) > 0 else 0,
            'MICE_MRE_mean': np.mean(mice_mre_array),
            'MICE_MRE_std': np.std(mice_mre_array),
            'MCMC_MICE_V1_MRE_mean': np.mean(mcmc_mean_mre_array) if len(mcmc_mean_mre_array) > 0 else np.inf,
            'MCMC_MICE_V1_MRE_std': np.std(mcmc_mean_mre_array) if len(mcmc_mean_mre_array) > 0 else 0,
            'MCMC_MICE_V2_MRE_mean': np.mean(mcmc_placeholder_mre_array) if len(mcmc_placeholder_mre_array) > 0 else np.inf,
            'MCMC_MICE_V2_MRE_std': np.std(mcmc_placeholder_mre_array) if len(mcmc_placeholder_mre_array) > 0 else 0,
            'MICE_NRMSE_mean': np.mean(mice_nrmse_array),
            'MICE_NRMSE_std': np.std(mice_nrmse_array),
            'MCMC_MICE_V1_NRMSE_mean': np.mean(mcmc_mean_nrmse_array) if len(mcmc_mean_nrmse_array) > 0 else np.inf,
            'MCMC_MICE_V1_NRMSE_std': np.std(mcmc_mean_nrmse_array) if len(mcmc_mean_nrmse_array) > 0 else 0,
            'MCMC_MICE_V2_NRMSE_mean': np.mean(mcmc_placeholder_nrmse_array) if len(mcmc_placeholder_nrmse_array) > 0 else np.inf,
            'MCMC_MICE_V2_NRMSE_std': np.std(mcmc_placeholder_nrmse_array) if len(mcmc_placeholder_nrmse_array) > 0 else 0,
            'MCMC_MICE_V1_success_rate': success_rate_mean,
            'MCMC_MICE_V2_success_rate': success_rate_placeholder,
            'successful_runs_MCMC_MICE_V1': len(mcmc_mean_rmse_array),
            'successful_runs_MCMC_MICE_V2': len(mcmc_placeholder_rmse_array)
        }
    
    # ==========================================
    # COMPREHENSIVE TIMING ANALYSIS
    # ==========================================
    print(f"\n{'='*100}")
    print("COMPREHENSIVE TIMING ANALYSIS")
    print('='*100)
    
    print(f"📊 Overall Experiment Timing:")
    print(f"   Total experiment time: {experiment_total_time:.2f}s ({experiment_total_time/3600:.2f} hours)")
    print(f"   Average time per run: {np.mean(timing_results['per_run_times']):.2f}s")
    print(f"   Fastest run: {np.min(timing_results['per_run_times']):.2f}s")
    print(f"   Slowest run: {np.max(timing_results['per_run_times']):.2f}s")
    print(f"   Setup overhead: {np.mean(timing_results['setup_times']):.2f}s")
    print(f"   Cleanup overhead: {np.mean(timing_results['cleanup_times']):.2f}s")
    
    print(f"\n🕒 Method Performance Comparison:")
    mice_total = timing_results['method_times']['MICE']['total']
    mean_total = timing_results['method_times']['MCMC_MICE_V1']['total']
    placeholder_total = timing_results['method_times']['MCMC_MICE_V2']['total']
    
    print(f"   MICE total time: {mice_total:.2f}s ({mice_total/experiment_total_time*100:.1f}% of experiment)")
    print(f"   MCMC_MICE_V1 total time: {mean_total:.2f}s ({mean_total/experiment_total_time*100:.1f}% of experiment)")
    print(f"   MCMC_MICE_V2 total time: {placeholder_total:.2f}s ({placeholder_total/experiment_total_time*100:.1f}% of experiment)")
    
    if mice_total > 0:
        print(f"\n⚡ Speed Comparison (relative to MICE):")
        print(f"   MCMC_MICE_V1 is {mean_total/mice_total:.1f}x slower than MICE")
        print(f"   MCMC_MICE_V2 is {placeholder_total/mice_total:.1f}x slower than MICE")
    
    print(f"\n📈 Average Time per Operation:")
    total_operations_mice = n_runs * len(cols_with_missing)
    successful_operations_mean = sum(len([x for x in all_results[col]['MCMC_MICE_V1_rmse'] if x != np.inf]) for col in cols_with_missing)
    successful_operations_placeholder = sum(len([x for x in all_results[col]['MCMC_MICE_V2_rmse'] if x != np.inf]) for col in cols_with_missing)
    
    print(f"   MICE: {mice_total/total_operations_mice:.2f}s per column imputation")
    if successful_operations_mean > 0:
        print(f"   MCMC_MICE_V1: {mean_total/successful_operations_mean:.2f}s per successful column imputation")
    if successful_operations_placeholder > 0:
        print(f"   MCMC_MICE_V2: {placeholder_total/successful_operations_placeholder:.2f}s per successful column imputation")
    
    # Per-column timing analysis
    print(f"\n🎯 Per-Column Timing Analysis:")
    for col in cols_with_missing:
        if col in timing_results['method_times']['MICE']['per_column']:
            mice_col_times = timing_results['method_times']['MICE']['per_column'][col]
            mean_col_times = timing_results['method_times']['MCMC_MICE_V1']['per_column'][col]
            placeholder_col_times = timing_results['method_times']['MCMC_MICE_V2']['per_column'][col]
            
            print(f"\n   {col}:")
            print(f"     MICE: {np.mean(mice_col_times):.2f}s ± {np.std(mice_col_times):.2f}s (n={len(mice_col_times)})")
            if len(mean_col_times) > 0:
                print(f"     MCMC_MICE_V1: {np.mean(mean_col_times):.2f}s ± {np.std(mean_col_times):.2f}s (n={len(mean_col_times)})")
            else:
                print(f"     MCMC_MICE_V1: No successful runs")
            if len(placeholder_col_times) > 0:
                print(f"     MCMC_MICE_V2: {np.mean(placeholder_col_times):.2f}s ± {np.std(placeholder_col_times):.2f}s (n={len(placeholder_col_times)})")
            else:
                print(f"     MCMC_MICE_V2: No successful runs")
    
    # Performance/Cost Analysis
    print(f"\n💰 Performance vs Cost Analysis:")
    print(f"{'Method':<20} {'Avg RMSE':<12} {'Success Rate':<12} {'Avg Time (s)':<12} {'Efficiency*':<12}")
    print("-" * 80)
    
    for col in cols_with_missing:
        # MICE analysis
        mice_rmse = summary_results[col]['MICE_RMSE_mean']
        mice_time = np.mean(timing_results['method_times']['MICE']['per_column'][col])
        mice_efficiency = 1.0 / (mice_rmse * mice_time) if mice_rmse != np.inf and mice_time > 0 else 0
        
        print(f"\n{col.upper()}:")
        print(f"{'MICE':<20} {mice_rmse:<12.4f} {'100.0%':<12} {mice_time:<12.2f} {mice_efficiency:<12.6f}")
        
        # MCMC-MEAN analysis
        if summary_results[col]['MCMC_MICE_V1_RMSE_mean'] != np.inf:
            mean_rmse = summary_results[col]['MCMC_MICE_V1_RMSE_mean']
            mean_time = np.mean(timing_results['method_times']['MCMC_MICE_V1']['per_column'][col])
            mean_success = summary_results[col]['MCMC_MICE_V1_success_rate']
            mean_efficiency = 1.0 / (mean_rmse * mean_time) if mean_rmse != np.inf and mean_time > 0 else 0
            print(f"{'MCMC_MICE_V1':<20} {mean_rmse:<12.4f} {mean_success*100:<11.1f}% {mean_time:<12.2f} {mean_efficiency:<12.6f}")
        else:
            print(f"{'MCMC_MICE_V1':<20} {'FAILED':<12} {'0.0%':<12} {'N/A':<12} {'0.000000':<12}")
        
        # MCMC-PLACEHOLDER analysis
        if summary_results[col]['MCMC_MICE_V2_RMSE_mean'] != np.inf:
            placeholder_rmse = summary_results[col]['MCMC_MICE_V2_RMSE_mean']
            placeholder_time = np.mean(timing_results['method_times']['MCMC_MICE_V2']['per_column'][col])
            placeholder_success = summary_results[col]['MCMC_MICE_V2_success_rate']
            placeholder_efficiency = 1.0 / (placeholder_rmse * placeholder_time) if placeholder_rmse != np.inf and placeholder_time > 0 else 0
            print(f"{'MCMC_MICE_V2':<20} {placeholder_rmse:<12.4f} {placeholder_success*100:<11.1f}% {placeholder_time:<12.2f} {placeholder_efficiency:<12.6f}")
        else:
            print(f"{'MCMC_MICE_V2':<20} {'FAILED':<12} {'0.0%':<12} {'N/A':<12} {'0.000000':<12}")

    for col in cols_with_missing:
        if col in summary_results:
            print(f"\n{col.upper()}:")
            print(f"  MICE Runs: {total_operations_mice//len(cols_with_missing)}/{n_runs} (100.0%)")
            print(f"  MCMC_MICE_V1 Runs: {summary_results[col]['successful_runs_MCMC_MICE_V1']}/{n_runs} ({summary_results[col]['MCMC_MICE_V1_success_rate']:.1%})")
            print(f"  MCMC_MICE_V2 Runs: {summary_results[col]['successful_runs_MCMC_MICE_V2']}/{n_runs} ({summary_results[col]['MCMC_MICE_V2_success_rate']:.1%})")
            
            mice_rmse_array = np.array(all_results[col]['MICE_rmse'])
            mcmc_mean_rmse_array = np.array([x for x in all_results[col]['MCMC_MICE_V1_rmse'] if x != np.inf])
            mcmc_placeholder_rmse_array = np.array([x for x in all_results[col]['MCMC_MICE_V2_rmse'] if x != np.inf])
            
            print(f"\n  Average Metrics over {n_runs} Runs (mean ± std):")

            # MICE
            print(f"    MICE:")
            print(f"      RMSE    : {np.mean(all_results[col]['MICE_rmse']):.4f} ± {np.std(all_results[col]['MICE_rmse']):.4f}")
            print(f"      MAE     : {np.mean(all_results[col]['MICE_mae']):.4f} ± {np.std(all_results[col]['MICE_mae']):.4f}")
            print(f"      MRE     : {np.mean(all_results[col]['MICE_mre']):.4f} ± {np.std(all_results[col]['MICE_mre']):.4f}")
            print(f"      NRMSE   : {np.mean(all_results[col]['MICE_nrmse']):.4f} ± {np.std(all_results[col]['MICE_nrmse']):.4f}")

            # MCMC-MEAN
            if len(mcmc_mean_rmse_array) > 0:
                print(f"    MCMC_MICE_V1:")
                print(f"      RMSE    : {np.mean(mcmc_mean_rmse_array):.4f} ± {np.std(mcmc_mean_rmse_array):.4f}")
                print(f"      MAE     : {np.mean(all_results[col]['MCMC_MICE_V1_mae']):.4f} ± {np.std(all_results[col]['MCMC_MICE_V1_mae']):.4f}")
                print(f"      MRE     : {np.mean(all_results[col]['MCMC_MICE_V1_mre']):.4f} ± {np.std(all_results[col]['MCMC_MICE_V1_mre']):.4f}")
                print(f"      NRMSE   : {np.mean(all_results[col]['MCMC_MICE_V1_nrmse']):.4f} ± {np.std(all_results[col]['MCMC_MICE_V1_nrmse']):.4f}")
            else:
                print(f"   MCMC_MICE_V1: FAILED in all runs")

            # MCMC-PLACEHOLDER
            if len(mcmc_placeholder_rmse_array) > 0:
                print(f"    MCMC_MICE_V2:")
                print(f"      RMSE    : {np.mean(mcmc_placeholder_rmse_array):.4f} ± {np.std(mcmc_placeholder_rmse_array):.4f}")
                print(f"      MAE     : {np.mean(all_results[col]['MCMC_MICE_V2_mae']):.4f} ± {np.std(all_results[col]['MCMC_MICE_V2_mae']):.4f}")
                print(f"      MRE     : {np.mean(all_results[col]['MCMC_MICE_V2_mre']):.4f} ± {np.std(all_results[col]['MCMC_MICE_V2_mre']):.4f}")
                print(f"      NRMSE   : {np.mean(all_results[col]['MCMC_MICE_V2_nrmse']):.4f} ± {np.std(all_results[col]['MCMC_MICE_V2_nrmse']):.4f}")
            else:
                print(f"    MCMC_MICE_V2: FAILED in all runs")
    
    # Create summary visualizations (same as original)
    try:
        for metric in ['RMSE', 'MAE', 'MRE', 'NRMSE']:
            fname = f'{output_dir}/experiment_summary_{metric}.png' if save_plots else None
            visualizer.plot_experiment_summary(
                summary_results=summary_results,
                all_results=all_results,
                metric=metric,
                fname=fname
            )
        
        print(f"  📊 All summary visualizations completed!")
        if save_plots:
            print(f"  💾 Plots saved to: {output_dir}")
        
    except Exception as e:
        print(f"  ⚠️  Summary visualization failed: {str(e)}")
    
    # Final comparison and winner determination (same as original)
    print(f"\n{'='*100}")
    print("OVERALL EXPERIMENT SUMMARY")
    print('='*100)
    
    mice_wins = 0
    mcmc_mean_wins = 0
    mcmc_placeholder_wins = 0
    ties = 0
    
    print(f"{'Column':<15} {'MICE RMSE':<12} {'MCMC_MICE_V1 RMSE':<12} {'MCMC_MICE_V2 RMSE':<12} {'Winner':<15}")
    print("-" * 80)
    
    for col in cols_with_missing:
        if col in summary_results:
            mice_rmse = summary_results[col]['MICE_RMSE_mean']
            mean_rmse = summary_results[col]['MCMC_MICE_V1_RMSE_mean']
            placeholder_rmse = summary_results[col]['MCMC_MICE_V2_RMSE_mean']
            
            valid_rmses = []
            if mice_rmse != np.inf:
                valid_rmses.append(('MICE', mice_rmse))
            if mean_rmse != np.inf:
                valid_rmses.append(('MCMC_MICE_V1', mean_rmse))
            if placeholder_rmse != np.inf:
                valid_rmses.append(('MCMC_MICE_V2', placeholder_rmse))
            
            if valid_rmses:
                winner = min(valid_rmses, key=lambda x: x[1])[0]
                if winner == 'MICE':
                    mice_wins += 1
                elif winner == 'MCMC_MICE_V1':
                    mcmc_mean_wins += 1
                elif winner == 'MCMC_MICE_V2':
                    mcmc_placeholder_wins += 1
            else:
                winner = "ALL FAIL"
                ties += 1
            
            print(f"{col:<15} {mice_rmse:<12.6f} {mean_rmse:<12.6f} {placeholder_rmse:<12.6f} {winner:<15}")
    
    print(f"\nFINAL SCORECARD:")
    print(f"  📊 MICE wins: {mice_wins}/{len(cols_with_missing)} columns")
    print(f"  🔬 MCMC_MICE_V1 wins: {mcmc_mean_wins}/{len(cols_with_missing)} columns")
    print(f"  🎯 MCMC_MICE_V2 wins: {mcmc_placeholder_wins}/{len(cols_with_missing)} columns") 
    print(f"  🤝 Ties/Fails: {ties}/{len(cols_with_missing)} columns")
    
    # Winner determination with timing consideration
    print(f"\n🏆 PERFORMANCE WINNER:")
    if mcmc_placeholder_wins > max(mice_wins, mcmc_mean_wins):
        print(f"   MCMC_MICE_V2 WINS! (Best accuracy)")
        print(f"   Time-series aware initialization is the best approach!")
    elif mcmc_mean_wins > max(mice_wins, mcmc_placeholder_wins):
        print(f"   MCMC_MICE_V1 WINS! (Best accuracy)")
        print(f"   Simple mean initialization with MCMC is the best!")
    elif mice_wins > max(mcmc_mean_wins, mcmc_placeholder_wins):
        print(f"   MICE WINS! (Best accuracy)")
        print(f"   Standard MICE outperforms both MCMC variants!")
    else:
        print(f"   CLOSE COMPETITION!")
    
    print(f"\n⚡ EFFICIENCY WINNER (considering time):")
    print(f"   MICE: Always fastest ({mice_total:.1f}s total), 100% success rate")
    if mean_total > 0 and mice_total > 0:
        print(f"   MCMC_MICE_V1: {mean_total/mice_total:.1f}x slower, {np.mean([summary_results[col]['MCMC_MICE_V1_success_rate'] for col in cols_with_missing])*100:.1f}% avg success rate")
    if placeholder_total > 0 and mice_total > 0:
        print(f"   MCMC_MICE_V2: {placeholder_total/mice_total:.1f}x slower, {np.mean([summary_results[col]['MCMC_MICE_V2_success_rate'] for col in cols_with_missing])*100:.1f}% avg success rate")
    
    # Return results with timing information
    return summary_results, all_results, timing_results