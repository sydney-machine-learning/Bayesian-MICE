from Comparison_runs import *

def enhanced_run_experiment(complete_data, data_with_time, missing_data, 
                          time_col='Time', n_runs=3, n_imputations=5, max_iter=5,
                          visualize_runs=[1, 15, 30], save_plots=True, output_dir='./plots'):
    """
    Enhanced main function with integrated visualizations
    
    Parameters:
    -----------
    complete_data : pd.DataFrame
        Original complete dataset
    data_with_time : pd.DataFrame
        Dataset with time information
    missing_data : pd.DataFrame
        Dataset with missing values
    time_col : str
        Name of time column
    n_runs : int
        Number of experimental runs
    n_imputations : int
        Number of imputations per run
    max_iter : int
        Maximum MICE iterations
    visualize_runs : list
        Which runs to create detailed visualizations for
    save_plots : bool
        Whether to save plots to files
    output_dir : str
        Directory to save plots
    """
    
    print("="*100)
    print("ENHANCED 30 RUNS × 5 IMPUTATIONS EXPERIMENT WITH VISUALIZATION")
    print("Comparing: MICE vs MCMC_MICE_V1 vs MCMC_MICE_V2")
    print("="*100)
    print(f"Configuration:")
    print(f"  • Experimental runs: {n_runs}")
    print(f"  • Imputations per run: {n_imputations}")
    print(f"  • MICE iterations: {max_iter}")
    #print(f"  • MCMC samples: 12,500")
    #print(f"  • MCMC burn-in: 4,800")
    print(f"  • Time column: {time_col}")
    print(f"  • Visualization runs: {visualize_runs}")
    print(f"  • Save plots: {save_plots}")
    if save_plots:
        print(f"  • Output directory: {output_dir}")
    print("="*100)
    
    # Create output directory
    if save_plots:
        import os
        os.makedirs(output_dir, exist_ok=True)
        print(f"📁 Created output directory: {output_dir}")
    
    # Run the enhanced experiment with visualization
    summary_results, all_results, timing_results = enhanced_comparison_with_runs(
        complete_data=complete_data,
        missing_data=missing_data,
        data_with_time=data_with_time,
        time_col=time_col,
        n_runs=n_runs,
        n_imputations=n_imputations,
        max_iter=max_iter,
        visualize_runs=visualize_runs,
        save_plots=save_plots,
        output_dir=output_dir
    )
    
    print(f"\n{'='*100}")
    print("🎉 ENHANCED EXPERIMENT COMPLETED!")
    print('='*100)
    print(f"✅ Completed {n_runs} runs with {n_imputations} imputations each")
    print(f"📊 Generated visualizations for runs: {visualize_runs}")
    if save_plots:
        print(f"💾 All plots saved to: {output_dir}")
        print(f"📈 Summary plots: experiment_summary_[METRIC].png")
        print(f"🔍 Detailed plots: imputation_comparison_run[X]_[COLUMN].png")
        print(f"📋 Accuracy plots: prediction_accuracy_run[X]_[COLUMN].png")
    
    return summary_results, all_results


if __name__ == "__main__":
    # Enhanced example usage with visualization
    
    # Load your data
    #data_with_time = pd.read_csv('physionet_5000patients.csv')
    #data_with_missing = pd.read_csv('physio_with_missing.csv')
    #data_subset = pd.read_csv('physio_subdata.csv')
    data = pd.read_csv("AirQualityUCI.csv", sep=';', decimal=',', parse_dates=['Date'])

    # Fix the time format (replace '.' with ':' so time is valid)
    data['Time'] = data['Time'].str.replace('.', ':', regex=False)

    # Combine date and time into one datetime column
    data['Date_Time'] = pd.to_datetime(data['Date'].astype(str) + ' ' + data['Time'], format='%d/%m/%Y %H:%M:%S')

    # Drop original Date and Time columns
    data.drop(columns=['Date', 'Time'], inplace=True)

    # Select relevant features
    selected_features = ['Date_Time', 'CO(GT)', 'PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)', 'PT08.S2(NMHC)', 'T']
    data = data[selected_features]

    # Replace true missing values (-200) with NaN
    data_with_time = data.replace(-200, np.nan)
    data_subset = pd.read_csv('Data_subset_AirQuality.csv')
    data_with_missing = pd.read_csv('Data_with_missing_AirQuality.csv')
    data_with_missing['Date_Time'] = pd.to_datetime(data_with_missing['Date_Time'])
    data_with_time['Date_Time'] = pd.to_datetime(data_with_time['Date_Time'])
    data_subset['Date_Time'] = pd.to_datetime(data_subset['Date_Time'])
    # Run the enhanced experiment with visualization
    summary_results, all_results = enhanced_run_experiment(
        complete_data=data_subset,
        data_with_time=data_with_time,
        missing_data=data_with_missing,
        time_col='Date_Time',
        n_runs=1,
        n_imputations=5,
        max_iter=5,
        visualize_runs=[1], # Create detailed plots for these runs
        save_plots=True,
        output_dir='./experiment_plots'
    )
    
    print("\n🎯 Enhanced 30×5 experiment with visualization finished!")
    print("📊 Check the experiment_plots directory for all visualizations!")