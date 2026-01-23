from packages import *


class MCMCMICEVisualizer:
    """Visualization class for MCMC-MICE experiment results"""
    
    def __init__(self, time_col='Time'):
        self.time_col = time_col
        sns.set_context("talk")
        sns.set_style("ticks", {'axes.grid': True})
    
    def plot_imputed_datasets_comparison(self, complete_data, missing_data, imputed_datasets_dict, 
                                       target_col, time_window=None, missing_indices=None, 
                                       dataset_name=None, fname=None):
        """
        Plot comparison of original, missing, and imputed datasets
        
        Parameters:
        -----------
        complete_data : pd.DataFrame
        
            Original complete dataset
        missing_data : pd.DataFrame
            Dataset with missing values
        imputed_datasets_dict : dict
            Dictionary with method names as keys and imputed datasets as values
            Example: {'MICE': mice_imputed_df, 'MCMC-MEAN': mcmc_mean_df, 'MCMC-PLACEHOLDER': mcmc_placeholder_df}
        target_col : str
            Column to visualize
        time_window : tuple, optional
            (start_idx, end_idx) to zoom into specific time period
        missing_indices : list, optional
            Indices of missing values to highlight
        dataset_name : str, optional
            Name for the plot title
        fname : str, optional
            Filename to save the plot
        """
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        #fig.suptitle(f'Imputation Comparison: {target_col}' + 
                   # (f' - {dataset_name}' if dataset_name else ''), fontsize=16, fontweight='bold')
        
        # Determine time window
        if time_window is None:
            start_idx, end_idx = 0, len(complete_data)
        else:
            start_idx, end_idx = time_window
        
        # Create time axis
        if self.time_col in complete_data.columns:
            x_axis = np.arange(start_idx, end_idx)
            x_label = 'Time Index'
        else:
            x_axis = np.arange(start_idx, end_idx)
            x_label = 'Index'
        
        # Slice data for window
        complete_slice = complete_data[target_col].iloc[start_idx:end_idx]
        missing_slice = missing_data[target_col].iloc[start_idx:end_idx]
        
        # Plot 1: Original Complete Data
        ax1 = axes[0, 0]
        ax1.plot(x_axis, complete_slice, 'b-', linewidth=1.5, alpha=0.8, label='Complete Data')
        #ax1.set_title('Original Complete Data', fontweight='bold')
        ax1.set_xlabel(x_label, fontsize=16)
        ax1.set_ylabel(target_col, fontsize=16)
        ax1.tick_params(axis='both', labelsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend(
            fontsize=10, frameon=False,
            handlelength=2.0, handleheight=1.0, markerscale=1.0,
            loc='upper right'
        )
        # Plot 2: Data with Missing Values
        ax2 = axes[0, 1]
        # Plot observed values
        observed_mask = ~missing_slice.isnull()
        if observed_mask.any():
            ax2.plot(x_axis[observed_mask], missing_slice[observed_mask], 
                    'g.', markersize=4, alpha=0.7, label='Observed')
        
        # Highlight missing regions
        if missing_indices is not None:
            # Filter missing indices to current window
            window_missing = [idx for idx in missing_indices 
                            if start_idx <= idx < end_idx]
            if window_missing:
                missing_x = [idx for idx in window_missing]
                missing_y = [complete_slice.iloc[idx - start_idx] for idx in window_missing]
                ax2.plot(missing_x, missing_y, 'r.', markersize=6, alpha=0.8, label='Missing Values')
        
        #ax2.set_title('Data with Missing Values', fontweight='bold')
        ax2.set_xlabel(x_label, fontsize=16)
        ax2.set_ylabel(target_col, fontsize=16)
        ax2.tick_params(axis='both', labelsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend(
            fontsize=10, frameon=False,
            handlelength=2.0, handleheight=1.0, markerscale=1.0,
            loc='upper right'
        )
        
        # Plot 3 & 4: Imputed datasets comparison
        colors = ['C1', 'C2', 'C3', 'C4', 'C5']
        
        if len(imputed_datasets_dict) == 1:
            # Single method comparison
            method_name, imputed_data = list(imputed_datasets_dict.items())[0]
            ax3 = axes[1, 0]
            
            # Plot original and imputed
            ax3.plot(x_axis, complete_slice, 'b-', linewidth=2, alpha=0.7, label='Original')
            imputed_slice = imputed_data[target_col].iloc[start_idx:end_idx]
            ax3.plot(x_axis, imputed_slice, colors[0], linewidth=2, alpha=0.8, 
                    label=f'{method_name} Imputed', linestyle='--')
            
            # Highlight imputed regions
            if missing_indices is not None:
                window_missing = [idx for idx in missing_indices 
                                if start_idx <= idx < end_idx]
                if window_missing:
                    missing_x = [idx for idx in window_missing]
                    imputed_y = [imputed_slice.iloc[idx - start_idx] for idx in window_missing]
                    ax3.scatter(missing_x, imputed_y, c=colors[0], s=30, alpha=0.9, 
                              label='Imputed Points', zorder=5)
            
            #ax3.set_title(f'{method_name} vs Original', fontweight='bold')
            ax3.set_xlabel(x_label, fontsize=16)
            ax3.set_ylabel(target_col, fontsize=16)
            ax3.tick_params(axis='both', labelsize=14)
            ax3.grid(True, alpha=0.3)
            ax3.legend(
                fontsize=10, frameon=False,
                handlelength=2.0, handleheight=1.0, markerscale=1.0,
                loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=min(2, len(method_name))
            )
            
            # Hide the fourth subplot
            axes[1, 1].set_visible(False)
            
        else:
            # Multiple methods comparison
            ax3 = axes[1, 0]
            ax4 = axes[1, 1]
            
            # Plot 3: All methods together
            ax3.plot(x_axis, complete_slice, 'k-', linewidth=2.5, alpha=0.95, label='Original')
            
            for i, (method_name, imputed_data) in enumerate(imputed_datasets_dict.items()):
                imputed_slice = imputed_data[target_col].iloc[start_idx:end_idx]
                ax3.plot(x_axis, imputed_slice, colors[i % len(colors)], 
                        linewidth=1.8, alpha=0.65, label=f'{method_name}', linestyle='--')
            
            #ax3.set_title('All Methods Comparison', fontweight='bold')
            ax3.set_xlabel(x_label, fontsize=16)
            ax3.set_ylabel(target_col, fontsize=16)
            ax3.tick_params(axis='both', labelsize=14)
            ax3.grid(True, alpha=0.3)
            ax3.legend(
                fontsize=10, frameon=False,
                handlelength=2.0, handleheight=1.0, markerscale=1.0,
                loc='upper center', bbox_to_anchor=(0.5, -0.18),
                ncol=min(2, len(method_name))
            )
            # Plot 4: Focus on missing regions only
            if missing_indices is not None:
                window_missing = [idx for idx in missing_indices 
                                if start_idx <= idx < end_idx]
                if window_missing:
                    # Create focused view around missing values
                    focus_window = 50  # Show ±50 points around missing values
                    focus_start = max(0, min(window_missing) - focus_window)
                    focus_end = min(len(complete_data), max(window_missing) + focus_window)
                    focus_x = np.arange(focus_start, focus_end)
                    
                    # Plot original in focus window
                    focus_complete = complete_data[target_col].iloc[focus_start:focus_end]
                    ax4.plot(focus_x, focus_complete, 'k-', linewidth=2.5, alpha=0.95, label='Original')
                    
                    # Plot each method's imputation
                    for i, (method_name, imputed_data) in enumerate(imputed_datasets_dict.items()):
                        focus_imputed = imputed_data[target_col].iloc[focus_start:focus_end]
                        ax4.plot(focus_x, focus_imputed, colors[i % len(colors)], 
                                linewidth=1.8, alpha=0.7, label=f'{method_name}', linestyle='--')
                    
                    # Highlight the actual missing points
                    focus_missing = [idx for idx in window_missing 
                                   if focus_start <= idx < focus_end]
                    if focus_missing:
                        original_missing_y = [complete_data[target_col].iloc[idx] for idx in focus_missing]
                        ax4.scatter(focus_missing, original_missing_y, c='red', s=50, alpha=0.9, 
                                  label='True Missing', zorder=5, marker='o')
                        
                        # Add vertical lines to highlight missing regions
                        for miss_idx in focus_missing:
                            ax4.axvline(x=miss_idx, color='red', alpha=0.3, linestyle=':', linewidth=1)
                    
                    #ax4.set_title('Focus on Missing Regions', fontweight='bold')
                    ax4.set_xlabel(x_label, fontsize=16)
                    ax4.set_ylabel(target_col, fontsize=16)
                    ax4.tick_params(axis='both', labelsize=14)
                    ax4.grid(True, alpha=0.3)
                    ax4.legend(
                        fontsize=10, frameon=False,
                        handlelength=2.0, handleheight=1.0, markerscale=1.0,
                        loc='upper center', bbox_to_anchor=(0.5, -0.18),
                        ncol=min(3, len(method_name))
                    )
                else:
                    ax4.text(0.5, 0.5, 'No missing values\nin time window', 
                            ha='center', va='center', transform=ax4.transAxes, fontsize=14)
                    #ax4.set_title('Focus on Missing Regions', fontweight='bold')
            else:
                ax4.text(0.5, 0.5, 'Missing indices\nnot provided', 
                        ha='center', va='center', transform=ax4.transAxes, fontsize=14)
                #ax4.set_title('Focus on Missing Regions', fontweight='bold')
        
        fig.tight_layout()
        fig.subplots_adjust(hspace=0.40, bottom=0.16) 
        if fname is not None:
            plt.savefig(fname, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_prediction_accuracy_comparison(self, true_values, predictions_dict, 
                                          method_names=None, target_col=None, 
                                          dataset_name=None, fname=None):
        """
        Plot prediction accuracy comparison across methods
        
        Parameters:
        -----------
        true_values : array-like
            True values for missing data points
        predictions_dict : dict
            Dictionary with method names as keys and prediction arrays as values
        method_names : list, optional
            Order of methods to display
        target_col : str, optional
            Name of target column
        dataset_name : str, optional
            Name of dataset
        fname : str, optional
            Filename to save the plot
        """
        
        if method_names is None:
            method_names = list(predictions_dict.keys())
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        title = f'Prediction Accuracy Comparison'
        if target_col:
            title += f': {target_col}'
        if dataset_name:
            title += f' - {dataset_name}'
        #fig.suptitle(title, fontsize=16, fontweight='bold')
        legend_kwargs = dict(
            loc='upper center', bbox_to_anchor=(0.5, -0.18),
            ncol=min(3, len(method_names)),
            fontsize=10, frameon=False,
            handlelength=2.0, handleheight=1.0, markerscale=1.0
        )
        
        colors = ['C1', 'C2', 'C3', 'C4', 'C5']
        
        # Plot 1: True vs Predicted scatter plots
        ax1 = axes[0, 0]
        for i, method in enumerate(method_names):
            if method in predictions_dict:
                predictions = predictions_dict[method]
                ax1.scatter(true_values, predictions, alpha=0.6, s=18, 
                           color=colors[i % len(colors)], label=method)
        
        # Add perfect prediction line
        lo = np.min([true_values.min()] + [predictions_dict[m].min() for m in method_names if m in predictions_dict])
        hi = np.max([true_values.max()] + [predictions_dict[m].max() for m in method_names if m in predictions_dict])
        pad = 0.03 * (hi - lo)
        min_val = lo - pad
        max_val = hi + pad
        ax1.plot([min_val, max_val], [min_val, max_val], 'k--', lw =1.2, alpha=0.8, label='_nolegend_')
        ax1.set_xlim(min_val, max_val), ax1.set_ylim(min_val, max_val)
        ax1.set_aspect('equal', 'box')  # maintain aspect of 1:1
        ax1.set_xlabel('True Values', fontsize=16)
        ax1.set_ylabel('Predicted Values', fontsize=16)
        ax1.tick_params(axis='both', labelsize=14)
        #ax1.set_title('True vs Predicted Values')
        ax1.grid(True, alpha=0.3)
        ax1.legend(**legend_kwargs)
        
        # Plot 2: Residuals
        ax2 = axes[0, 1]
        all_resid = []
        for i, method in enumerate(method_names):
            if method in predictions_dict:
                predictions = predictions_dict[method]
                residuals = predictions - true_values
                all_resid.append(residuals)
                ax2.scatter(true_values, residuals, alpha=0.6, s=18, 
                           color=colors[i % len(colors)], label=method)
        
        ax2.axhline(y=0, color='k', linestyle='--', lw=1.2, alpha=0.8)
        if all_resid:
            sigma = np.std(np.concatenate(all_resid))
            ax2.axhline(+2*sigma, color='k', ls=':', lw=1.0, alpha=0.6, label ='_nolegend_')
            ax2.axhline(-2*sigma, color='k', ls=':', lw=1.0, alpha=0.6, label ='_nolegend_')

        # align x with Plot 1 and make y symmetric
        ax2.set_xlim(ax1.get_xlim())
        ymin, ymax = ax2.get_ylim()
        m = max(abs(ymin), abs(ymax))
        ax2.set_ylim(-m, m)
        ax2.set_xlabel('True Values', fontsize=16)
        ax2.set_ylabel('Residuals (Predicted - True)', fontsize=16)
        ax2.tick_params(axis='both', labelsize=14)
        #ax2.set_title('Residual Plot')
        ax2.grid(True, alpha=0.3)
        ax2.legend(**legend_kwargs)
        
        # Plot 3: Time series view of predictions
        ax3 = axes[1, 0]
        x_indices = np.arange(len(true_values))
        
        # Sort by true values for better visualization
        sort_idx = np.argsort(true_values)
        sorted_true = true_values[sort_idx]
        
        ax3.plot(x_indices, sorted_true, 'k-', linewidth=2.0, alpha=0.9, label='True Values')
        
        for i, method in enumerate(method_names):
            if method in predictions_dict:
                sorted_pred = predictions_dict[method][sort_idx]
                ax3.plot(x_indices, sorted_pred, colors[i % len(colors)], 
                        linewidth=1.8, alpha=0.65, label=method, linestyle='--')
        
        ax3.set_xlabel('Sorted Index', fontsize=16)
        ax3.set_ylabel('Values', fontsize=16)
        ax3.tick_params(axis='both', labelsize=14)
        #ax3.set_title('Sorted True vs Predicted Values')
        ax3.grid(True, alpha=0.3)
        ax3.legend(**legend_kwargs)
        
        # Plot 4: Error distributions
        ax4 = axes[1, 1]
        error_data = []
        error_labels = []
        
        for method in method_names:
            if method in predictions_dict:
                predictions = predictions_dict[method]
                errors = np.abs(predictions - true_values)
                error_data.append(errors)
                error_labels.append(method)
        
        if error_data:
            box_plot = ax4.boxplot(error_data, labels=error_labels, patch_artist=True, showfliers=True)
            
            # Color the boxes
            for patch, color in zip(box_plot['boxes'], colors[:len(error_data)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.55)
            for med in box_plot['medians']:
                med.set(color='k', linewidth=1.5)
            for w in box_plot['whiskers'] + box_plot['caps']:
                w.set(color='0.3', linewidth=1.5)
            for f1 in box_plot.get('fliers', []):
                f1.set(marker='o', markersize=3, alpha=0.6, markeredgecolor='0.3')
        
        ax4.set_ylabel('Absolute Error', fontsize=16)
        ax4.tick_params(axis='both', labelsize=14)
        #ax4.set_title('Error Distribution Comparison')
        ax4.grid(True, alpha=0.3)
        
        fig.tight_layout()
        fig.subplots_adjust(hspace=0.40, bottom=0.16)  # space for below-legends

        
        if fname is not None:
            plt.savefig(fname, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_experiment_summary(self, summary_results, all_results, metric='NRMSE', fname=None):
        """
        Plot summary of 30-run experiment results
        
        Parameters:
        -----------
        summary_results : dict
            Summary statistics from the experiment
        all_results : dict
            All individual run results
        metric : str
            Which metric to visualize ('RMSE', 'MAE', 'MRE')
        fname : str, optional
            Filename to save the plot
        """
        
        columns = list(summary_results.keys())
        methods = ['MICE', 'MCMC_MICE_V1', 'MCMC_MICE_V2']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'30-Run Experiment Summary: {metric} Comparison', fontsize=16, fontweight='bold')
        
        # Plot 1: Mean performance with error bars
        ax1 = axes[0, 0]
        x_pos = np.arange(len(columns))
        width = 0.25
        legend_kwargs = dict(
            loc='upper center', bbox_to_anchor=(0.5, -0.18),
            ncol=3,
            fontsize=10, frameon=False,
            handlelength=2.0, handleheight=1.0, markerscale=1.0
        )

        for i, method in enumerate(methods):
            means = []
            stds = []
            
            for col in columns:
                if f'{method}_{metric}_mean' in summary_results[col]:
                    mean_val = summary_results[col][f'{method}_{metric}_mean']
                    std_val = summary_results[col][f'{method}_{metric}_std']
                    if mean_val != np.inf:
                        means.append(mean_val)
                        stds.append(std_val)
                    else:
                        means.append(0)  # or np.nan
                        stds.append(0)
                else:
                    means.append(0)
                    stds.append(0)
            
            bars = ax1.bar(x_pos + i*width, means, width, 
                          yerr=stds, capsize=5, alpha=0.8,
                          label=method.replace('_', '-'))
        
        ax1.set_xlabel('Variables', fontsize=16)
        ax1.set_ylabel(f'{metric}', fontsize=16)
        ax1.set_title(f'Mean {metric} Across 30 Runs')
        ax1.set_xticks(x_pos + width)
        ax1.set_xticklabels(columns, rotation=45, ha='right')
        ax1.tick_params(axis='both', labelsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend(**legend_kwargs)

        # Plot 2: Success rates
        ax2 = axes[0, 1]
        success_rates = {method: [] for method in methods}
        
        for col in columns:
            success_rates['MICE'].append(1.0)  # MICE always succeeds
            success_rates['MCMC_MICE_V1'].append(
                summary_results[col].get('MCMC_MICE_V1_success_rate', 0))
            success_rates['MCMC_MICE_V2'].append(
                summary_results[col].get('MCMC_MICE_V2_success_rate', 0))
        
        for i, method in enumerate(methods):
            ax2.bar(x_pos + i*width, success_rates[method], width, 
                   alpha=0.8, label=method.replace('_', '-'))
        
        ax2.set_xlabel('Variables', fontsize=16)
        ax2.set_ylabel('Success Rate', fontsize=16)
        #ax2.set_title('Method Success Rates')
        ax2.set_xticks(x_pos + width)
        ax2.set_xticklabels(columns, rotation=45, ha='right')
        ax2.set_ylim(0, 1.1)
        ax2.tick_params(axis='both', labelsize=14)
        ax2.legend(**legend_kwargs)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Distribution of results across runs
        ax3 = axes[1, 0]
        if len(columns) > 0:
            # Pick first column for detailed analysis
            col = columns[0]
            method_data = []
            method_labels = []
            
            for method in methods:
                if f'{method}_{metric.lower()}' in all_results[col]:
                    data = [x for x in all_results[col][f'{method}_{metric.lower()}'] if x != np.inf]
                    if data:
                        method_data.append(data)
                        method_labels.append(method.replace('_', '-'))
            
            if method_data:
                box_plot = ax3.boxplot(method_data, labels=method_labels, patch_artist=True)
                colors = ['C0', 'C1', 'C2']
                for patch, color in zip(box_plot['boxes'], colors[:len(method_data)]):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
        
        ax3.set_ylabel(f'{metric}', fontsize=16)
        #ax3.set_title(f'{metric} Distribution - {columns[0] if columns else "N/A"}')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Overall winner count
        ax4 = axes[1, 1]
        winner_counts = {'MICE': 0, 'MCMC_MICE_V1': 0, 'MCMC_MICE_V2': 0, 'Failed': 0}
        
        for col in columns:
            mice_val = summary_results[col][f'MICE_{metric}_mean']
            mean_val = summary_results[col][f'MCMC_MICE_V1_{metric}_mean']
            placeholder_val = summary_results[col][f'MCMC_MICE_V2_{metric}_mean']
            
            valid_results = []
            if mice_val != np.inf:
                valid_results.append(('MICE', mice_val))
            if mean_val != np.inf:
                valid_results.append(('MCMC-MICE_V1', mean_val))
            if placeholder_val != np.inf:
                valid_results.append(('MCMC-MICE_V2', placeholder_val))
            
            if valid_results:
                winner = min(valid_results, key=lambda x: x[1])[0]
                winner_counts[winner] += 1
            else:
                winner_counts['Failed'] += 1
        
        labels = list(winner_counts.keys())
        values = list(winner_counts.values())
        colors = ['C0', 'C1', 'C2', 'C3']
        
        wedges, texts, autotexts = ax4.pie(values, labels=labels, autopct='%1.1f%%', 
                                          colors=colors[:len(labels)])
        #ax4.set_title('Overall Winner Distribution')
        
        fig.tight_layout()
        fig.subplots_adjust(hspace=0.40, bottom=0.16)
        
        if fname is not None:
            plt.savefig(fname, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
# Convergence diagnostics plots for MCMC
    def convergence_plots(self, mcmc_chain, chain_label="chain1", target_col=None,
                      run_number=None, output_dir="./plots_RWM"):

        # Choose what you want to show: variance or std dev
        eta_samples = mcmc_chain.pos_eta.flatten()          # eta = log(τ²)
        var_samples = np.exp(eta_samples)                   # τ² (variance)
        # sd_samples  = np.exp(0.5 * eta_samples)          # τ (std dev), if preferred
    
        fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    
        # Trace (use variance or sd)
        axes[0].plot(var_samples)
        axes[0].set_xlabel("Iteration", fontsize=20)
        axes[0].set_ylabel("Variance (τ²)", fontsize=20)    # or "Std dev (τ)" if you plot sd
        axes[0].tick_params(axis='both', labelsize=18)
        axes[0].grid(True)
    
        # Histogram
        sns.histplot(var_samples, bins=50, kde=True, ax=axes[1])
        axes[1].set_xlabel("Variance (τ²)", fontsize=20)
        axes[1].set_ylabel("Density", fontsize=20)
        axes[1].tick_params(axis='both', labelsize=18)
        axes[1].grid(True)
    
        # Autocorrelation (eta is fine, or use var_samples)
        max_lags = min(50, len(eta_samples) - 1)
        sm.graphics.tsa.plot_acf(eta_samples, lags=max_lags, ax=axes[2])
        axes[2].set_xlabel("Lag", fontsize=20)
        axes[2].set_ylabel("Autocorrelation", fontsize=20)
        axes[2].tick_params(axis='both', labelsize=18)
        axes[2].grid(True)
    
        plt.tight_layout()
    
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            parts = ["convergence_plot"]
            if target_col: parts.append(target_col)
            if run_number is not None: parts.append(f"run{run_number}")
            parts.append(chain_label)
            filepath = os.path.join(output_dir, "_".join(parts) + ".png")
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"✅ Saved: {filepath}")
    
        plt.show()
    

    def plot_credible_interval_trace(self, samples, param_name="tau", output_dir="./plots_RWM", run_number=None):
        """
        Plot trace with 95% rolling credible intervals over iterations for a given parameter.

        Parameters:
        - samples: np.array of shape (n_samples,)
        - param_name: name of the parameter (for labeling and filename)
        - output_dir: where to save the plot
        - run_number: optional run number for filename
        """
        legend_kwargs = dict(
            loc='upper center',          # anchor the legend’s RIGHT side
            bbox_to_anchor=(0.5, -0.18), # x<0 pushes it left of the axes; y=0.5 centers vertically
            ncol=1,
            fontsize=10, frameon=False,
            handlelength=2.0, handleheight=1.0, markerscale=1.0,
            borderaxespad=0.0
        )
        n = len(samples)
        window = min(100, max(10, n // 20))  # Set rolling window size

        rolling_mean = np.convolve(samples, np.ones(window)/window, mode='valid')

        lower = []
        upper = []
        for i in range(window, n + 1):
            segment = samples[i - window:i]
            lower.append(np.percentile(segment, 2.5))
            upper.append(np.percentile(segment, 97.5))

        iterations = np.arange(window, n + 1)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(iterations, rolling_mean, label=f"Rolling Mean of {param_name}", color="blue")
        ax.fill_between(iterations, lower, upper, color='blue', alpha=0.3, label="95% CI")
        ax.set_xlabel("Iteration", fontsize=16)
        ax.set_ylabel(param_name, fontsize=16)
        ax.tick_params(axis='both', labelsize=14)
        #plt.title(f"Trace with 95% CI for {param_name}")
        # legend (capture handle for saving)
        leg = ax.legend(**legend_kwargs)

        # layout
        fig.tight_layout()
        fig.subplots_adjust(left=0.)  # add left margin for outside legend

       # save
        os.makedirs(output_dir, exist_ok=True)
        fname = f"ci_trace_{param_name}"
        if run_number is not None:
            fname += f"_run{run_number}"
        filepath = os.path.join(output_dir, fname + ".png")

        fig.savefig(filepath, dpi=300, bbox_inches='tight', bbox_extra_artists=(leg,))
        plt.show()
        plt.close(fig)
        print(f"📈 Saved CI trace for {param_name}: {output_dir}")

    def plot_prediction_with_ci(self, y_true, train_sims, title=None, save_path=None, run_number=None):
        """
        Plots the observed vs. modeled predictions with 95% CI from MCMC simulations.

        Parameters:
        - y_true: (n,) true values
        - train_sims: (n_samples, n) MCMC simulations of predictions
        - title: plot title
        - save_path: if provided, saves the figure to this path
        """
        # Compute posterior predictive mean and 95% CI
        pred_mean = np.mean(train_sims, axis=0)
        lower = np.percentile(train_sims, 2.5, axis=0)
        upper = np.percentile(train_sims, 97.5, axis=0)
        timesteps = np.arange(len(y_true))

        # Plot
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(timesteps, y_true, label="Observed", color='blue', linewidth=1.5)
        ax.plot(timesteps, pred_mean, label="Modelled", color='red', linewidth=1.5)
        ax.fill_between(timesteps, lower, upper, color='red', alpha=0.2, label="Modelled 95% CI")

        #plt.title(title)
        ax.set_xlabel("Timestep", fontsize=16)
        ax.set_ylabel(f"Y ({title})" if title else "Y", fontsize=16)
        ax.tick_params(axis='both', labelsize=14)
        ax.grid(True, alpha=0.3)
        leg = ax.legend(
            loc='upper center', bbox_to_anchor=(0.5, -0.18),
            ncol=3, fontsize=10, frameon=False,
            handlelength=2.0, handleheight=1.0, markerscale=1.0
        )
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.18) 
        if save_path:                                  # <-- guard
            fig.savefig(save_path, dpi=300, bbox_inches='tight', bbox_extra_artists=(leg,))
        else:
            plt.show()
        plt.close(fig)
        print(f"📈 Saved plot to {save_path}")
       


def visualize_single_run_results(complete_data, missing_data, imputed_datasets_dict, 
                                target_col, missing_indices, true_values, predictions_dict,
                                run_number=1, time_col='Time', save_plots=False, output_dir='./plots_RWM'):
    """
    Visualize results from a single experimental run
    
    Parameters:
    -----------
    complete_data : pd.DataFrame
        Original complete dataset
    missing_data : pd.DataFrame
        Dataset with missing values
    imputed_datasets_dict : dict
        Dictionary of imputed datasets by method
    target_col : str
        Target column name
    missing_indices : list
        Indices of missing values
    true_values : pd.Series
        True values for missing indices
    predictions_dict : dict
        Dictionary of predictions by method
    run_number : int
        Current run number for labeling
    time_col : str
        Time column name
    save_plots : bool
        Whether to save plots to files
    output_dir : str
        Directory to save plots
    """
    
    visualizer = MCMCMICEVisualizer(time_col=time_col)
    
    # Plot imputed datasets comparison
    fname1 = f'{output_dir}/imputation_comparison_run{run_number}_{target_col}.png' if save_plots else None
    visualizer.plot_imputed_datasets_comparison(
        complete_data=complete_data,
        missing_data=missing_data,
        imputed_datasets_dict=imputed_datasets_dict,
        target_col=target_col,
        missing_indices=missing_indices,
        dataset_name=f'Run {run_number}',
        fname=fname1
    )
    
    # Plot prediction accuracy comparison
    fname2 = f'{output_dir}/prediction_accuracy_run{run_number}_{target_col}.png' if save_plots else None
    visualizer.plot_prediction_accuracy_comparison(
        true_values=true_values.values,
        predictions_dict=predictions_dict,
        target_col=target_col,
        dataset_name=f'Run {run_number}',
        fname=fname2
    )

def visualize_experiment_summary(summary_results, all_results, save_plots=False, output_dir='./plots_RWM'):
    """
    Create summary visualizations for the entire 30-run experiment
    
    Parameters:
    -----------
    summary_results : dict
        Summary statistics from all runs
    all_results : dict
        All individual run results
    save_plots : bool
        Whether to save plots to files
    output_dir : str
        Directory to save plots
    """
    
    visualizer = MCMCMICEVisualizer()
    
    # Create plots for each metric
    for metric in ['NRMSE', 'NMAE', 'NMRE']:
        fname = f'{output_dir}/experiment_summary_{metric}.png' if save_plots else None
        visualizer.plot_experiment_summary(
            summary_results=summary_results,
            all_results=all_results,
            metric=metric,
            fname=fname
        )