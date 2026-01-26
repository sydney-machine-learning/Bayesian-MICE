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
 
        COLORS = {
            "Original": "black",
            "MICE": "#E69F00",          # bold orange
            "MCMC_MICE_V1": "#009E73",  # strong green
            "MCMC_MICE_V2": "#D55E00",  # strong red
            "BRITS": "#7B61FF",          # bold violet
            'missing_points': '#CC79A7' # Pink/Magenta for missing point markers
        }
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
        ax1.plot(x_axis, complete_slice, color=COLORS["Original"],
             linewidth=2.5, alpha=0.95, label="Original")
        #ax1.set_title('Original Complete Data', fontweight='bold')
        ax1.set_xlabel(x_label, fontsize=18)
        ax1.set_ylabel(target_col, fontsize=18)
        ax1.tick_params(axis='both', labelsize=16)
        ax1.grid(True, alpha=0.3)
        ax1.legend(
            fontsize=14, frameon=False,
            handlelength=2.0, handleheight=1.0, markerscale=1.0,
            loc='upper right'
        )
        # ==========================================
        # Plot 2: Original vs MICE
        # ==========================================
        ax2 = axes[0, 1]
        ax2.plot(x_axis, complete_slice, color=COLORS["Original"], linewidth=2.5, alpha=0.95, label="Original")
        
        if 'MICE' in imputed_datasets_dict:
            mice_slice = imputed_datasets_dict['MICE'][target_col].iloc[start_idx:end_idx]
            ax2.plot(x_axis, mice_slice, color=COLORS['MICE'], linewidth=2.0, alpha=0.85, 
                    label='MICE', linestyle='--')
            
            # Highlight imputed points
            if missing_indices is not None:
                window_missing = [idx for idx in missing_indices if start_idx <= idx < end_idx]
                if window_missing:
                    missing_x = [idx for idx in window_missing]
                    imputed_y = [mice_slice.iloc[idx - start_idx] for idx in window_missing]
                    ax2.scatter(missing_x, imputed_y, c=COLORS['MICE'], s=15, alpha=0.7, 
                              edgecolors='none', zorder=5)
        
        ax2.set_xlabel(x_label, fontsize=18)
        ax2.set_ylabel(target_col, fontsize=18)
        ax2.tick_params(axis='both', labelsize=16)
        ax2.grid(True, alpha=0.3)
        ax2.legend(
                fontsize=14, frameon=False,
                handlelength=2.0, handleheight=1.0, markerscale=1.0,
                loc='upper center', bbox_to_anchor=(0.5, -0.18),
                ncol=2)
        
        # -------------------- Plot 3: Original vs MCMC methods --------------------
        ax3 = axes[1, 0]
        ax3.plot(x_axis, complete_slice, color=COLORS["Original"],
                 linewidth=2.5, alpha=0.95, label="Original")
        
        for method in ["MCMC_MICE_V1", "MCMC_MICE_V2"]:
            if method in imputed_datasets_dict:
                mcmc_slice = imputed_datasets_dict[method][target_col].iloc[start_idx:end_idx]
                ax3.plot(
                    x_axis, mcmc_slice,
                    color=COLORS[method],
                    linewidth=2.0,
                    linestyle="--",
                    alpha=0.85,
                    label=method.replace("_", " ")
                )
        
        ax3.set_xlabel(x_label, fontsize=18)
        ax3.set_ylabel(target_col, fontsize=18)
        ax3.tick_params(axis="both", labelsize=16)
        ax3.grid(True, alpha=0.3)
        ax3.legend(
                fontsize=14, frameon=False,
                handlelength=2.0, handleheight=1.0, markerscale=1.0,
                loc='upper center', bbox_to_anchor=(0.5, -0.18),
                ncol=3
            )

        # ==========================================
        # Plot 4: Original vs BRITS
        # ==========================================
        ax4 = axes[1, 1]
        ax4.plot(x_axis, complete_slice, color=COLORS["Original"], linewidth=2.5, alpha=0.95, label="Original")
        
        if 'BRITS' in imputed_datasets_dict:
            brits_slice = imputed_datasets_dict['BRITS'][target_col].iloc[start_idx:end_idx]
            ax4.plot(x_axis, brits_slice, color=COLORS['BRITS'], linewidth=2.0, alpha=0.85,
                    label='BRITS', linestyle='--')
            
            # Highlight imputed points
            if missing_indices is not None:
                window_missing = [idx for idx in missing_indices if start_idx <= idx < end_idx]
                if window_missing:
                    missing_x = [idx for idx in window_missing]
                    imputed_y = [brits_slice.iloc[idx - start_idx] for idx in window_missing]
                    ax4.scatter(missing_x, imputed_y, c=COLORS['BRITS'], s=15, alpha=0.7,
                              edgecolors='none', zorder=5)
        
        ax4.set_xlabel(x_label, fontsize=18)
        ax4.set_ylabel(target_col, fontsize=18)
        ax4.tick_params(axis='both', labelsize=16)
        ax4.grid(True, alpha=0.3)
        ax4.legend(
                fontsize=14, frameon=False,
                handlelength=2.0, handleheight=1.0, markerscale=1.0,
                loc='upper center', bbox_to_anchor=(0.5, -0.18),
                ncol=2
            )
        
        fig.tight_layout()
        fig.subplots_adjust(hspace=0.40, bottom=0.16) 
        if fname is not None:
            plt.savefig(fname, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_imputation_errors(self, complete_data, imputed_datasets_dict, 
                            missing_indices, target_col=None, fname=None):
        """
        Plot prediction errors (Actual - Predicted) for each method in a 2-column grid.
        All methods share the same y-axis scale for fair comparison.
        
        Parameters:
        -----------
        complete_data : pd.Series or pd.DataFrame
            Original complete dataset
        imputed_datasets_dict : dict
            Dictionary with method names as keys and imputed Series as values
        missing_indices : list
            Indices of missing values
        target_col : str, optional
            Column name if DataFrame
        fname : str, optional
            Filename to save the plot
        """
        
        COLORS = {
            "MICE": "#E69F00",          # bold orange
            "MCMC_MICE_V1": "#009E73",  # strong green
            "MCMC_MICE_V2": "#D55E00",  # strong red
            "BRITS": "#7B61FF",          # bold violet
        }
        
        # Handle Series vs DataFrame
        if isinstance(complete_data, pd.Series):
            true_values = complete_data.loc[missing_indices].values
        else:
            true_values = complete_data[target_col].loc[missing_indices].values
        
        # ========================================
        # Calculate errors for each method
        # ========================================
        errors_dict = {}
        for method_name, imputed_data in imputed_datasets_dict.items():
            if isinstance(imputed_data, pd.Series):
                imputed_values = imputed_data.loc[missing_indices].values
            else:
                imputed_values = imputed_data[target_col].loc[missing_indices].values
            
            # Error = Actual - Predicted
            errors = true_values - imputed_values
            errors_dict[method_name] = errors
        
        # ========================================
        # Calculate global y-axis limits across ALL methods
        # ========================================
        all_errors = np.concatenate(list(errors_dict.values()))
        global_min = np.min(all_errors)
        global_max = np.max(all_errors)
        
        # Add 5% padding for better visualization
        error_range = global_max - global_min
        padding = 0.05 * error_range if error_range > 0 else 1.0
        ylim_min = global_min - padding
        ylim_max = global_max + padding
        
        # Determine grid size (2 columns)
        n_methods = len(errors_dict)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # Flatten axes for easier iteration
        if n_methods == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        # X-axis: missing value positions
        x_positions = np.arange(len(missing_indices))
        
        # ========================================
        # Plot each method with SHARED y-axis
        # ========================================
        for idx, (method_name, errors) in enumerate(errors_dict.items()):
            ax = axes[idx]
            color = COLORS.get(method_name, "#333333")
            
            # Plot errors as scatter + line
            ax.plot(x_positions, errors, color=color, linewidth=1.5, alpha=0.6, 
                    marker='o', markersize=4, label=method_name)
            
            # Add zero line
            ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
            
            # Calculate statistics
            mae = np.mean(np.abs(errors))
            rmse = np.sqrt(np.mean(errors**2))
            
            # Add statistics text
            stats_text = f'MAE: {mae:.3f}\nRMSE: {rmse:.3f}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                    fontsize=11, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # ✅ SET SHARED Y-AXIS LIMITS FOR ALL SUBPLOTS
            ax.set_ylim(ylim_min, ylim_max)
            
            ax.set_xlabel('Missing Value Index', fontsize=18)
            ax.set_ylabel('Error (Actual - Predicted)', fontsize=18)
            ax.set_title(method_name, fontsize=13, pad=10)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='both', labelsize=16)
        
        # Hide extra subplots
        for idx in range(n_methods, len(axes)):
            axes[idx].axis('off')
        
        fig.tight_layout()
        
        if fname is not None:
            plt.savefig(fname, dpi=300, bbox_inches='tight')
            print(f"📊 Plot saved to: {fname}")
            plt.close()
        else:
            plt.show()
        
    
    def plot_prediction_accuracy_comparison(self, true_values, predictions_dict, 
                                          method_names=None, target_col=None, 
                                          dataset_name=None, fname=None):
        
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
            ncol=2,
            fontsize=14, frameon=False,
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
        ax1.set_xlabel('True Values', fontsize=18)
        ax1.set_ylabel('Predicted Values', fontsize=18)
        ax1.tick_params(axis='both', labelsize=16)
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
        ax2.set_xlabel('True Values', fontsize=18)
        ax2.set_ylabel('Residuals (Predicted - True)', fontsize=18)
        ax2.tick_params(axis='both', labelsize=16)
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
        
        ax3.set_xlabel('Sorted Index', fontsize=18)
        ax3.set_ylabel('Values', fontsize=18)
        ax3.tick_params(axis='both', labelsize=16)
        #ax3.set_title('Sorted True vs Predicted Values')
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18),
            ncol=min(3, len(method_names)),
            fontsize=14, frameon=False,
            handlelength=2.0, handleheight=1.0, markerscale=1.0
        )
        
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
        
        ax4.set_ylabel('Absolute Error', fontsize=18)
        ax4.tick_params(axis='both', labelsize=16)
        plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')
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
    
        # Check which methods have data in the results
        available_methods = set()
        for col in columns:
            for key in summary_results[col].keys():
                if f'_{metric}_mean' in key:
                    method = key.replace(f'_{metric}_mean', '')
                    available_methods.add(method)
        
        # Sort methods for consistent ordering
        method_order = ['MICE', 'MCMC_MICE_V1', 'MCMC_MICE_V2', 'BRITS']
        methods = [m for m in method_order if m in available_methods]
        
        print(f"📊 Plotting {len(methods)} methods: {', '.join(methods)}")
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Mean performance with error bars
        ax1 = axes[0]
        x_pos = np.arange(len(columns))
        
        # Dynamic width based on number of methods
        width = 0.8 / len(methods) if len(methods) > 0 else 0.2
        
        legend_kwargs = dict(
            loc='upper center', bbox_to_anchor=(0.5, -0.32),
            ncol=min(4, len(methods)), 
            fontsize=14, frameon=False,
            handlelength=2.0, handleheight=1.0, markerscale=1.0
        )
        
        # ✅ FIX 3: Color palette for up to 4 methods
        colors = ['C0', 'C1', 'C2', 'C3', 'C4']

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
                        means.append(0)  
                        stds.append(0)
                else:
                    means.append(0)
                    stds.append(0)
            
            bars = ax1.bar(x_pos + i*width, means, width, 
                        yerr=stds, capsize=5, alpha=0.8,
                        color=colors[i % len(colors)],  
                        label=method.replace('_', '-'))
        
        ax1.set_xlabel('Variables', fontsize=20)
        ax1.set_ylabel(f'{metric}', fontsize=20)
        ax1.set_xticks(x_pos + width * (len(methods) - 1) / 2)  # ✅ Center the ticks
        ax1.set_xticklabels(columns, rotation=45, ha='right')
        ax1.tick_params(axis='both', labelsize=16)
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.legend(**legend_kwargs)

        
        # Plot 2: Distribution of results across runs (box plot)
        ax2 = axes[1]
        if len(columns) > 0:
            # Pick first column for detailed analysis
            col = columns[0]
            method_data = []
            method_labels = []
            
            for method in methods:  # ✅ Use dynamic methods list
                metric_key = f'{method}_{metric.lower()}'
                if metric_key in all_results[col]:
                    data = [x for x in all_results[col][metric_key] if x != np.inf]
                    if data:
                        method_data.append(data)
                        method_labels.append(method.replace('_', '-'))
            
            if method_data:
                box_plot = ax2.boxplot(method_data, labels=method_labels, patch_artist=True)
                
                # ✅ FIX 4: Color boxes based on actual number of methods
                for patch, color in zip(box_plot['boxes'], colors[:len(method_data)]):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                
                # Style the box plot elements
                for med in box_plot['medians']:
                    med.set(color='k', linewidth=1.5)
                for w in box_plot['whiskers'] + box_plot['caps']:
                    w.set(color='0.3', linewidth=1.5)
                for f1 in box_plot.get('fliers', []):
                    f1.set(marker='o', markersize=3, alpha=0.6, markeredgecolor='0.3')
        
        ax2.set_ylabel(f'{metric}', fontsize=20)
        ax2.tick_params(axis='both', labelsize=18)
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        ax2.grid(True, alpha=0.3, axis='y')    
        
        fig.tight_layout()
        fig.subplots_adjust(hspace=0.40, bottom=0.20)  # ✅ More space for legend
        
        if fname is not None:
            plt.savefig(fname, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
            
# Convergence diagnostics plots for MCMC
    def convergence_plots(self, mcmc_chain, chain_label="chain1", target_col=None,
                      run_number=None, sampler ='RWM', output_dir="./plots_RWM_BRITS"):

       # --- Samples ---
        eta_samples = np.asarray(mcmc_chain.pos_eta).ravel()   # η = log(τ²)
        var_samples = np.exp(eta_samples)                      # τ² (variance)
        n = len(var_samples)
        x = np.arange(n)
        # --- Figure ---
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.4), constrained_layout=True)

        # ===================== Trace (variance) =====================
        ax = axes[0]
        ax.plot(x, var_samples, linewidth=0.9)
        ax.set_xlabel("Iteration (post–burn-in)", fontsize=24)
        ax.set_ylabel("Variance (τ²)", fontsize=24)
        ax.set_xlim(0, n - 1)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(axis='both', labelsize=18)
        ax.tick_params(axis='x', pad=15)
        ax.grid(True, which="major", alpha=0.35)
        ax.grid(True, which="minor", alpha=0.15)

        # ===================== Histogram (+ optional KDE) =====================
        ax = axes[1]
        sns.histplot(var_samples, bins=50, kde=True, ax=axes[1])
        ax.set_xlabel("Variance (τ²)", fontsize=24)
        ax.set_ylabel("Density", fontsize=24)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(axis='both', labelsize=18)
        ax.grid(True, which="major", alpha=0.35)
        ax.grid(True, which="minor", alpha=0.15)

        # ===================== Autocorrelation of η =====================
        ax = axes[2]
        max_lags = min(50, len(eta_samples) - 1)
        sm.graphics.tsa.plot_acf(eta_samples, lags=max_lags, ax=axes[2])
        ax.set_title("") 
        ax.set_xlabel("Lag", fontsize=24)
        ax.set_ylabel("Autocorrelation", fontsize=24)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(axis='both', labelsize=18)
        ax.grid(True, which="major", alpha=0.35)
        ax.grid(True, which="minor", alpha=0.15)

        # Consistent numeric formatting
        for ax in axes:
            ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))

        # ----- Save -----
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            parts = ["convergence_plot"]
            if target_col: parts.append(target_col)
            if run_number is not None: parts.append(f"run{run_number}")
            parts.append(chain_label)
            parts.append(sampler)
            filepath = os.path.join(output_dir, "_".join(parts) + ".png")
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"✅ Saved: {filepath}")

        plt.show()

    def plot_credible_interval_trace(self, samples, param_name="tau", output_dir="./plots_RWM_BRITS", run_number=None, window_idx=None, target=None):
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
            bbox_to_anchor=(0.5, -0.26), # x<0 pushes it left of the axes; y=0.5 centers vertically
            ncol=2,
            fontsize=12, frameon=False,
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
        ax.set_xlabel("Iteration", fontsize=18)
        ax.set_ylabel(param_name, fontsize=18)
        ax.tick_params(axis='both', labelsize=16)
        #plt.title(f"Trace with 95% CI for {param_name}")
        # legend (capture handle for saving)
        leg = ax.legend(**legend_kwargs)

        # layout
        fig.tight_layout()
        fig.subplots_adjust(left=0.18)  # add left margin for outside legend

       # save
        os.makedirs(output_dir, exist_ok=True)
        fname = f"ci_trace_{param_name}"
        if run_number is not None:
            fname += f"_run{run_number}"
            fname += f"_window{window_idx}"
            fname += f"_{target}"
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
        ax.set_xlabel("Timestep", fontsize=18)
        ax.set_ylabel(f"Y ({title})" if title else "Y", fontsize=18)
        ax.tick_params(axis='both', labelsize=16)
        ax.grid(True, alpha=0.3)
        leg = ax.legend(
            loc='upper center', bbox_to_anchor=(0.5, -0.18),
            ncol=3, fontsize=14, frameon=False,
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
                                run_number=1, time_col='Time', save_plots=False, output_dir='./plots_RWM_BRITS'):
    
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

def visualize_experiment_summary(summary_results, all_results, save_plots=False, output_dir='./plots_RWM_BRITS'):
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
