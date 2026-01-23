from packages import *


def run_brits_separately(
    complete_data: pd.DataFrame,
    missing_data: pd.DataFrame,
    numeric_cols: List[str],
    time_col: str = 'Date_Time',
    n_runs: int = 30,
    output_path: str = './brits_results.pkl',
    n_steps: int = 48,
    stride: int = 24,
    rnn_hidden_size: int = 32,
    epochs: int = 50,
    device: str = "cuda"
) -> Dict:

    print("=" * 80)
    print("🚀 BRITS SEPARATE EXPERIMENT (GPU) - WITH SCALING")
    print("=" * 80)

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    print(f"🖥️ Device: {device}")

    missing_mask = missing_data.isnull() & ~complete_data.isnull()
    cols_with_missing = [c for c in numeric_cols if missing_mask[c].sum() > 0]

    # ---------------- helpers ----------------
    def calculate_metrics(y_true, y_pred, min_std=1e-8):
        y_true = np.asarray(y_true, dtype=float).flatten()
        y_pred = np.asarray(y_pred, dtype=float).flatten()

        if len(y_true) != len(y_pred) or len(y_true) == 0:
            return {"RMSE": np.inf, "NMAE": np.inf, "NMRE": np.inf, "NRMSE": np.inf}

        valid = ~(np.isnan(y_true) | np.isnan(y_pred) |
                  np.isinf(y_true) | np.isinf(y_pred))

        if not np.any(valid):
            return {"RMSE": np.inf, "NMAE": np.inf, "NMRE": np.inf, "NRMSE": np.inf}

        y_t = y_true[valid]
        y_p = y_pred[valid]

        rmse = np.sqrt(np.mean((y_p - y_t) ** 2))
        mae = np.mean(np.abs(y_p - y_t))
        std_t = np.std(y_t)

        nmae = mae / std_t if std_t >= min_std else mae
        nrmse = rmse / std_t if std_t >= min_std else rmse

        abs_true = np.abs(y_t)
        denom = np.maximum(abs_true, 1e-8)
        rel_err = np.abs(y_p - y_t) / denom
        finite_rel = rel_err[np.isfinite(rel_err) & (rel_err < 10.0)]
        nmre = np.mean(finite_rel) * 100 if len(finite_rel) > 0 else np.inf

        return {"RMSE": rmse, "NMAE": nmae, "NMRE": nmre, "NRMSE": nrmse}

    def make_windows(data):
        arr = data[numeric_cols].to_numpy(dtype=float)
        windows, starts = [], []
        for s in range(0, len(arr) - n_steps + 1, stride):
            w = arr[s:s + n_steps]
            if np.any(~np.isnan(w)):
                windows.append(w)
                starts.append(s)
        return np.array(windows), starts

    def reconstruct(windows, starts):
        T, F = len(missing_data), len(numeric_cols)
        vals = np.zeros((T, F), dtype=np.float64)
        cnts = np.zeros((T, F), dtype=np.float64)
        for w, s in zip(windows, starts):
            vals[s:s + n_steps] += w
            cnts[s:s + n_steps] += 1
        cnts[cnts == 0] = 1
        return vals / cnts

    # ---------------- storage ----------------
    results = {
        col: {"BRITS_rmse": [], "BRITS_nmae": [], "BRITS_nmre": [], "BRITS_nrmse": []}
        for col in cols_with_missing
    }

    timing = []
    imputed_datasets = {}
    predictions = {}

    # Generate windows
    X_train, starts = make_windows(missing_data)
    
    # ✅ FIX #3: Check BEFORE any processing
    if X_train.shape[0] == 0:
        raise RuntimeError("No valid BRITS windows generated")
    
    # ✅ FIX #3: Renamed to avoid shadowing parameter
    n_windows, n_steps_actual, n_features = X_train.shape
    print(f"📊 Generated {n_windows} windows of shape ({n_steps_actual}, {n_features})")

    # ✅ FIX #2: NaN-safe scaling
    X_flat = X_train.reshape(-1, n_features)
    
    # Compute mean and std ignoring NaN
    scaler_mean = np.nanmean(X_flat, axis=0)
    scaler_scale = np.nanstd(X_flat, axis=0)
    scaler_scale[scaler_scale < 1e-8] = 1.0  # Avoid division by zero
    
    # Scale the data (NaN remains NaN)
    X_scaled_flat = (X_flat - scaler_mean) / scaler_scale
    X_train_scaled = X_scaled_flat.reshape(n_windows, n_steps_actual, n_features)
    
    print(f"✅ Data scaled using NaN-safe method")
    print(f"   Feature means: {scaler_mean[:3]}... (first 3)")
    print(f"   Feature stds:  {scaler_scale[:3]}... (first 3)")

    # ---------------- runs ----------------
    for run in range(n_runs):
        print(f"\n🔁 BRITS Run {run + 1}/{n_runs}")
        torch.manual_seed(1000 + run)
        np.random.seed(1000 + run)

        start = time_module.time()

        brits = BRITS(
            n_steps=n_steps_actual,  # ✅ Use actual window size
            n_features=n_features,
            rnn_hidden_size=rnn_hidden_size,
            epochs=epochs,
            device=device
        )
        
        # ✅ FIX #1: Correct variable name (lowercase 's')
        brits.fit({"X": X_train_scaled})

        # Impute in scaled space
        imputed_windows_scaled = brits.impute({"X": X_train_scaled})
        
        # ✅ FIX #2: Inverse transform using stored parameters
        imputed_flat_scaled = imputed_windows_scaled.reshape(-1, n_features)
        imputed_flat_original = imputed_flat_scaled * scaler_scale + scaler_mean
        imputed_windows = imputed_flat_original.reshape(n_windows, n_steps_actual, n_features)
        
        # Reconstruct
        full_imputed = reconstruct(imputed_windows, starts)

        brits_df = pd.DataFrame(
            full_imputed,
            index=missing_data.index,
            columns=numeric_cols
        )

        imputed_datasets[run] = {}
        predictions[run] = {}

        for col in cols_with_missing:
            idx = missing_mask[col]
            y_true = complete_data.loc[idx, col].values
            y_pred = brits_df.loc[idx, col].values

            metrics = calculate_metrics(y_true, y_pred)

            results[col]["BRITS_rmse"].append(metrics["RMSE"])
            results[col]["BRITS_nmae"].append(metrics["NMAE"])
            results[col]["BRITS_nmre"].append(metrics["NMRE"])
            results[col]["BRITS_nrmse"].append(metrics["NRMSE"])

            imputed_datasets[run][col] = brits_df
            predictions[run][col] = {
                "BRITS": y_pred,
                "missing_indices": missing_data.index[idx]
            }
            
            print(f"  {col}: RMSE={metrics['RMSE']:.4f}, NRMSE={metrics['NRMSE']:.4f}")

        elapsed = time_module.time() - start
        timing.append(elapsed)
        print(f"⏱️ Run time: {elapsed:.2f}s")

        del brits
        torch.cuda.empty_cache()

    # ---------------- output ----------------
    output = {
        "all_results": results,
        "timing_results": {
            "per_run": timing,
            "total": sum(timing)
        },
        "imputed_datasets": imputed_datasets,
        "predictions": predictions,
        "cols_with_missing": cols_with_missing,
        "config": {
            "n_runs": n_runs,
            "n_steps": n_steps_actual,
            "stride": stride,
            "epochs": epochs,
            "scaled": True
        }
    }

    with open(output_path, "wb") as f:
        pickle.dump(output, f)

    print(f"\n💾 Results saved to: {output_path}")
    print(f"⏱️ Total time: {sum(timing):.2f}s ({sum(timing)/60:.1f} minutes)")

    return output


if __name__ == "__main__":
    data_with_time = pd.read_csv('physionet_5000patients.csv')
    data_with_missing = pd.read_csv('physio_with_missing.csv')
    data_subset = pd.read_csv('physio_subdata.csv')
    time_col = "Time"
    n_runs = 30

    numeric_cols = data_with_missing.select_dtypes(include=[np.number]).columns.tolist()
    if time_col in numeric_cols:
        numeric_cols.remove(time_col)

    brits_results = run_brits_separately(
        complete_data=data_subset,
        missing_data=data_with_missing,
        numeric_cols=numeric_cols,
        time_col=time_col,
        n_runs=n_runs,
        output_path='./brits_results.pkl'
    )

    print(f"\n{'='*100}")
    print("🎉 BRITS EXPERIMENT COMPLETED!")
    print('='*100)