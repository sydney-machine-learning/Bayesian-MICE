from packages import *

class MissingnessPattern():
    def __init__(self):
        """
        Initialize with a DateFrame containing time series data.
        """

    def produce_NA(self, X, p_miss, mecha="MCAR", p_obs=None):
        """
        Generate missing values for specific missing-data mechanism and proportion of missing values.
        Pure NumPy/Pandas implementation.
        
        Parameters
        ----------
        X : np.ndarray or pd.DataFrame, shape (n, d)
            Data for which missing values will be simulated.
        p_miss : float
            Proportion of missing values to generate for variables which will have missing values.
        mecha : str, 
                Indicates the missing-data mechanism to be used. "MCAR" by default, "MAR", "MNAR" 
        opt: str, 
            For mecha = "MNAR", it indicates how the missing-data mechanism is generated: 
            "logistic", "quantile" or "selfmasked".
        p_obs : float
                If mecha = "MAR" or "MNAR", proportion of variables with *no* missing values 
                that will be used for the masking model.
        q : float
            If mecha = "MNAR" and opt = "quantile", quantile level at which the cuts should occur.
        
        Returns
        ----------
        A dictionary containing:
        'X_init': the initial data matrix.
        'X_incomp': the data with the generated missing values.
        'mask': a matrix indexing the generated missing values (True = missing).
        """
        
        # Convert to numpy array if DataFrame
        if isinstance(X, pd.DataFrame):
            X_np = X.values.astype(np.float32)
            columns = X.columns
            is_dataframe = True
        else:
            X_np = X.astype(np.float32)
            columns = None
            is_dataframe = False
        
        if mecha == "MCAR":
            mask = self.MCAR_mask(X_np, p_miss)
        elif mecha == "MAR":
            mask = self.MAR_mask(X_np, p_miss, p_obs)
        else:
            raise ValueError(f"Unknown mechanism: {mecha}")
    
        # Create incomplete data
        X_incomp = X_np.copy()
        X_incomp[mask] = np.nan
        
        # Convert back to DataFrame if input was DataFrame
        if is_dataframe:
            X_init = pd.DataFrame(X_np, columns=columns)
            X_incomp = pd.DataFrame(X_incomp, columns=columns)
            mask = pd.DataFrame(mask, columns=columns)
        else:
            X_init = X_np
        print(f"Generated missing values: {np.sum(mask)} ({np.mean(mask)*100:.2f}% missing)")
        return {
            'X_init': X_init, 
            'X_incomp': X_incomp, 
            'mask': mask
        }
    
    def MCAR_mask(self, X, p_miss):
        """
        Generate Missing Completely At Random (MCAR) mask
        """
        n, d = X.shape
        mask = np.random.rand(n, d) < p_miss
        return mask

    def MAR_mask(self, X, p_miss, p_obs):
        """
        Generate Missing At Random (MAR) mask using logistic regression
        """
        n, d = X.shape
        
        if p_obs is None:
            p_obs = 0.5
        
        # Number of variables that will have missing values
        d_miss = max(1, int(d * (1 - p_obs)))
        # Number of variables that will be fully observed (used as predictors)
        d_obs = d - d_miss
        
        # Randomly select which variables will have missing values
        idxs_miss = np.random.choice(d, d_miss, replace=False)
        idxs_obs = np.array([i for i in range(d) if i not in idxs_miss])
        
        mask = np.zeros((n, d), dtype=bool)
        
        # Generate missing values for each selected variable
        for idx in idxs_miss:
            if len(idxs_obs) == 0:
                # Fallback to MCAR if no observed variables
                mask[:, idx] = np.random.rand(n) < p_miss
            else:
                # Use observed variables to predict missingness
                X_obs = X[:, idxs_obs]
                
                # Standardize the observed variables
                scaler = StandardScaler()
                try:
                    X_obs_scaled = scaler.fit_transform(X_obs)
                except:
                    # If standardization fails, use original values
                    X_obs_scaled = X_obs
                
                # Generate logistic coefficients
                coeffs = np.random.randn(X_obs_scaled.shape[1])
                
                # Compute logistic probabilities
                logits = X_obs_scaled @ coeffs
                probs = 1 / (1 + np.exp(-logits))
                
                # Adjust probabilities to get desired missing rate
                probs = self.adjust_probs_to_rate(probs, p_miss)
                
                # Generate missing mask
                mask[:, idx] = np.random.rand(n) < probs
        
        return mask
    
    def adjust_probs_to_rate(self, probs, target_rate):
        """
        Adjust probabilities to achieve target missing rate
        """
        if target_rate <= 0:
            return np.zeros_like(probs)
        if target_rate >= 1:
            return np.ones_like(probs)
        
        # Sort probabilities and find threshold
        sorted_probs = np.sort(probs)[::-1]  # Descending order
        n_target = int(len(probs) * target_rate)
        
        if n_target >= len(sorted_probs):
            threshold = 0
        else:
            threshold = sorted_probs[n_target]
        
        # Scale probabilities
        adjusted_probs = probs.copy()
        
        # Simple scaling approach
        current_expected = np.mean(probs)
        if current_expected > 0:
            scale_factor = target_rate / current_expected
            adjusted_probs = np.clip(probs * scale_factor, 0, 1)
        
        return adjusted_probs

class PhysioDataLoader:
    def __init__(self):
        # Non-time-series parameters to exclude
        self.static_vars = {'Age', 'Gender', 'Height', 'ICUType', 'Weight', 'RecordID'}

    def process_patient_file_with_time(self, file_path):
        try:
            df = pd.read_csv(file_path)
            df['Value'] = pd.to_numeric(df['Value'], errors='coerce')

            record_id_row = df[df['Parameter'] == 'RecordID']
            if record_id_row.empty:
                print(f"No RecordID in {file_path}")
                return pd.DataFrame()
            record_id = int(record_id_row['Value'].values[0])

            df = df[~df['Parameter'].isin(self.static_vars)].copy()
            if df.empty:
                print(f"No time-series data in {file_path}")
                return pd.DataFrame()

            df['RecordID'] = record_id
            df = df[['RecordID', 'Time', 'Parameter', 'Value']]
            return df

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return pd.DataFrame()

    def combine_all_patients_data(self, folder_path, max_patients=None):
        all_patient_dfs = []
        file_paths = glob(os.path.join(folder_path, '*.txt'))

        if max_patients:
            file_paths = file_paths[:max_patients]

        print(f"Processing {len(file_paths)} patient files...")
        for i, file_path in enumerate(file_paths):
            if i % 100 == 0:
                print(f"  Processed {i}/{len(file_paths)} files...")

            df_patient = self.process_patient_file_with_time(file_path)
            if not df_patient.empty:
                all_patient_dfs.append(df_patient)

        if not all_patient_dfs:
            raise ValueError("No valid patient data found.")

        print(f"Processed {len(all_patient_dfs)} patients")
        return pd.concat(all_patient_dfs, ignore_index=True)

    def convert_time_to_string(self, time_val):
        try:
            if pd.isna(time_val):
                return "00:00"
            if isinstance(time_val, str):
                if ':' in time_val:
                    return time_val
                try:
                    time_val = float(time_val)
                except ValueError:
                    return "00:00"
            if isinstance(time_val, (int, float)):
                hours = int(time_val)
                minutes = int((time_val % 1) * 60)
                return f"{hours:02d}:{minutes:02d}"
            return "00:00"
        except Exception as e:
            print(f"Error converting time: {e}")
            return "00:00"

    def create_physionet_dataset(self, folder_path, target_variables=None, max_patients=None,
                                 missing_threshold=0.7, output_file='patient_time_series_data.csv'):

        if target_variables is None:
            target_variables = ["HR", "Glucose", "HCO3", "Mg", "Na", "Platelets", "WBC"]

        final_df = self.combine_all_patients_data(folder_path, max_patients)

        print("Converting to wide format...")
        df_pivoted = final_df.pivot_table(index=['RecordID', 'Time'], columns='Parameter',
                                          values='Value', aggfunc='first').reset_index()
        df_pivoted.columns.name = None

        print("Converting time format...")
        df_pivoted['Time'] = df_pivoted['Time'].apply(self.convert_time_to_string)

        #print("Creating unique Time_ID column...")
        #df_pivoted['Time_ID'] = df_pivoted['RecordID'].astype(str) + '_' + df_pivoted['Date_Time']

        #df_pivoted = df_pivoted.sort_values(['RecordID', 'Time'])
        #df_pivoted = df_pivoted.drop(columns=['RecordID', 'Date_Time', 'Time'])
        print(f"Before filtering: {df_pivoted.shape}")
        df_pivoted = df_pivoted[df_pivoted.isnull().mean(axis=1) < missing_threshold]
        print(f"After filtering (<{missing_threshold*100:.0f}% missing): {df_pivoted.shape}")

        available_vars = [v for v in target_variables if v in df_pivoted.columns]
        missing_vars = [v for v in target_variables if v not in df_pivoted.columns]

        if missing_vars:
            print(f"Missing variables: {missing_vars}")
        if not available_vars:
            print("No target variables found")
            return None

        final_cols = ['RecordID', 'Time'] + available_vars
        data = df_pivoted[final_cols].copy()
        data = data.sort_values(['RecordID', 'Time']) # Sort by the combined unique identifier
        data.to_csv(output_file, index=False)

        print(f"Data saved to {output_file}. Shape: {data.shape}")
        print(f"Patients: {data['RecordID'].nunique()}, Time points: {len(data)}")
        print(f"Overall missing: {data[available_vars].isnull().sum().sum() / (len(data) * len(available_vars)) * 100:.2f}%")

        return data

    def load_physionet_for_mcmc(self, folder_path='set-a', target_variables=None, max_patients=100):
        if target_variables is None:
            target_variables = ["Glucose", "HCO3", "Mg", "Na", "Platelets", "WBC"]

        data = self.create_physionet_dataset(folder_path=folder_path,
                                             target_variables=target_variables,
                                             max_patients=max_patients,
                                             missing_threshold=0.6,
                                             output_file=f'physionet_{max_patients}patients.csv')

        if data is None:
            return None

        mcmc_data = data.copy()
        physio_data=mcmc_data.dropna()
        physio_data = physio_data.drop("RecordID", axis=1)
        physio_data.to_csv('physio_subdata_fake.csv', index=False)
        physio_with_missing = physio_data.copy()
        numeric_cols = physio_with_missing.select_dtypes(include=[np.number]).columns
        physio_numeric = physio_with_missing[numeric_cols]
        mask_pattern = MissingnessPattern()
        masked_result = mask_pattern.produce_NA(physio_numeric, p_miss=0.4, mecha="MAR", p_obs=0.3)
        physio_with_missing = masked_result['X_incomp']
        physio_with_missing['Time'] = physio_data['Time'].values
        physio_with_missing.to_csv('physio_with_missing.csv', index=False)
        print('Physio_with_missing shape:', physio_with_missing.isnull().sum())
        print(f"\nMCMC-ready dataset shape: {mcmc_data.shape}")
        return mcmc_data, physio_data, physio_with_missing
physionet = PhysioDataLoader()
data_with_time, physio_subdata, physio_with_missing = physionet.load_physionet_for_mcmc(folder_path='set-a', target_variables=None, max_patients=5000)
print(f'Data with time shape: {data_with_time.shape}')