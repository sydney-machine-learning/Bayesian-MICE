# Bayesian-MICE
This repository contains the implementation and experimental results for tBayes-MICE, a Bayesian extension of Multiple Imputation by Chained Equation (MICE) designed for uncertainty-aware imputation of time-series data. The method integrated Markov Chain Monte Cralo (MCMC) samply within the fully Condition Specification (FCS) framework and is evaluated on both environmental and clinical datasets. 

# Overview
Missing data are pervasive in real-world time-series applications, particularly in environmental monitoring and healthcare, where reliable uncertainty quantification is essential. Bayes-MICE extends classical MICE by replacing deterministic regression updates with Bayesian regression models whose parameters and imputations are jointly sampled via Markov Chain Monte Carlo (MCMC).
The method support **Random Walk Metropolis (RWM)** or other samplers, with theoretically motivated optimal scaling to improve convergence and mixing.

# Repository structure and Contents
The repository contents four main folders that organise the projects's codebase: **Datasets**, **MCMC_MICE_codes**, **AirQuality_Plots** and **PhysioNet_Plots**.
Bayesian-MICE/
├── Datasets/                    # All datasets used in experiments
│   ├── AirQualityUCI.csv        # Original AirQuality data (hourly, unprocessed)
│   ├── Data_subset_AirQuality.csv       # AirQuality after removing original NaNs
│   ├── Data_with_missing_AirQuality.csv # AirQuality with artificial missing values
│   ├── physionet_5000patients.csv       # PhysioNet tabular data (≤60% missingness)
│   ├── physio_subdata.csv               # PhysioNet after removing all NaNs
│   └── physio_with_missing.csv          # PhysioNet with artificial missing values
│
├── MCMC_MICE_codes/             # Core implementation
│   ├── placeholder.py           # Missing value initialisation (mean + time-aware)
│   ├── PhysioData_Loader.py     # Raw PhysioNet → structured format + masking
│   ├── MCMC_CHAIN.py            # MCMC samplers (RWM)
│   ├── SimpleMCMC.py            # Lagged predictors + parallel MCMC chains
│   ├── Run_Single_MCMC.py       # MCMC within each MICE iteration + convergence
│   ├── Comparison_runs.py       # 30-run multiple imputation comparison
│   ├── Run_experiments.py       # Full experimental workflow manager
│   ├── Visualisation.py         # All plots used in the paper
│   ├── BRITS.py                 # BRITS baseline (via pypots)
│   └── packages.py              # Full package list
│
├── AirQuality_Plots/            # Figures from AirQuality experiments
├── PhysioNet_Plots/             # Figures from PhysioNet experiments
├── requirements.txt             # Python dependencies
└── README.md

6. **Requirements**
   The implementations relies on standard scientific Python libraries:
   pip install numpy pandas scikit-learn matplotlib arviz scipy pypots

# Reproducing the experiment
* Step 1 — Prepare the data                            # Load and process PhysioNet data
  python MCMC_MICE_codes/PhysioData_Loader.py          # Output: Datasets/physio_subdata.csv and Datasets/physio_with_missing.csv
* Step 2 — Run comparison across methods               # Compare tBayes-MICE_V1, tBayes-MICE_V2 and BRITS over 30 runs
  python MCMC_MICE_codes/Comparison_runs.py
* Step 3 — Run the full experiment                     # Run all experiments (both datasets, 30 runs each)
  python MCMC_MICE_codes/Run_experiments.py
* Step 4 — Generate plots                              # Reproduce all figures from the paper
  python MCMC_MICE_codes/Visualisation.py

# Reproducibility
   * Random seeds are controlled for reproducibility.
   * Experiments were run on the Katana High Performance Computing (HPC) cluster, supported by the University of New South Wales (DOI:10.26190/669XA286).
  
9. **Citation**
   If you use this code or results in your work, please cite the corresponding paper (details to be added).
