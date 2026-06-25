# Bayesian-MICE
This repository contains the implementation and experimental results for Bayes-MICE, a Bayesian extension of Multiple Imputation by Chained Equation (MICE) designed for uncertainty-aware imputation of time-series data. The method integrated Markov Chain Monte Cralo (MCMC) samply within the fully Condition Specification (FCS) framework and is evaluated on both environmental and clinical datasets. 

# Overview
Missing data are pervasive in real-world time-series applications, particularly in environmental monitoring and healthcare, where reliable uncertainty quantification is essential. tBayes-MICE extends classical MICE by replacing deterministic regression updates with Bayesian regression models whose parameters and imputations are jointly sampled via Markov Chain Monte Carlo (MCMC).
The method support **Random Walk Metropolis (RWM)** or other samplers, with theoretically motivated optimal scaling to improve convergence and mixing.

# Repository structure and Contents
The repository contents four main folders that organise the projects's codebase: **Datasets**, **MCMC_MICE_codes**, **AirQuality_Plots** and **PhysioNet_Plots**.

Bayesian-MICE/
├── Datasets/
│   ├── AirQualityUCI.csv
│   ├── Data_subset_AirQuality.csv
│   ├── Data_with_missing_AirQuality.csv
│   ├── physionet_5000patients.csv
│   ├── physio_subdata.csv
│   └── physio_with_missing.csv
│
├── MCMC_MICE_codes/
│   ├── placeholder.py
│   ├── PhysioData_Loader.py
│   ├── MCMC_CHAIN.py
│   ├── SimpleMCMC.py
│   ├── Run_Single_MCMC.py
│   ├── Comparison_runs.py
│   ├── Run_experiments.py
│   ├── Visualisation.py
│   ├── BRITS.py
│   └── packages.py
│
├── AirQuality_Plots/
├── PhysioNet_Plots/
├── requirements.txt
└── README.md

# Requirements
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

**Reproducibility**
   * Random seeds are controlled for reproducibility.
   * Experiments were run on the Katana High Performance Computing (HPC) cluster, supported by the University of New South Wales (DOI:10.26190/669XA286).
  
**Citation**
   If you use this code or results in your work, please cite the corresponding paper (details to be added).
