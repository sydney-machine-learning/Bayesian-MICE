# Bayesian-MICE
This repository contains the implementation and experimental results for Bayes-MICE, a Bayesian extension of Multiple Imputation by Chained Equation (MICE) designed for uncertainty-aware imputation of time-series data. The method integrated Markov Chain Monte Cralo (MCMC) samply within the fully Condition Specification (FCS) framework and is evaluated on both environmental and clinical datasets. 

# Overview
Missing data are pervasive in real-world time-series applications, particularly in environmental monitoring and healthcare, where reliable uncertainty quantification is essential. Bayes-MICE extends classical MICE by replacing deterministic regression updates with Bayesian regression models whose parameters and imputations are jointly sampled via MCMC.

Two variants are implemented:
* **MCMC_MICE_V1:** Mean-based initialisation
* **MCMC_MICE_V2:** Time-aware initialisation exploiting temporal structure and autocorrelation
  
Both variants support **Random Walk Metropolis (RWM)** and **Metropolis-Adjusted Langevin Algorithm (MALA)** samplers, with theoretically motivated optimal scaling to improve convergence and mixing.

# Repository structure and Contents
The repository contents four main folders that organise the projects's codebase: **Datasets**, **MCMC_MICE_codes**, **AirQuality_Plots** and **PhysioNet_Plots**.

1. **Datasets**
   Contains the datasets used in the experiments:
   * **AirQualityUCI.csv:** Original unprocessed AirQuality environmental dataset with hourly measurements.
   * **Data_subset_AirQuality.csv:** Complete AirQuality dataset after removal of original missing values.
   * **Data_with_missing_AirQuality.csv:** AirQuality dataset with artifically masked missing values for controlled evaluation.
   * **physionet_5000patients.csv:** Transformed PhysioNet dataset (tabular structure), filtered to remove rows with more than 60% missingness
   * **physio_subdata.csv:** Complete PhysioNet dataset after removal of all missing values.
   * **physio_with_missing.cvs:** PhysioNet dataset with artifically masked missing values.
  
2. **MCMC_MICE_codes**
   Python scripts implementing the Bayes-MICE framework and experiment pipeline:
   * placeholder.py: Initialises missing values for MICE (mean-based and time-aware variants).
   * PhysioData_Loader.py: Converts raw PhysioNet data into structured format, and applies mask missingness masks.
   * MCMC_CHAIN.py: Implements the MCMC samplers (RWM).
   * SimpleMCMC.py: Constructs lagged predictors and runs parallel MCMC chains for convergence diagnostics
   * Run_Single_MCMC.py: Executes MCMC within each MICE iteration and check convergences
   * Comparison_runs.py: Performs multiple imputation across 30 experimental runs for methods comparison.
   * Run_experiments.py: Manages the full experimental workflow.
   * Visualisation.py: Generates the all plots used in the study.
   * BRITS.py: BRITS baseline implementation using pypots.
   * packages.py: Lists all the packages utilised in this study.
  
3. **AirQuality_Plots**
     Contains all figures generated from experiments on the AirQuality dataset.
     
5. **PhysioNet_Plots**
     Contains all figures generated from experiments on the PhysioNet dataset.

6. **Requirements**
   The implementations relies on standard scientific Python libraries:
   * numpy
   * pandas
   * scikit-learn
   * matplotlib
   * arviz
   * scipy
   * pypots

7. **Reproducibility**
   * Random seeds are controlled for reproducibility.
   * Experiments were run on the Katana High Performance Computing (HPC) cluster, supported by the University of New South Wales (DOI:10.26190/669XA286).
  
8. **Citation**
   If you use this code or results in your work, please cite the corresponding paper (details to be added).
