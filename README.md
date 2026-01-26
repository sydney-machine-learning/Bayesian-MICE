# Bayesian-MICE
This repository contains the implementation and experimental results for Bayes-MICE, a Bayesian extension of Multiple Imputation by Chained Equation (MICE) designed for uncertainty-aware imputation of time-series data. The method integrated Markov Chain Monte Cralo (MCMC) samply within the fully Condition Specification (FCS) framework and is evaluated on both environmental and clinical datasets. 

# Overview
Missing data are pervasive in real-world time-series applications, particularly in environmental monitoring and healthcare, where reliable uncertainty quantification is essential. Bayes-MICE extends classical MICE by replacing deterministic regression updates with Bayesian regression models whose parameters and imputations are jointly sampled via MCMC.

Two variants are implemented:
* **MCMC_MICE_V1:** Mean-based initialisation
* **MCMC_MICE_V2:** Time-aware initialisation exploiting temporal structure
Both variants support **Random Walk Metropolis (RWM)** and **Metropolis-Adjusted Langevin Algorithm (MALA)** samplers with theoretically motivated optimal scaling.

# Repository structure and Contents
The repository contents four main folders that organise the projects's codebase: **Datasets**, **MCMC_MICE_codes**, **AirQuality_Plots** and **PhysioNet_Plots**.

1. **Datasets**
   This folder contains the datasets used in the study, including:
   * **AirQualityUCI.csv:** The original, unprocessed AirQuality environmental dataset with hourly measurements.
   * **Data_subset_AirQuality.csv:** The complete dataset after removal of original missing values.
   * **Data_with_missing_AirQuality.csv:** The AirQuality dataset with artifically masked missing values for controlled evaluation.
   * **physionet_5000patients.csv:** The transformed (tabular structure) raw PhysioNet dataset after removing rows with more 60% missingness to enhance data quality and reduce the influence of extreme sparsity.
   * **physio_subdata.csv:** The complete PhysioNet dataset after removal of all missing values.
   * **physio_with_missing.cvs:** The PhysioNet dataset with artifically masked missing values.
  
2. **MCMC_MICE_codes**
   This folder includes the python scripts for the Bayes-MICE implementation and experiment:
   * placeholder.py: Initialises the missing values as the first stage of MICE. Containing the two variants: mean-based and time-aware initialisation.
   * PhysioData_Loader.py: Converts the raw PhysioNet data to structured data, create a complete subdata and artifically mask missing values.
   * MCMC_CHAIN.py: Implements the MCMC samplers pipeline.
   * SimpleMCMC.py: prepares the lag variables and implements two MCMC chain for convergence checks
   * Run_Single_MCMC.py: Implements MCMC inside the MICE iteration and check convergences
   * Comparison_runs.py: Generate multiple imputation under 30 experimental runs for each of the methods for comparison.
   * Run_experiments.py: Calls the entire functions for final experiments.
   * Visualisation.py: Generates the entire plots from this experiments and study.
   * BRITS.py: BRITS implementation using pypots.
   * packages.py: Lists all the packages utilised in this study.
  
3. **AirQuality_Plots**
     This folder contains all the plots generated from the experiments using the AirQuality dataset.
     
5. **PhysioNet_Plots**
     This folder contains all the plots generated from the experiments using the PhysioNet dataset.

6. **Requirements**
   The implementations relies on standard scientific Python libraries, including
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
