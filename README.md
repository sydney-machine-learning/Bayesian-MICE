# Bayesian-MICE (tBayes-MICE)

> **Bayesian Multiple Imputation by Chained Equations for uncertainty-aware imputation of time-series data**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2603.27142)
[![HPC](https://img.shields.io/badge/HPC-UNSW%20Katana-orange)](https://doi.org/10.26190/669XA286)

---

## Overview

Missing data are pervasive in real-world time-series applications, particularly in environmental monitoring and healthcare, where reliable uncertainty quantification is essential. **tBayes-MICE** extends classical MICE by replacing deterministic regression updates with Bayesian regression models whose parameters and imputations are jointly sampled via Markov Chain Monte Carlo (MCMC).
The method support **Random Walk Metropolis (RWM)** sampling, with theoretically motivated optimal scaling to improve convergence and mixing.

---

## Repository Structure

```
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
```

### File descriptions

| File | Description |
|---|---|
| `Datasets/AirQualityUCI.csv` | Original AirQuality data (hourly, unprocessed) |
| `Datasets/Data_subset_AirQuality.csv` | AirQuality after removing original NaNs |
| `Datasets/Data_with_missing_AirQuality.csv` | AirQuality with artificial missing values for evaluation |
| `Datasets/physionet_5000patients.csv` | PhysioNet tabular data, filtered to rows with ≤60% missingness |
| `Datasets/physio_subdata.csv` | PhysioNet after removing all NaNs |
| `Datasets/physio_with_missing.csv` | PhysioNet with artificial missing values |
| `MCMC_MICE_codes/placeholder.py` | Missing value initialisation (mean-based and time-aware variants) |
| `MCMC_MICE_codes/PhysioData_Loader.py` | Converts raw PhysioNet data to structured format and applies missingness masks |
| `MCMC_MICE_codes/MCMC_CHAIN.py` | MCMC samplers — Random Walk Metropolis (RWM) |
| `MCMC_MICE_codes/SimpleMCMC.py` | Lagged predictor construction and parallel MCMC chains |
| `MCMC_MICE_codes/Run_Single_MCMC.py` | Runs MCMC within each MICE iteration and checks convergence |
| `MCMC_MICE_codes/Comparison_runs.py` | 30-run multiple imputation comparison across all methods |
| `MCMC_MICE_codes/Run_experiments.py` | Full experimental workflow manager |
| `MCMC_MICE_codes/Visualisation.py` | Generates all figures used in the paper |
| `MCMC_MICE_codes/BRITS.py` | BRITS baseline implementation using pypots |
| `MCMC_MICE_codes/packages.py` | Full list of packages used |

---

## Installation

**1. Clone the repository**

```bash
git clone https://github.com/sydney-machine-learning/Bayesian-MICE.git
cd Bayesian-MICE
```

**2. Create a virtual environment (recommended)**

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install numpy pandas scikit-learn matplotlib arviz scipy pypots
```

---

## Datasets

### AirQuality (UCI)

- **Source:** [UCI Machine Learning Repository — Air Quality Dataset](https://archive.ics.uci.edu/dataset/360/air+quality)
- **Description:** Hourly air quality measurements from an Italian city, March 2004 to February 2005
- **Variables:** CO, NO2, NOx, O3, temperature, humidity (15 columns)
- **Missing rate:** ~11% naturally occurring, plus artificially masked values for evaluation

### PhysioNet

- **Source:** [PhysioNet Challenge 2012](https://physionet.org/content/challenge-2012/1.0.0/)
- **Description:** ICU patient records, 48-hour time series of 37 clinical variables
- **Processing:** Filtered to 5000 patients; rows with more than 60% missingness removed
- **Missing rate:** Varies by variable (10–80%)

---

## Reproducing the Experiments

### Step 1 — Prepare the data

```bash
python MCMC_MICE_codes/PhysioData_Loader.py
```

Outputs:
- `Datasets/physio_subdata.csv`
- `Datasets/physio_with_missing.csv`

### Step 2 — Run comparison across methods

```bash
python MCMC_MICE_codes/Comparison_runs.py
```

Compares tBayes-MICE, MICE and BRITS over 30 runs.

### Step 3 — Run the full experiment

```bash
python MCMC_MICE_codes/Run_experiments.py
```

Runs all methods on both datasets over 30 independent runs.

### Step 4 — Generate plots

```bash
python MCMC_MICE_codes/Visualisation.py
```

Reproduces all figures from the paper. Output saved to `AirQuality_Plots/` and `PhysioNet_Plots/`.

---


---

## Reproducibility

| Item | Detail |
|---|---|
| Random seed | Fixed at `42` across all experiments |
| Number of runs | 30 independent runs per method |
| Hardware | UNSW Katana HPC cluster |
| HPC citation | [DOI: 10.26190/669XA286](https://doi.org/10.26190/669XA286) |
| Python version | 3.8+ |
| OS | Linux (Ubuntu 20.04) |

> Results may vary slightly on different hardware due to floating-point precision differences. Reported metrics are means over 30 runs to account for this variability.

---

## Results Summary

Performance on AirQuality dataset (MAE, lower is better):

| Method | MAE | RMSE |
|---|---|---|
| MICE (classical) | — | — |
| tBayes-MICE V1 | — | — |
| BRITS | — | — |

Full results with confidence intervals are reported in the paper.

---

## Citation

If you use this code or results in your work, please cite:

```bibtex
@article{ibenegbu2026tbayes,
  title={tBayes-MICE: A Bayesian Approach to Multiple Imputation for Time Series Data},
  author={Ibenegbu, Amuche and de Micheaux, Pierre Lafaye and Chandra, Rohitash},
  journal={arXiv preprint arXiv:2603.27142},
  year={2026}
  url     = {https://arxiv.org/abs/2603.27142}
}
```

---

## Issues and Contributions

- Found a bug? Open an [issue](https://github.com/sydney-machine-learning/Bayesian-MICE/issues)
- Want to contribute? Fork the repository and submit a pull request


*Experiments were run on the Katana High Performance Computing cluster, supported by Research Technology Services at UNSW Sydney.*
