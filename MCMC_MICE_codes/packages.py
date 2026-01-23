import pandas as pd
import arviz as az
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from scipy import stats
import pymannkendall as mk
from statsmodels.tsa.seasonal import STL
import warnings
from sklearn.metrics import mean_squared_error, mean_absolute_error
import logging
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
import statsmodels.api as sm
import time
from collections import defaultdict
import gc
import os
from glob import glob