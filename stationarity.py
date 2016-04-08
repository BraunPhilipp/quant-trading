import numpy as np
import pandas as pd

import statsmodels.tsa.stattools as ts

import matplotlib.pyplot as plt
import math


from pandas.io.data import DataReader
import datetime

def hurst(ts):
	"""
    Returns the Hurst Exponent of a time series vector ts
    """
	# Create a range of arbitrary lag values
	lags = range(2, 100)
	# Calculate the array of the variances of the lagged differences
	tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
	# Use a linear fit to estimate the Hurst Exponent
	poly = polyfit(np.log(lags), np.log(tau), 1)
	# Return the Hurst exponent from the polyfit output
	return poly[0]*2.0

def vratiotest(ts):
    """
    Performs variance ratio test on ts
    Returns (h, pValue)
    h=1 means rejection of the random walk hypothesis ( prob. > 90 percent )
    """
    # Create a range of arbitrary lag values
    lags = range(2, 100)
    arr = [ np.var(np.subtract(ts[lag:], ts[:-lag])) / (lag * np.var(np.subtract(ts[1:], ts[:-1]))) for lag in lags ]

    print(arr)

# Get Data
data = pd.read_csv('USDCAD.csv')
data.index = data['Date']
data = data['Rate']

#print(ts.adfuller(data.values))
#print(hurst(data.values))

log_y = [np.log(val) for val in data.values]

print(vratiotest(log_y))
