import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import math

# Get Data
data = pd.read_csv('USDCAD.csv')
data.index = data['Date']
data = data['Rate']

lag = 1

y = data[lag:]
y_lag = data[:-lag]

delta_y = y.values - y_lag.values

# Create Parameters
param0 = np.ones(len(y_lag.values))
param1 = y_lag.values
params = np.array( list(zip(param0,param1)) )

# OLS
mod = sm.OLS(delta_y, params).fit()
halflife = -math.log(2)/mod.params[1]

print(halflife)
