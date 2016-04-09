import numpy as np
import pandas as pd

from scipy.stats import norm

import statsmodels.tsa.stattools as ts
import statsmodels.api as sm

import matplotlib.pyplot as plt
import math

def hurst(ts):
	"""
    Returns the Hurst Exponent of a time series vector ts
    """
	# Create a range of arbitrary lag values
	lags = range(2, 100)
	# Calculate the array of the variances of the lagged differences
	tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
	# Use a linear fit to estimate the Hurst Exponent
	poly = np.polyfit(np.log(lags), np.log(tau), 1)
	# Return the Hurst exponent from the polyfit output
	return poly[0]*2.0

def vratiotest(ts, k):
	"""
	Performs variance ratio test on ts
	Returns (h, pValue)
	h=1 means rejection of the random walk hypothesis ( prob. > 90 percent )

	The algorithm below is adapted from the matlab code below.
	http://www.mathworks.com/matlabcentral/fileexchange/8489-variance-ratio-test/content/varatio.m
    """
	# Calculate one perios return series parameters
	ts = np.log(ts)
	rt1 = np.subtract(ts[1:], ts[:-1])
	T = len(rt1)
	mu = np.mean(rt1)
	v = np.var(rt1)
	# k periods rate of return series
	M = []
	for j in np.arange(0, k):
		# Create lagged rows
		tmp = []
		for i in np.arange(0, T-k+1):
			tmp.append(rt1[i+j])
		M.append(tmp)
	rtk = np.sum(M, axis=0)
	# Vriance ratio statistic
	m = k*(T-k+1)*(1-k/T);
	VR = 1/m*np.sum(np.square(rtk-k*mu))/v;
	# Homoskedastic statistic
	Zk = np.sqrt(T)*(VR-1)*(1/np.sqrt(2*(2*k-1)*(k-1)/(3*k)));
	# Heteroskedastic statistic
	j = np.array([ i for i in np.arange(1, k) ])
	vec1 = np.square((2/k*(k-j)));
	rst = np.square((rt1-mu));
	aux = []
	for i in np.arange(0, k-1):
		aux.append(np.dot(rst[i+1:T], rst[1:T-i]))
	vec2 = aux/np.square(((T-1)*v));
	Zhk = (VR-1)*(1/np.sqrt(np.dot(vec1,vec2)))
	# Calculate p-Values
	p_Zk = norm.cdf([-np.absolute(Zk), np.absolute(Zk)], 0, 1)
	p_Zk = 1 - (p_Zk[1] - p_Zk[0])
	p_Zhk = norm.cdf([-np.absolute(Zhk), np.absolute(Zhk)], 0, 1)
	p_Zhk = 1 - (p_Zhk[1] - p_Zhk[0])

	if (p_Zhk > 0.05):
		return (0, p_Zhk)
	else:
		return (1, p_Zhk)

def halflife(ts):
	# Lag variables
	y = ts[1:]
	y_lag = ts[:-1]
	delta_y = y - y_lag
	# Create Parameters
	param0 = np.ones(len(y_lag))
	param1 = y_lag
	params = np.array( list(zip(param0,param1)) )
	# OLS
	mod = sm.OLS(delta_y, params).fit()
	halflife = -math.log(2)/mod.params[1]

	return halflife

def movingAvg(ts, lookback):
    weights = np.repeat(1.0, lookback)/lookback
    sma = np.convolve(ts, weights, 'valid')
    return sma

def strategy(ts):
	lookback = halflife(ts)
	sma = np.array(movingAvg(ts, lookback))
	sms = np.std(sma)

	mktVal = -(ts[lookback-1:]-sma)/sms
	pnl = mktVal[:-1] * (ts[1+lookback-1:]-ts[lookback-1:-1])/(ts[lookback-1:-1])

	return pnl

# Get Data
data = pd.read_csv('USDCAD.csv')
data.index = data['Date']
data = data['Rate']

# print(ts.adfuller(data.values[-500:]))
# h = hurst(data.values)
# (h, d) = vratiotest(data.values, 3))
# hl = halflife(data.values[1000:3000])

pnl = strategy(data.values[-2000:])
plt.plot(pnl)
plt.show()
