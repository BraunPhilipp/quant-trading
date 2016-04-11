import numpy as np
import pandas as pd

from scipy.stats import norm

import statsmodels.tsa.stattools as ts
import statsmodels.api as sm
from johansen import coint_johansen

import matplotlib.pyplot as plt
import math

from yahoo_finance import Share

import pickle

from numpy.matlib import repmat

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

def vratio(ts, lag = 2, cor = 'hom'):
    n = len(ts)
    mu  = sum(ts[1:n]-ts[:n-1])/n;
    m=(n-lag+1)*(1-lag/n);
    #print( mu, m, lag)
    b=sum(square(ts[1:n]-ts[:n-1]-mu))/(n-1)
    t=sum(square(ts[lag:n]-ts[:n-lag]-lag*mu))/m
    vratio = t/(lag*b);

    la = float(lag)

    if cor == 'hom':
        varvrt=2*(2*la-1)*(la-1)/(3*la*n)

    elif cor == 'het':
        varvrt=0;
        sum2=sum(square(a[1:n]-a[:n-1]-mu));
        for j in range(lag-1):
            sum1a=square(a[j+1:n]-a[j:n-1]-mu);
            sum1b=square(a[1:n-j]-a[0:n-j-1]-mu)
            sum1=dot(sum1a,sum1b);
            delta=sum1/(sum2**2);
            varvrt=varvrt+((2*(la-j)/la)**2)*delta

    zscore = (vratio - 1) / sqrt(float(varvrt))
    pval = normcdf(zscore);

    return  vratio, zscore, pval

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

	# delta_ts = np.diff(ts)
	# lag_ts = np.transpose(np.vstack([ts[1:], np.ones(len(ts[1:]))]))
	# mod = sm.OLS(lag_ts, delta_ts).fit()
	# half_life = math.log(2) / mod.params[0]

	return halflife

def strategy(ts):
	lookback = math.ceil(halflife(ts))
	sma = pd.rolling_mean(ts, window=lookback)
	sms = pd.rolling_std(ts, window=lookback)

	mktVal = -(ts[lookback-1:]-sma)/sms
	pnl = mktVal[:-1] * (ts[1+lookback-1:]-ts[lookback-1:-1])/(ts[lookback-1:-1])
	pnl = np.cumprod(pnl+1)
	pnl = pnl - 1
	return pnl

def cadf(x, y):
	param0 = np.ones(len(x))
	param1 = x
	params = np.array( list(zip(param0,param1)) )
	# OLS
	mod = sm.OLS(y, params).fit()
	# Stationarity
	st = y - mod.params[1] * param1
	print(ts.adfuller(st))

def strategy2(x, y):
	y = pd.DataFrame({'col1': x, 'col2': y})
	results = coint_johansen(y, 0, 1)
	# Take first egienvector strongest relationship
	w = results.evec[:, 0]
	yport = pd.DataFrame.sum(w*y, axis=1).values
	lookback = int(halflife(yport))

	moving_mean = pd.rolling_mean(yport, window=lookback)
	moving_std = pd.rolling_std(yport, window=lookback)
	# Number of units in unit portfolio equal to negative z-score (unit portfolio)
	z_score = (yport - moving_mean) / moving_std
	numunits = pd.DataFrame(z_score * -1, columns=['numunits'])

	# Calculate P&L
	AA = repmat(numunits,1,2)
	BB = np.multiply(repmat(w,len(y),1), y)
	position = pd.DataFrame(np.multiply(AA, BB))

	pnl = np.sum(np.divide(np.multiply(position[:-1],np.diff(y,axis = 0)), y[:-1]),1)
	# gross market value of portfolio
	mrk_val = pd.DataFrame.sum(np.absolute(position), axis=1)
	# return is P&L divided by gross market value of portfolio
	rtn = np.cumsum(pd.DataFrame(pnl/mrk_val, columns=['rtn']))

	plt.plot(rtn)
	plt.show()

def data():
	# Get Data
	ewa = Share('USO')
	x = pd.DataFrame(ewa.get_historical('2007-01-01', '2016-01-01'))
	x.index = x['Date']
	x = x['Adj_Close']

	ewc = Share('GLD')
	y = pd.DataFrame(ewc.get_historical('2007-01-01', '2016-01-01'))
	y.index = y['Date']
	y = y['Adj_Close']

	with open('uso.pickle','wb') as f:
	    pickle.dump(x, f)

	with open('gld.pickle','wb') as f:
		pickle.dump(y, f)

	# Plot
	plt.plot(x,y,'bo')
	plt.plot(x,y_hat,'-r')
	plt.xlabel('EWA share price')
	plt.ylabel('EWC share price')
	plt.show()

	# # Load Data
	# x = np.array(list(map(float, pickle.load(open('ewa.pickle', 'rb')))))
	# y = np.array(list(map(float, pickle.load(open('ewc.pickle', 'rb')))))

# # Get Data
# data = pd.read_csv('USDCAD.csv')
# data.index = data['Date']
# data = data['Rate']
#
# # Stationarity Tests
# print(ts.adfuller(data.values[-500:]))
# h = hurst(data.values)
# (h, d) = vratiotest(data.values, 3))
# hl = halflife(data.values[1000:3000])
#
# # Strategy
# pnl = strategy(data.values[-2000:])
# plt.plot(pnl)
# plt.show()

# data()
