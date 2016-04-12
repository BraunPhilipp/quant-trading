import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import math

import pickle
from numpy.matlib import repmat

from yahoo_finance import Share

start = '2007-01-01'
end = '2016-01-01'

x_ticket = 'GLD'
y_ticket = 'GDX'

# Get Data
x = Share(x_ticket)
x = pd.DataFrame(x.get_historical(start, end))['Adj_Close']
x = list(map(float, x.values))

y = Share(y_ticket)
y = pd.DataFrame(y.get_historical(start, end))['Adj_Close']
y = list(map(float, y.values))

data = pd.DataFrame({'x':x, 'y':y})

x = data['x']
y = data['y']

# Calculate Beta
lookback = 20
mod = pd.ols(y=np.log(y), x=np.log(x), window_type='rolling', window=lookback)
data = data[lookback-1:]
betas = mod.beta
# Portfolio
yport = pd.DataFrame(np.log(data['y']) - (betas['x'] * np.log(data['x'])))
# Moving average and standard deviation
moving_mean = pd.rolling_mean(yport, window=lookback)
moving_std = pd.rolling_std(yport, window=lookback)
# Negative z score
numunits = pd.DataFrame(-(yport - moving_mean) / moving_std)
# Double number of col in DataFrame
AA = pd.DataFrame(repmat(numunits,1,2))
BB = pd.DataFrame(-betas['x'])
BB['ones'] = np.ones((len(betas)))
# Amount of positions we hold in each asset
position = np.multiply(np.multiply(AA, BB), data)
# P&L in $$
pnl = np.sum(np.multiply(position[:-1], np.divide(np.diff(data, axis=0), data[:-1])),1)
# Market Value
mrk_val = pd.DataFrame.sum(abs(position), axis=1)
mrk_val = mrk_val[:-1]
rtn = pnl / mrk_val

# compute performance statistics
sharpe = (np.sqrt(252)*np.mean(rtn)) / np.std(rtn)
APR = np.prod(1+rtn)**(252/len(rtn))-1

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(np.cumsum(rtn))
ax.set_title('{}-{} Price Spread Acum Return'.format('x', 'y'))
ax.set_xlabel('Data points')
ax.set_ylabel('acumm rtn')
ax.text(1000, 0, 'Sharpe: {:.4}'.format(sharpe))
ax.text(1000, -0.03, 'APR: {:.4%}'.format(APR))

plt.show()
