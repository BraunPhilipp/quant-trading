import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import math

import pickle
from numpy.matlib import repmat

# Load Data
x = pd.DataFrame(pickle.load(open('uso.pickle', 'rb')))
x['Adj_Close'] = [float(i[0]) for i in x.values]
x.rename(columns={'Adj_Close':'x'}, inplace=True)
y = pd.DataFrame(pickle.load(open('gld.pickle', 'rb')))
y['Adj_Close'] = [float(i[0]) for i in y.values]
y.rename(columns={'Adj_Close':'y'}, inplace=True)

data = x
data['y'] = y.values

x = data['x']
y = data['y']

# Calculate Ratio
lookback = 20
ratio = np.divide(y, x)
# Removed to receive same test set as log and spread strategy
ratio = ratio[lookback-1:]
x = x[lookback-1:]
y = y[lookback-1:]
data = data[lookback-1:]

moving_mean = pd.rolling_mean(ratio, window=lookback)
moving_std = pd.rolling_std(ratio, window=lookback)

numunits = pd.DataFrame(-(ratio - moving_mean) / moving_std)

AA = repmat(numunits,1,2)
BB = repmat([-1,1],len(numunits),1)
position = np.multiply(data, np.multiply(AA,BB))
# P&L
pnl = np.sum((np.multiply(position[:-1], np.divide(np.diff(data,axis = 0), data[:-1]))),1)
# Gross market value of portfolio
mrk_val = pd.DataFrame.sum(abs(position), axis=1)
mrk_val = mrk_val[lookback-1:-1]
# Return is P&L divided by gross market value of portfolio
rtn = pnl / mrk_val
# Compute performance statistics
sharpe = (np.sqrt(252)*np.mean(rtn)) / np.std(rtn)
APR = np.prod(1+rtn)**(252/len(rtn))-1

print('Price spread Sharpe: {:.4}'.format(sharpe))
print('Price Spread APR: {:.4%}'.format(APR))

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(np.cumsum(rtn.values))
ax.set_title('{}-{} Price Ratio Acum Return'.format('x', 'y'))
ax.set_xlabel('Data points')
ax.set_ylabel('acumm rtn')
ax.text(1000, 0, 'Sharpe: {:.4}'.format(sharpe))
ax.text(1000, -0.06, 'APR: {:.4%}'.format(APR))

plt.show()
