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

entryZscore = 1
exitZscore = 1

# Calculate Beta
lookback = 20
mod = pd.ols(y=y, x=x, window_type='rolling', window=lookback)
data = data[lookback-1:]
betas = mod.beta
# Portfolio
yport = pd.DataFrame(data['y'] - (betas['x'] * data['x']))
# Moving average and standard deviation
moving_mean = pd.rolling_mean(yport, window=lookback)
moving_std = pd.rolling_std(yport, window=lookback)
zscore = (yport - moving_mean) / moving_std

long_entry = zscore < -entryZscore
long_exit = zscore >= -entryZscore
short_entry = zscore > entryZscore
short_exit = zscore <= entryZscore

numunits_long= np.zeros((len(yport),1))
numunits_long = pd.DataFrame(np.where(long_entry,1,0))

numunits_short= np.zeros((len(yport),1))
numunits_short = pd.DataFrame(np.where(short_entry,-1,0))

numunits = numunits_long + numunits_short

# compute the $ position for each asset
AA = pd.DataFrame(repmat(numunits,1,2))
BB = pd.DataFrame(-betas['x'])
BB['ones'] = np.ones((len(betas)))
position = np.multiply(np.multiply(AA, BB), data)

pnl = np.sum(np.multiply(position[:-1], np.divide(np.diff(data, axis = 0), data[:-1])),1)

# gross market value of portfolio
mrk_val = pd.DataFrame.sum(np.absolute(position), axis=1)
mrk_val = mrk_val[:-1]
# return is P&L divided by gross market value of portfolio
rtn = pnl / mrk_val
acum_rtn = pd.DataFrame(np.cumsum(rtn))
acum_rtn = acum_rtn.fillna(method='pad')
# compute performance statistics
sharpe = (np.sqrt(252)*np.mean(rtn)) / np.std(rtn)
APR = np.prod(1+rtn)**(252/len(rtn))-1

print('Price spread Sharpe: {:.4}'.format(sharpe))
print('Price Spread APR: {:.4%}'.format(APR))

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(acum_rtn)
ax.set_title('{}-{} Bollinger Band     Acum Return'.format('x', 'y'))
ax.set_xlabel('Data points')
ax.set_ylabel('acumm rtn')
ax.text(1000, 0, 'Sharpe: {:.4}'.format(sharpe))
ax.text(1000, -0.03, 'APR: {:.4%}'.format(APR))

plt.show()
