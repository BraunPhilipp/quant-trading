import pandas as pd
import numpy as np

from numpy.matlib import repmat
import matplotlib.pyplot as plt
import math

from johansen import coint_johansen

from yahoo_finance import Share

from functions import *

import pickle

start = '2007-01-01'
end = '2016-01-01'

# XOP IEO PXE
x_ticket = 'XOP'
y_ticket = 'IEO'
z_ticket = 'PXE'

entryZscore = 0.42
exitZscore = 0

# # Get Data
# x = Share(x_ticket)
# x = pd.DataFrame(x.get_historical(start, end))['Adj_Close']
# x = list(map(float, x.values))
#
# with open('xop.pickle','wb') as f:
#     pickle.dump(x, f)
#
# y = Share(y_ticket)
# y = pd.DataFrame(y.get_historical(start, end))['Adj_Close']
# y = list(map(float, y.values))
#
# with open('ieo.pickle','wb') as f:
#     pickle.dump(y, f)
#
# z = Share(z_ticket)
# z = pd.DataFrame(z.get_historical(start, end))['Adj_Close']
# z = list(map(float, z.values))
#
# with open('pxe.pickle','wb') as f:
#     pickle.dump(z, f)

x = list(map(float, pickle.load(open('xop.pickle', 'rb'))))
y = list(map(float, pickle.load(open('ieo.pickle', 'rb'))))
z = list(map(float, pickle.load(open('pxe.pickle', 'rb'))))

y = pd.DataFrame({'x': x, 'y': y, 'z': z})
results = coint_johansen(y, 0, 1)
# Take first egienvector strongest relationship
w = results.evec[:, 0]
yport = pd.DataFrame.sum(w*y, axis=1).values
#lookback = int(halflife(yport))
lookback = 20

moving_mean = pd.rolling_mean(yport, window=lookback)
moving_std = pd.rolling_std(yport, window=lookback)

# Number of units in unit portfolio equal to negative z-score (unit portfolio)
zscore = (yport - moving_mean) / moving_std

#numunits = pd.DataFrame(z_score * -1, columns=['numunits'])

long_entry = zscore < -entryZscore
long_exit = zscore >= -entryZscore
short_entry = zscore > entryZscore
short_exit = zscore <= entryZscore

numunits_long= np.zeros((len(yport),1))
numunits_long = pd.DataFrame(np.where(long_entry,1,0))

numunits_short= np.zeros((len(yport),1))
numunits_short = pd.DataFrame(np.where(short_entry,-1,0))

numunits = numunits_long + numunits_short

# Calculate P&L
AA = repmat(numunits,1,3)
BB = np.multiply(repmat(w,len(y),1), y)
position = pd.DataFrame(np.multiply(AA, BB))

pnl = np.sum(np.divide(np.multiply(position[:-1],np.diff(y,axis = 0)), y[:-1]),1)
# gross market value of portfolio
mrk_val = pd.DataFrame.sum(np.absolute(position), axis=1)
mrk_val = mrk_val[:-1]
# return is P&L divided by gross market value of portfolio
rtn = pnl / mrk_val

# compute performance statistics
sharpe = (np.sqrt(252)*np.mean(rtn)) / np.std(rtn)
APR = np.prod(1+rtn)**(252/len(rtn))-1

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(np.cumsum(rtn))
ax.set_title('{}-{}-{} Price Spread Acum Return'.format(x_ticket, y_ticket, z_ticket))
ax.set_xlabel('Data points')
ax.set_ylabel('acumm rtn')
ax.text(1000, 0, 'Sharpe: {:.4}'.format(sharpe))
ax.text(1000, -0.03, 'APR: {:.4%}'.format(APR))

plt.show()
plt.close()
