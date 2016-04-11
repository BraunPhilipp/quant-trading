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

y = np.asarray(data['y'])
x = pd.DataFrame(data['x'])
index = x.index

# Kalman Filter Implementation
x['ones'] = np.ones(len(x))
x = np.asarray(x)

delta = 0.0001

yhat = np.zeros(shape=(len(x),1))     # measurement prediction
e = np.zeros(shape=(len(x),1))        # measurement prediction error
Q = np.zeros(shape=(len(x),1))        # measurement prediction error variance

# For clarity, we denote R(t|t) by P(t).
# initialize R, P and beta.
R = np.zeros((2,2));
P = np.zeros((2,2));
beta = np.zeros((2, len(x)))
Vw = delta/(1-delta)*np.eye(2)
Ve = 0.001

# Given initial beta and R (and P)
for t in range(len(y)):
    if t > 0:
        beta[:, t] = beta[:, t-1]                   # state prediction. Equation 3.7
        R = P + Vw                                   # state covariance prediction. Equation 3.8
        yhat[t] = np.dot(x[t, :], beta[:, t])           # measurement prediction. Equation 3.9
        Q[t] = np.dot(np.dot(x[t, :], R), np.transpose(x[t, :])) + Ve   # measurement variance prediction. Equation 3.10
        # Observe y(t)
        e[t] = y[t] - yhat[t]                       # measurement prediction error
        K = np.dot(R, np.transpose(x[t, :])) / Q[t]                 # Kalman gain
        beta[:, t] = beta[:, t] + K * e[t]          # State update. Equation 3.11
        P = R - K * x[t, :] * R                      # State covariance update. Euqation 3.12

sqrt_Q = np.sqrt(Q)
beta = pd.DataFrame(np.transpose(beta), index= index, columns=('x', 'intercept'))
e = pd.DataFrame(e, index=index)

long_entry = e < -sqrt_Q   # a long position means we should buy EWC
long_exit = e > -sqrt_Q
short_entry = e > sqrt_Q
short_exit =  e < sqrt_Q

numunits_long= np.zeros((len(data),1))
numunits_long = pd.DataFrame(np.where(long_entry,1,0))
numunits_short= np.zeros((len(data),1))
numunits_short = pd.DataFrame(np.where(short_entry,-1,0))
numunits = numunits_long + numunits_short

# compute the $ position for each asset
AA = pd.DataFrame(repmat(numunits,1,2))
BB = pd.DataFrame(-beta['x'])
BB['ones'] = np.ones((len(beta)))
position = np.multiply(np.multiply(AA, BB), data)

# compute the daily pnl in $$
pnl = np.sum(np.multiply(position[:-1], np.divide(np.diff(data,axis = 0), data[:-1])),1)
# gross market value of portfolio
mrk_val = pd.DataFrame.sum(np.absolute(position), axis=1)
mrk_val = mrk_val[:-1]
# return is P&L divided by gross market value of portfolio
rtn = pnl / mrk_val

# cumulative return and smoothing of series for plotting
acum_rtn = pd.DataFrame(np.cumsum(rtn))
acum_rtn = acum_rtn.fillna(method='pad')
# compute performance statistics
sharpe = (np.sqrt(252)*np.mean(rtn)) / np.std(rtn)
APR = np.prod(1+rtn)**(252/len(rtn))-1

print('Sharpe: {:.4}'.format(sharpe))
print('APR: {:.4%}'.format(APR))

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(acum_rtn)
ax.set_title('{}-{} Price Ratio CumRet\n(Using Kalman filter)'.format('x', 'y'))
ax.set_xlabel('Data points')
ax.set_ylabel('cum rtn')
ax.text(1200, 0.2, 'Sharpe: {:.4}'.format(sharpe))
ax.text(1200, 0.15, 'APR: {:.4%}'.format(APR))

plt.show()
