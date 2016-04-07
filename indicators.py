
"""
Indicator Toolset
=================

filler(s, N)
norm(s)
absol(s)
sma(s, N)
ema(s, N)
rsi(s, N)
macd(s, A, B)
adx(h, l, N)
stoch(c, N)

"""

def filler(s, N):
    """
    Used to fill missing data of rolling mean etc. in lists
    """
    n = [None] * N
    return n + s

def absol(s):
    """
    Returns an absolute list of values
    """
    a = []
    for i in s:
        if i < 0 and i != None:
            a.append(-i)
        else:
            a.append(i)

    return a

def norm(s):
    """
    Norms any list to values between -1 and 1
    """
    n = []

    for i in s:
        if i != None:
            n.append(i / max(absol(s)))
        else:
            n.append(None)

    return n

def sma(s, N):
    """
    Returns Standard Moving Average (SMA) for time series s.
    """
    sma = []

    for i in range(N,len(s)):
        sma.append(sum(s[i-N:i]) / N)

    return sma

def ema(s, N):
    """
    Returns an N period Exponential Moving Average (EMA) for time series s.
    s is a list ordered from oldest (index 0) to most recent (index -1)
    N is an integer
    """
    # define variables
    ema = []
    j = 1

    # get n sma first and calculate the next n period ema

    # calculate average of first n period
    sma = sum(s[:N]) / N
    k = 2 / float(1 + N)
    # first period does not contain any multiplier
    ema.append(sma)

    # EMA(current) = ( (Price(current) - EMA(prev) ) x k) + EMA(prev)
    ema.append(( (s[N] - sma) * k) + sma)

    # now calculate the rest of the values
    for i in s[N+1:]:
        tmp = ( (i - ema[j]) * k) + ema[j]
        j = j + 1
        ema.append(tmp)

    return ema

def stoch(c, N):
    """
    STOCHASTICS
    %K = 100[(C - L14)/(H14 - L14)]
    ## altered stochastics osciallator uses ema and not extremest breakout points (high, low)
    """
    stoch = []
    for i in range(0,len(c[N:])):
        stoch.append(-100 * ((c[N] - min(c[i:N+i])) / (max(c[i:i+N]) - min(c[i:i+N]))))

    stoch = filler(norm(ema(stoch, N)), N)

    return stoch

def rsi(s, N):
    """
    Returns Relative Strength Index (RSI) using EMA.
    N typically has a value of 14 but maybe varied.
    """
    # calculating ups and downs
    shift = s[1:]
    real = s[:-1]

    diff = real - shift

    ups = []
    downs = []
    for i in diff:
        if i > 0:
            ups.append(i)
            downs.append(0)
        else:
            ups.append(0)
            downs.append(-i)

    ema_ups = ema(ups, N)
    ema_downs = ema(downs, N)

    # actual RSI calculation
    rs = [float(a) / float(b) for a, b in zip(ema_ups, ema_downs)]
    rsi = [100 - (100 / (1 + float(a))) for a in rs]

    return rsi

def macd(s, A, B):
    '''
    Used for convergence and divergence testing this indicator calculates the difference between 2 EMAs.
    A and B define periods of both EMA indicators, s defines the list() of data provided for the formula.
    Standard values are for A = 26 and B = 12. MACD maybe compared to a 9 day EMA.
    '''
    ema_a = filler(ema(s, A), B)
    ema_b = filler(ema(s, B), B)

    # cannot subtract nan
    div = []

    for i in range(0,len(s)):
        if ema_a[i] and ema_b[i] != None:
            div.append(ema_a[i] - ema_b[i])
        else:
            div.append(None)

    return div



def adx(h, l, c, N):
    """
    Returns Average Directional Index (ADX). High ADX indicates a trend formation.
    h and l are HIGH and LOW stock data. N defines the time series.
    """

    ##### TRUE RANGE MISSSING !!!!! WRONG CALCUALTIONS!!!!!

    h_diff = h[1:] - h[:-1]
    l_diff = l[:-1] - l[1:]

    # True Range Calculation

    tr_range = []

    for i in range(1, len(c)):
        tr_range.append(max([(h[i] - l[i]), abs(h[i]-c[i-1]), abs(l[i]-c[i-1]), 1]))

    # positive directional movement
    dmp = []

    for i in range(len(h_diff)):
        if h_diff[i] > 0.0 and h_diff[i] > l_diff[i]:
            dmp.append(i)
        else:
            dmp.append(0.0)

    # negative directional movement
    dmn = []

    for i in range(len(l_diff)):
        if l_diff[i] > 0.0 and l_diff[i] > h_diff[i]:
            dmn.append(i)
        else:
            dmn.append(0.0)

    # directional movement
    #dmp = absol(dmp)
    #dmn = absol(dmn)

    for i in range(len(dmp)):
        dmp[i] = 100 * dmp[i] / tr_range[i]

    for i in range(len(dmn)):
        dmn[i] = 100 * dmn[i] / tr_range[i]

    dmp = sma(dmp, N)
    dmn = sma(dmn, N)

    dm = []

    for i in range(0, len(dmp)):
        if dmp[i] + dmn[i] != 0:
            dm.append(100 * abs(dmp[i] - dmn[i]) / abs(dmp[i] + dmn[i]))
        else:
            dm.append(0)

    # smoothing and filling
    return filler(sma(dm, N), 2*N+1)
