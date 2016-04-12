import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from yahoo_finance import Share



start = '2008-01-01'
end = '2010-01-01'

etf = ['GLD', 'GDX', 'USO']

for e in etf:
	# Asset Selection
	asset = pd.DataFrame(Share(e).get_historical(start, end))
	asset = pd.DataFrame({'price': list(map(float, asset['Adj_Close'].values))}, index=asset['Date'].values)
	asset = ( 1 + asset.pct_change()[1:] ).cumprod()
	plt.plot(asset.values, label=e)

plt.show()
plt.close('all')
