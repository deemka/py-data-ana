import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

df = None
try:
    df = pd.read_csv('data/C2A2_data/BinnedCsvs_d400/c4d108a4477ba0ef3f607b577681f1c6d85287a94d8132171a1a0429.csv')
except:
    df = pd.read_csv('c4d108a4477ba0ef3f607b577681f1c6d85287a94d8132171a1a0429.csv')

df = df.sort_values(['Date', 'Data_Value'])
df['Data_Value'] = df['Data_Value']*0.1

df_orig = df.copy()

df = df_orig[df_orig['Date'] <= '2014-12-31']
df15 = df_orig[(df_orig['Date'] >= '2015-01-01') & (df_orig['Date'] <= '2015-12-31')]

df['Date'].map(pd.to_datetime)
df['Date'].year = 2000
df['Date'] = df['Date'].map(lambda d: pd.to_datetime('2000-' + d.split('-')[1] + '-' + d.split('-')[2]))

df = df.groupby(['Date'])['Data_Value'].agg({'min': np.min, 'max': np.max}).reset_index()[['Date', 'min', 'max']]
df = df[df['Date'] != '2000-02-29']

df15 = df15.groupby(['Date'])['Data_Value'].agg({'min': np.min, 'max': np.max}).reset_index()[['Date', 'min', 'max']]

df = df.reset_index()
plt.close()
df['brokenmin'] = df15['min']
df['brokenmax'] = df15['max']

df['brokenmin'][df['brokenmin'] > df['min']] = 666
df['brokenmax'][df['brokenmax'] < df['max']] = 666

months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
dates = list(map(lambda d: str(months[d.month-1]) + ' ' + str(d.day), df['Date']))
linemax, = plt.plot(np.arange(len(dates)), df['max'], '-', linewidth=0.5, color='salmon', label='High')
linemin, = plt.plot(np.arange(len(dates)), df['min'], '-', linewidth=0.5, color='cornflowerblue', label='Low')
plt.plot(np.arange(len(dates)), df['brokenmax'], 'o', linewidth=0.5, color='salmon', label='2015')
plt.plot(np.arange(len(dates)), df['brokenmin'], 'o', linewidth=0.5, color='steelblue', label='2015')

# linemin.set_antialiased(False)
# linemax.set_antialiased(False)
plt.gca().fill_between(np.arange(len(dates)), df['max'], df['min'], alpha=.75, color='linen')
plt.xlabel('Date')
plt.ylabel('Temperature, $\mathrm{{}^{o}C}$')
plt.title('Temperature data for North Holland (the Netherlands) 2005-2014')
plt.legend(['High (daily maximum)', 'Low (daily minimum)', '2015 broke High', '2015 broke Low'], loc=0, bbox_to_anchor=(.8, 0.95))

# Remove all the ticks (both axes), and tick labels on the Y axis
plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='on', labelbottom='on')

# Remove frame
for spine in plt.gca().spines.values():
    spine.set_visible(False)

ax = plt.axes()
# Horizontal grid lines 
ax.yaxis.grid(linestyle='dotted', alpha=0.75)
ax.format_xdata = mdates.DateFormatter('%m-%d')
# ax.set_xticklabels(dates)

plt.axhline(0, linestyle='--', color='gray', alpha=0.75, lw=.7)
#plt.ylim((-25, 38))
plt.locator_params(axis='x', nticks=12)

ax.set_xticks(np.arange(1,25, 2)*15)
ax.set_xticklabels(months)


plt.ylim((-25, 40))
plt.show()
