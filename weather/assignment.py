import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# df = pd.read_csv('data/C2A2_data/BinnedCsvs_d400/c4d108a4477ba0ef3f607b577681f1c6d85287a94d8132171a1a0429.csv')
df = pd.read_csv('c4d108a4477ba0ef3f607b577681f1c6d85287a94d8132171a1a0429.csv')
df = df.sort_values(['Date', 'Data_Value'])

df['Date'] = df['Date'].map(pd.Timestamp)
df['Data_Value'] = df['Data_Value']*0.1
df = df[df['Date'] <= '2014-12-31']
df = df.groupby(['Date'])['Data_Value'].agg({'min': np.min, 'max': np.max}).reset_index()[['Date', 'min', 'max']]

linemin, = plt.plot_date(df['Date'], df['min'], '-', linewidth=0.5, color='cornflowerblue', label='Low')
linemax, = plt.plot_date(df['Date'], df['max'], '-', linewidth=0.5, color='salmon', label='High')

# linemin.set_antialiased(False)
# linemax.set_antialiased(False)
plt.fill_between(df['Date'].values, df['min'], df['max'], alpha=.75, color='moccasin')
plt.xlabel('Date')
# plt.ylim((-35, 35))
plt.ylabel('Temperature, Celsius')
plt.title('Temperature data for North Holland (the Netherlands) 2005-2014')
plt.legend()
ax = plt.axes()
ax.yaxis.grid(linestyle='..', alpha=0.75)

plt.show()
