import pandas as pd
import matplotlib.pyplot as plt

df = pd .read_csv('Veiligheidszorg__ker_240917213103.csv', encoding='latin-1', sep=',',
                  skiprows=1, skip_footer=1)

spen = df.iloc[0].loc[list(map(str, range(2002, 2015)))].copy()
spendf = pd.DataFrame(spen)
spendf.index.names=['Perioden']
spendf = spendf.rename(columns={0: 'uitgaven'}).reset_index()
spendf[['Perioden', 'uitgaven']] = spendf[['Perioden', 'uitgaven']].astype(int)

#1999-2007
df1 = pd.read_csv('Gereg.criminaliteit__240917222841.csv', encoding='latin-1', sep=';', skiprows=4)
df11 = df1.iloc[0:9][['Perioden', 'aantal']]
df11[['Perioden', 'aantal']] = df11[['Perioden', 'aantal']].astype(int)

# 2010-2016
df2 = pd.read_csv('Geregistreerde_crimi_240917213156.csv', encoding='latin-1', sep=';', skiprows=3)
df21 = df2.copy().iloc[1]
df22 = pd.DataFrame(df21.iloc[2:9])
df22.index.names=['Perioden']
df22 = df22.rename(columns={1: 'aantal'}).reset_index()
df22['Perioden'] = df22['Perioden'].apply(lambda x: x.split('*')[0])
df22[['Perioden', 'aantal']] = df22[['Perioden', 'aantal']].astype(int)

crimdf = df11.append(df22).reset_index()


fig, ax1 = plt.subplots()
ax1.plot(spendf['Perioden'], spendf['uitgaven'], 'b-o')
ax1.set_xlabel('year')
# Make the y-axis label, ticks and tick labels match the line color.
ax1.set_ylabel('Spending', color='b')
ax1.tick_params('y', colors='b')

ax2 = ax1.twinx()
ax2.plot(crimdf['Perioden'], crimdf['aantal'], 'r-o')
ax2.set_ylabel('Crimes', color='r')
ax2.tick_params('y', colors='r')

fig.tight_layout()
plt.show()
