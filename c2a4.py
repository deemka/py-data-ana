import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


df = pd .read_csv('Veiligheidszorg__ker_240917213103.csv', encoding='latin-1', sep=',',
                  skiprows=1, skip_footer=1)

spen = df.iloc[0].loc[list(map(str, range(2002, 2015)))].copy()
spendf = pd.DataFrame(spen)
spendf.index.names=['Perioden']
spendf = spendf.rename(columns={0: 'uitgaven'}).reset_index()
spendf[['Perioden', 'uitgaven']] = spendf[['Perioden', 'uitgaven']].astype(int)

#1999-2007
df1 = pd.read_csv('Gereg.criminaliteit__240917222841.csv', encoding='latin-1', sep=';', skiprows=4)
df11 = df1.iloc[0:9][['Perioden', 'aantal', 'aantal.1']]
df11[['Perioden', 'aantal', 'aantal.1']] = df11[['Perioden', 'aantal', 'aantal.1']].astype(int)
df11 = df11.rename(columns={'aantal.1':'opgehelderd'}).reset_index()

# 2010-2016
df2 = pd.read_csv('Geregistreerde_crimi_240917213156.csv', encoding='latin-1', sep=';', skiprows=3)
df21 = df2.copy().iloc[1]
df22 = pd.DataFrame(df21.iloc[2:9])
df22.index.names=['Perioden']
df22 = df22.rename(columns={1: 'aantal'}).reset_index()
df22['Perioden'] = df22['Perioden'].apply(lambda x: x.split('*')[0])
df22[['Perioden', 'aantal']] = df22[['Perioden', 'aantal']].astype(int)
df22['opgehelderd'] = pd.DataFrame(df21.iloc[9:16]).reset_index().loc[:, 1]

crimdf = pd.concat([df11, df22])


fig, ax1 = plt.subplots()

crimdf[['Perioden', 'aantal', 'opgehelderd']] = crimdf[['Perioden', 'aantal', 'opgehelderd']].astype(int)
crimdf.reset_index(inplace=True)

# crimdf = crimdf[crimdf['Perioden'] >= 2002].reset_index()
ax1.plot(spendf['Perioden'], spendf['uitgaven']/spendf['uitgaven'][0], label='Expenditures')
ax1.plot(crimdf['Perioden'], crimdf['aantal']/crimdf['aantal'][3], label='Tot. crimes')
ax1.plot(crimdf['Perioden'], crimdf['opgehelderd']/crimdf['opgehelderd'][3], label='Solved crimes')

# plt.legend(['Expenditures on public order', 'Registered crimes', 'Solved crime cases'])
plt.title('Expenditures on public order, number of registered crimes,\n and resolved cases relatively to the year 2002')

for spine in plt.gca().spines.values():
    spine.set_visible(False)

plt.axhline(.8, linewidth=1, alpha=.15, color='gray')
plt.axhline(1, linewidth=1, alpha=.15, color='gray')
plt.axhline(1.2, linewidth=1, alpha=.15, color='gray')
plt.axhline(1.4, linewidth=1, alpha=.15, color='gray')

plt.xlabel('Year')
plt.ylabel('Relative change to 2002')
plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='on', labelbottom='on')


ax1.set_xticks(np.arange(2000, 2017, 2))

ax1.text(2008, 1.38, ax1.lines[0].get_label(), 
         horizontalalignment='left',
         verticalalignment='center',
         fontsize=9, color=ax1.lines[0].get_color(), alpha=0.75)
ax1.text(2008, 0.82, ax1.lines[1].get_label(), 
         horizontalalignment='left',
         verticalalignment='center',
         fontsize=9, color=ax1.lines[1].get_color(), alpha=0.75)
ax1.text(2008, 1.11, ax1.lines[2].get_label(), 
         horizontalalignment='left',
         verticalalignment='center',
         fontsize=9, color=ax1.lines[2].get_color(), alpha=0.75)

plt.show()
