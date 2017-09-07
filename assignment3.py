import pandas as pd
import numpy as np

energy = pd.read_excel('Energy Indicators.xls', skiprows=17, skipfooter=(284-246)
                       ,parse_cols='C:F'
                       ,na_values=['...', '…']
                       ,names=['Country', 'Energy Supply', 'Energy Supply per Capita', '% Renewable'])\
           .fillna(np.nan)

energy['Energy Supply'] *= 1000000

long_names = {"Republic of Korea": "South Korea",
              "United States of America": "United States",
              "United Kingdom of Great Britain and Northern Ireland": "United Kingdom",
              "China, Hong Kong Special Administrative Region": "Hong Kong"}

def clean_cnames(s):
    res = s.split('(')[0].strip().rstrip('0123456789')
    if res in long_names:
        res = long_names[res]
    return res

energy['Country'] = energy['Country'].apply(clean_cnames)

# Load the wold bank data
GDP = pd.read_csv('world_bank.csv', skiprows=4)

long_names2 = {"Korea, Rep.": "South Korea", 
              "Iran, Islamic Rep.": "Iran",
              "Hong Kong SAR, China": "Hong Kong"}

def clean_cnames(s):
    res = s
    if res in long_names2:
        res = long_names2[res]
    return res

GDP['Country Name'] = GDP['Country Name'].apply(clean_cnames)

# Load the sciamgojr data

ScimEn = pd.read_excel('scimagojr-3.xlsx')

# join
mrg_i = pd.merge(pd.merge(energy, GDP, left_on='Country', right_on='Country Name', left_index=False),
                 ScimEn, on='Country')\
          .sort_values('Rank', ascending=True)\
          .set_index('Country')

mrg15 = mrg_i.head(15)

def answer_one():
    res1 = mrg15[['Rank', 'Documents', 'Citable documents', 'Citations', 'Self-citations', 'Citations per document', 'H index', 'Energy Supply', 'Energy Supply per Capita', '% Renewable', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015']]
    return res1

def answer_two():
    # wrong: 168, 162, 160
    mrg_o = pd.merge(pd.merge(energy,
                              GDP,
                              left_on='Country',
                              right_on='Country Name',
                              how='outer'),
                     ScimEn,
                     left_on='Country Name',
                     right_on='Country',
                     how='outer')

    return len(mrg_o) - len(mrg_i)

answer_two()


def answer_three():
    res3 = answer_one()
    gdp = res3[res3.columns[(res3.columns.get_loc('2006')):]]
    gdp['mean'] = gdp.mean(axis=1, skipna=True)
    
    return gdp.sort_values('mean', ascending=False)['mean']


def answer_four():
    top15 = answer_one()
    gdp = top15[top15.columns[(top15.columns.get_loc('2006')):]]

    tmp = pd.merge(gdp,
                   pd.DataFrame(answer_three()),
                   left_index=True,
                   right_index=True)\
            .sort_values('mean', ascending=False)
    
    return tmp.iloc[5][-2] - tmp.iloc[5][0]

def answer_five():
    Top15 = answer_one()
    return Top15['Energy Supply per Capita'].mean(axis=0)

def answer_six():
    Top15 = answer_one()\
            .sort_values('% Renewable', ascending=False)\
            .reset_index()
    return tuple(Top15.head(1)[['Country', '% Renewable']].iloc[0])

def answer_seven():
    Top15 = answer_one()[['Self-citations', 'Citations']].reset_index()
    Top15['ratio'] = Top15['Self-citations']/Top15['Citations']
    Top15.sort_values('ratio', ascending=False, inplace=True)
    return tuple(Top15.head(1)[['Country', 'ratio']].iloc[0])


def answer_eight():
    Top15 = answer_one()[['Energy Supply', 'Energy Supply per Capita']]
    Top15['pop'] = Top15['Energy Supply']/Top15['Energy Supply per Capita']
    Top15.sort_values('pop', ascending=False, inplace=True)
    Top15.reset_index(inplace=True)
    return Top15.iloc[2]['Country']

def answer_nine():
    Top15 = answer_one()
    Top15['PopEst'] = Top15['Energy Supply'] / Top15['Energy Supply per Capita']
    Top15['Citable docs per Capita'] = Top15['Citable documents'] / Top15['PopEst']
    return Top15.corr().loc['Energy Supply per Capita', 'Citable docs per Capita']

def answer_ten():
    Top15 = answer_one()
    Top15['HighRenew'] = 1*(Top15['% Renewable'] >= Top15['% Renewable'].median())
    return Top15.sort_values('Rank')['HighRenew']

def answer_eleven():
    ContinentDict  = {'China':'Asia', 
                  'United States':'North America', 
                  'Japan':'Asia', 
                  'United Kingdom':'Europe', 
                  'Russian Federation':'Europe', 
                  'Canada':'North America', 
                  'Germany':'Europe', 
                  'India':'Asia',
                  'France':'Europe', 
                  'South Korea':'Asia', 
                  'Italy':'Europe', 
                  'Spain':'Europe', 
                  'Iran':'Asia',
                  'Australia':'Australia', 
                  'Brazil':'South America'}
    Top15 = answer_one().reset_index()
    Top15['PopEst'] = Top15['Energy Supply'] / Top15['Energy Supply per Capita']  
    Top15['Continent'] = Top15['Country'].map(lambda c: ContinentDict[c])
    
    res = pd.DataFrame({'size': Top15.groupby('Continent')['Country'].count(),
                        'sum': Top15.groupby('Continent')['PopEst'].sum(),
                        'mean': Top15.groupby('Continent')['PopEst'].mean(),
                        'std': Top15.groupby('Continent')['PopEst'].std()},
                      columns = ['size', 'sum', 'mean', 'std'])
    return res

answer_eleven()


def answer_thirteen():
    Top15 = answer_one()
    def setcommas(s):
        ls = list(str(s))
        try:
            idx = s.index('.')
        except:
            idx = len(s)
        for i in range(idx-3,0,-3):
            ls.insert(i, ',')
        return ''.join(ls)

    PopEst = (Top15['Energy Supply'] / Top15['Energy Supply per Capita']).map(str).map(setcommas)
    return PopEst
answer_thirteen()
