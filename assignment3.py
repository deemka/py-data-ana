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

def answer_one():
    mrg15 = mrg_i.head(15)
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
