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
