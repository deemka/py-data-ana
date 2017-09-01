import pandas as pd

census_df = pd.read_csv('census.csv')


def answer_five():
    cdf50 = census_df[census_df['SUMLEV'] == 50].copy()
    cdf50 = cdf50.groupby(['STNAME']).size().reset_index(name='Count').sort_values('Count', ascending=False)
    return cdf50.head(1)['STNAME'].iloc[0]


answer_five()
