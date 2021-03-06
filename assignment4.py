import pandas as pd
states = {'OH': 'Ohio', 'KY': 'Kentucky', 'AS': 'American Samoa', 'NV': 'Nevada', 'WY': 'Wyoming',
          'NA': 'National', 'AL': 'Alabama', 'MD': 'Maryland', 'AK': 'Alaska', 'UT': 'Utah', 'OR': 'Oregon',
          'MT': 'Montana', 'IL': 'Illinois', 'TN': 'Tennessee', 'DC': 'District of Columbia', 'VT': 'Vermont',
          'ID': 'Idaho', 'AR': 'Arkansas', 'ME': 'Maine', 'WA': 'Washington', 'HI': 'Hawaii', 'WI':
          'Wisconsin', 'MI': 'Michigan', 'IN': 'Indiana', 'NJ': 'New Jersey', 'AZ': 'Arizona', 'GU': 'Guam',
          'MS': 'Mississippi', 'PR': 'Puerto Rico', 'NC': 'North Carolina', 'TX': 'Texas', 'SD': 'South Dakota',
          'MP': 'Northern Mariana Islands', 'IA': 'Iowa', 'MO': 'Missouri', 'CT': 'Connecticut',
          'WV': 'West Virginia', 'SC': 'South Carolina', 'LA': 'Louisiana', 'KS': 'Kansas', 'NY': 'New York',
          'NE': 'Nebraska', 'OK': 'Oklahoma', 'FL': 'Florida', 'CA': 'California', 'CO': 'Colorado', 'PA':
          'Pennsylvania', 'DE': 'Delaware', 'NM': 'New Mexico', 'RI': 'Rhode Island', 'MN': 'Minnesota', 'VI':
          'Virgin Islands', 'NH': 'New Hampshire', 'MA': 'Massachusetts', 'GA': 'Georgia',
          'ND': 'North Dakota', 'VA': 'Virginia'}


def get_list_of_university_towns():
    import re
    utowns = pd.DataFrame(pd.read_csv('university_towns.txt', squeeze=True, header=None,
                                      names=['RegionName'], sep='!@#$$', engine='python'))

    def st(s):
        if '[edit]' in s:
            st.state = str(re.sub('\[.*$', '', s).strip())
        return st.state

    utowns['State'] = utowns['RegionName'].map(st)
    utowns['RegionName'] = utowns['RegionName'].map(lambda s: re.sub('[\(\[].*$', '', s)).map(str.strip)
    utowns = utowns[['State', 'RegionName']]
    utowns = utowns[utowns['RegionName'] != utowns['State']]
    return(utowns)


def get_recession_start():
    '''Returns the year and quarter of the recession start time as a
    string value in a format such as 2005q3'''

    gdp = pd.read_excel('gdplev.xls', skiprows=219,
                        names=['Quarter', 'GDP'],
                        parse_cols='E,G')
    return gdp[(gdp['GDP'].diff() < 0).shift(-1).rolling(2).sum() == 2].iloc[0].Quarter


def get_recession_end():
    '''Returns the year and quarter of the recession end time as a
    string value in a format such as 2005q3'''
    gdp = pd.read_excel('gdplev.xls', skiprows=219,
                        names=['Quarter', 'GDP'],
                        parse_cols='E,G')
    gdp = gdp[gdp['Quarter'] > get_recession_start()]
    return gdp[(gdp['GDP'].diff() > 0).rolling(2).sum() == 2].iloc[0].Quarter


def get_recession_bottom():
    '''Returns the year and quarter of the recession bottom time as a
    string value in a format such as 2005q3'''
    gdp = pd.read_excel('gdplev.xls', skiprows=219,
                        names=['Quarter', 'GDP'],
                        parse_cols='E,G')

    gdp = gdp[(gdp['Quarter'] > get_recession_start()) & (gdp['Quarter'] < get_recession_end())]
    return gdp['Quarter'].ix[gdp['GDP'].idxmin()]


def convert_housing_data_to_quarters():
    '''Converts the housing data to quarters and returns it as mean
    values in a dataframe. This dataframe should be a dataframe with
    columns for 2000q1 through 2016q3, and should have a multi-index
    in the shape of ["State","RegionName"].
    Note: Quarters are defined in the assignment description, they are
    not arbitrary three month periods.

    The resulting dataframe should have 67 columns, and 10,730 rows.
    '''
    from numpy import mean

    hp = pd.read_csv('City_Zhvi_AllHomes.csv')
    hp['State'] = hp['State'].map(lambda s: states[s])
    hp = hp[['State','RegionName'] + list(filter(lambda c: '20' in c, hp.columns.values.tolist() ))]
    
    def to_quarter(s):
        from math import floor
        return 'q'.join((s.split('-')[0], str(1 + int((int(s.split('-')[1]) - 1) / 3))))

    hp = hp.set_index(['State', 'RegionName']).groupby(to_quarter, axis=1).agg(mean)
    return hp


def run_ttest():
    '''First creates new data showing the decline or growth of housing prices
    between the recession start and the recession bottom. Then runs a ttest
    comparing the university town values to the non-university towns values, 
    return whether the alternative hypothesis (that the two groups are the same)
    is true or not as well as the p-value of the confidence. 
    
    Return the tuple (different, p, better) where different=True if the t-test is
    True at a p<0.01 (we reject the null hypothesis), or different=False if 
    otherwise (we cannot reject the null hypothesis). The variable p should
    be equal to the exact p value returned from scipy.stats.ttest_ind(). The
    value for better should be either "university town" or "non-university town"
    depending on which has a lower mean price ratio (which is equivilent to a
    reduced market loss).'''

    hd = convert_housing_data_to_quarters()\
         .reset_index()[['RegionName', get_recession_start(), get_recession_bottom()]]

    dat = pd.merge(hd, get_list_of_university_towns(), on ='RegionName',  how='left')\
            .rename(columns={'State': 'UniTown'})
    dat['UniTown'].fillna(False, inplace=True)
    dat['UniTown'] = dat['UniTown'].map(lambda s: s and True)
    dat = dat.drop_duplicates()
    dat['delta'] = dat[get_recession_bottom()] - dat[get_recession_start()]
    
    from scipy.stats import ttest_ind

    ttest_res = ttest_ind(dat[dat['UniTown'] == True]['delta'], dat[dat['UniTown'] == False]['delta'], nan_policy='omit')
    ut_mean = dat.groupby('UniTown').agg(np.mean).reset_index()

    better = 'university town' if (ut_mean[ut_mean['UniTown'] == True]['delta'].iloc[0] > ut_mean[ut_mean['UniTown'] == False]['delta'].iloc[0]) else 'non-university town'
    
    res = (ttest_res.pvalue < .01, ttest_res.pvalue, better)
    return res

