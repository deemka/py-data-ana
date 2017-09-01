import pandas as pd

# ##################################
# ############  PART 1 #############
df = pd.read_csv('olympics.csv', index_col=0, skiprows=1)

for col in df.columns:
    if col[:2] == '01':
        df.rename(columns={col: 'Gold' + col[4:]}, inplace=True)
    if col[:2] == '02':
        df.rename(columns={col: 'Silver' + col[4:]}, inplace=True)
    if col[:2] == '03':
        df.rename(columns={col: 'Bronze' + col[4:]}, inplace=True)
    if col[:1] == '№':
        df.rename(columns={col: '#' + col[1:]}, inplace=True)

names_ids = df.index.str.split('\s\(')  # split the index by '('

df.index = names_ids.str[0]  # the [0] element is the country name (new index)
df['ID'] = names_ids.str[1].str[:3]  # the [1] element is the abbreviation or ID (take first 3 characters from that)


# Question 1
# Which country has won the most gold medals in summer games?
# This function should return a single string value.
def answer_one():
    return df[df['Gold'] == df['Gold'].max()].index[0]


answer_one()


# Question 2
# Which country had the biggest difference between their summer and winter gold medal counts?
# This function should return a single string value.
def answer_two():
    return df[(df['Gold'] - df['Gold.1']) == (df['Gold'] - df['Gold.1']).max()].index[0]


answer_two()


# Question 3
# Which country has the biggest difference between their summer gold medal counts and winter gold medal
# counts relative to their total gold medal count?
# (Summer Gold−Winter Gold)/Total Gold
# Only include countries that have won at least 1 gold in both summer and winter.
# This function should return a single string value.
def answer_three():
    tmp = df[(df['Gold'] > 0) & (df['Gold.1'] > 0)].copy()
    tmp['a3'] = (tmp['Gold'] - tmp['Gold.1']) / tmp['Gold.2']
    return tmp[tmp['a3'] == tmp['a3'].max()].index[0]


answer_three()


# Question 4
# Write a  function that creates a  Series called "Points" which  is a weighted value  where each gold
# medal  (Gold.2) counts  for 3  points, silver  medals  (Silver.2) for  2 points,  and bronze  medals
# (Bronze.2) for 1 point. The function should return only the column (a Series object) which you created.
# This function should return a Series named Points of length 146census_df = pd.read_csv('census.csv')
def answer_four():
    Points = df['Gold.2'] * 3 + df['Silver.2'] * 2 + df['Bronze.2']
    return Points


answer_four()

# ##################################
# ############  PART 2 #############

census_df = pd.read_csv('census.csv')

# Question 5
# Which state has the most counties in it? (hint: consider the sumlevel key carefully! You'll need
# this for future questions too...)  This function should return a single string value.


def answer_five():
    cdf50 = census_df[census_df['SUMLEV'] == 50].copy()
    cdf50 = cdf50.groupby(['STNAME']).size().reset_index(name='Count').sort_values('Count', ascending=False)
    return cdf50.head(1)['STNAME'].iloc[0]


answer_five()
