import pandas as pd

def date_sorter():

    doc = []
    with open('dates.txt') as file:
        for line in file:
            doc.append(line)

    df = pd.Series(doc)

    df[72] = '7/11/77'
    df[271] = 'August 2008'
    df[272] = '01 Feb 1993'
    df[248] = 'July 1995'
    #####################
    # Match all M/D/Y
    r1i = df.copy()
    r1 = r1i.str.extractall('(?P<month>\d{1,2})[-/](?P<day>\d{1,2})[-/](?P<year>(?:\d{4}|\d{2}))')[['day', 'month', 'year']]
    r1.reset_index(inplace=True)

    r1['day'] = r1['day'].apply(int)
    r1['month'] = r1['month'].apply(int)
    r1['year'] = r1['year'].apply(int)

    # add 1900 or 2000 to two-digit year values
    r1['year'][(17 < r1['year']) & (r1['year'] < 100)] += 1900
    r1['year'][r1['year'] < 100] += 2000

    #######################
    # Match all M/Y
    r2i = r1i[~r1i.index.isin(list(r1.level_0))]
    r2 = r2i.str.extractall('(?P<month>\d{1,2})[/-](?P<year>(?:\d{4}|\d{2}))')[['month', 'year']]
    r2.reset_index(inplace=True)
    r2['day'] = 1
    r2['month'] = r2['month'].apply(int)
    r2['year'] = r2['year'].apply(int)

    # add 1900 or 2000 to two-digit year values
    r2['year'][(17 < r2['year']) & (r2['year'] < 100)] += 1900
    r2['year'][r2['year'] < 100] += 2000

    r3i = r2i[~r2i.index.isin(list(r2.level_0))]

    re_month = '(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May?|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)'


    r3 = r3i.str.extractall('(?P<day>\d{1,2}) +(?P<month>' + re_month + ') +(?P<year>(?:\d{4}|\d{2}))')[['day', 'month', 'year']]
    r3.reset_index(inplace=True)

    r4i = r3i[~r3i.index.isin(list(r3.level_0))]
    r4 = r4i.str.extractall('(?P<month>' + re_month + ')\.? (?P<day>\d{1,2}),? +(?P<year>(?:\d{4}|\d{2}))')[['day', 'month', 'year']]
    r4.reset_index(inplace=True)

    r5i = r4i[~r4i.index.isin(list(r4.level_0))]
    r5 = r5i.str.extractall('\w?(?P<month>' + re_month + '),? +(?P<year>(?:\d{4}|\d{2}))')[['month', 'year']]
    r5.reset_index(inplace=True)
    r5['day'] = 1

    r6i = r5i[~r5i.index.isin(list(r5.level_0))]
    r6 = pd.DataFrame(r6i.str.extractall('(?P<year>(?:19\d\d)|(?:20\d\d))')['year'])
    r6.reset_index(inplace=True)
    r6['day'] = 1
    r6['month'] = 1

    def num_month(s):
        ms = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        for m in ms:
            if s.startswith(m):
                return ms.index(m) + 1

    r3['month'] = r3['month'].apply(num_month)
    r4['month'] = r4['month'].apply(num_month)
    r5['month'] = r5['month'].apply(num_month)


    r1['iter'] = 1
    r2['iter'] = 2
    r3['iter'] = 3
    r4['iter'] = 4
    r5['iter'] = 5
    r6['iter'] = 6


    
    res = pd.concat([r1, r2, r3, r4, r5, r6]).sort_values(['year', 'month', 'day'])


    res['day'] = res['day'].apply(int)
    res['month'] = res['month'].apply(int)
    res['year'] = res['year'].apply(int)
    res = res[res['level_0'] != 9]
        
    return res.reset_index().level_0
