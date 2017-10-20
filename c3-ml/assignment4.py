import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import recall_score, accuracy_score, precision_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler

from datetime import datetime
from dateutil import parser

dftrain = pd.read_csv('train.csv', encoding='ISO-8859-1', low_memory=False)
dftest = pd.read_csv('test.csv', encoding='ISO-8859-1', low_memory=False)

latlons = pd.read_csv('latlons.csv')
addr = pd.read_csv('addresses.csv')

""" 
Train columns:
['ticket_id', 'agency_name', 'inspector_name', 'violator_name',
   'violation_street_number', 'violation_street_name',
   'violation_zip_code', 'mailing_address_str_number',
   'mailing_address_str_name', 'city', 'state', 'zip_code',
   'non_us_str_code', 'country', 'ticket_issued_date', 'hearing_date',
   'violation_code', 'violation_description', 'disposition', 'fine_amount',
   'admin_fee', 'state_fee', 'late_fee', 'discount_amount',
   'clean_up_cost', 'judgment_amount', 'payment_amount', 'balance_due',
   'payment_date', 'payment_status', 'collection_status',
   'grafitti_status', 'compliance_detail', 'compliance']

Test coulumns:
['ticket_id', 'agency_name', 'inspector_name', 'violator_name',
   'violation_street_number', 'violation_street_name',
   'violation_zip_code', 'mailing_address_str_number',
   'mailing_address_str_name', 'city', 'state', 'zip_code',
   'non_us_str_code', 'country', 'ticket_issued_date', 'hearing_date',
   'violation_code', 'violation_description', 'disposition', 'fine_amount',
   'admin_fee', 'state_fee', 'late_fee', 'discount_amount',
   'clean_up_cost', 'judgment_amount', 'grafitti_status']
"""
traincols = [
    'ticket_id',
    #'agency_name',
    #'violation_street_number',
    #'violation_street_name',
    #'violation_zip_code',
    #'mailing_address_str_name',
    'city',
    #'state',
    #'zip_code',
    'ticket_issued_date',
    'violation_code',
    'fine_amount',
    'discount_amount',
    #'graffiti_status',
    'compliance']

USE_GEO = True

# Remove rows with compliance = Null
dftrain.dropna(subset=['compliance'], inplace=True)

testcols = traincols.copy()
testcols.remove('compliance')


# Feature selection
dftrain = dftrain[traincols]
dftest = dftest[testcols]


def parse_date(s):
    return parser.parse(s)

def get_month(dt):
    return dt.month

def get_day(dt):
    return dt.day



dftrain['ticket_issued_date'] = dftrain['ticket_issued_date'].apply(parse_date)
dftrain['ticket_issued_month'] = dftrain['ticket_issued_date'].apply(get_month)
dftrain['ticket_issued_quartal'] = dftrain['ticket_issued_month']/3
dftrain['ticket_issued_day'] = dftrain['ticket_issued_date'].apply(get_day)
dftrain['fine_amount_squared'] = dftrain['fine_amount'].apply(np.square)
dftrain.drop('ticket_issued_date', axis=1, inplace=True)
dftrain.drop('ticket_issued_day', axis=1, inplace=True)

if USE_GEO:
    # Merge geo coordinates
    dftrain = dftrain.merge(addr, on='ticket_id').merge(latlons, on='address')
    dftrain.drop('address', axis=1, inplace=True)

dftest['ticket_issued_date'] = dftest['ticket_issued_date'].apply(parse_date)
dftest['ticket_issued_month'] = dftest['ticket_issued_date'].apply(get_month)
dftest['ticket_issued_quartal'] = dftest['ticket_issued_month']/3
dftest['ticket_issued_day'] = dftest['ticket_issued_date'].apply(get_day)
dftest['fine_amount_squared'] = dftest['fine_amount'].apply(np.square)
dftest.drop('ticket_issued_date', axis=1, inplace=True)
dftest.drop('ticket_issued_day', axis=1, inplace=True)

if USE_GEO:
    # Merge geo coordinates
    dftest = dftest.merge(addr, on='ticket_id').merge(latlons, on='address')
    dftest.drop('address', axis=1, inplace=True)
    
# Remove NaN rows
dftrain.dropna(axis=0, how='any', inplace=True)
dftest.dropna(axis=0, how='any', inplace=True)

#dftrain.drop(['fine_amount', 'fine_amount_squared', 'ticket_issued_month'], axis=1, inplace=True)
#dftest.drop(['fine_amount', 'fine_amount_squared', 'ticket_issued_month'], axis=1, inplace=True)

categorical = [    'agency_name',
                   #'violation_street_number',
                   'violation_street_name',
                   #'violation_zip_code',
                   'mailing_address_str_name',
                   'city',
                   'state',
                   #'zip_code',
                   'violation_code',
                   #'graffiti_status'
]

if True:
    label_encoders = {}
    for lbl in categorical:
        if lbl in dftrain.columns:
            dftrain[lbl] = dftrain[lbl].apply(str)
            dftest[lbl] = dftest[lbl].apply(str)
            label_encoders[lbl] = LabelEncoder().fit(pd.concat([dftrain[lbl], dftest[lbl]])) 
            dftrain[lbl] = label_encoders[lbl].transform(dftrain[lbl])
            dftest[lbl] = label_encoders[lbl].transform(dftest[lbl])


tidtrain = dftrain['ticket_id']
y = dftrain['compliance']
X = dftrain.drop('compliance', axis=1)
X = X.drop('ticket_id', axis=1)


X = X[['fine_amount', 'lon', 'lat']]
X_train, X_test, y_train, y_test = train_test_split(X, y)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Prepare evaluation input
tid_test = dftest['ticket_id']
dftest.drop('ticket_id', axis=1, inplace=True)
dftest = dftest[X_test.columns]
dftest_scaled = scaler.transform(dftest)

# RandomForest
clf = RandomForestClassifier().fit(X_train, y_train)
gs_rf = GridSearchCV(clf, n_jobs=-1, verbose=5,
                     param_grid={'n_estimators': [100, 200, 500],
                                 'max_depth': [10, 20, 40, 100],
                                 'max_features': [1,2,3,4,5]},
                     scoring='roc_auc').fit(X_train, y_train)
y_score_rf = clf.predict_proba(X_test)[:, 1]
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_score_rf)
auc_rf = auc(fpr_rf, tpr_rf)
res_clf = pd.Series(clf.predict_proba(dftest)[:, 1], index=tid_test)

# Logistic Regression
lr = LogisticRegression(C=8, penalty='l1').fit(X_train_scaled, y_train)
gs_lr = GridSearchCV(lr, n_jobs=-1, verbose=5,
                     param_grid={'C': [1, 5, 10]},
                     scoring='roc_auc').fit(X_train_scaled, y_train)
y_score_lr = lr.predict_proba(X_test_scaled)[:, 1]
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_score_lr)
auc_lr = auc(fpr_lr, tpr_lr)
res_lr = pd.Series(lr.predict_proba(dftest_scaled)[:, 1], index=tid_test)

mlp = MLPClassifier(hidden_layer_sizes = [200, 200], alpha = 1.0, activation='logistic',
                    random_state = 0, solver='lbfgs').fit(X_train_scaled, y_train)
gs_mlp = GridSearchCV(mlp, n_jobs=-1, verbose=5,
                      param_grid={'hidden_layer_sizes': [[10, 10], [50, 50], [20, 20]],
                                  'alpha': [.001, 0.01, 0.1, 1., 10.]},
                      scoring='roc_auc').fit(X_train, y_train)
y_score_mlp = mlp.predict_proba(X_test_scaled)[:, 1]
fpr_mlp, tpr_mlp, _ = roc_curve(y_test, y_score_mlp)
auc_mlp = auc(fpr_mlp, tpr_mlp)

# GradientBoost
gb = GradientBoostingClassifier().fit(X_train, y_train)
y_score_gb = gb.predict_proba(X_test)[:, 1]
fpr_gb, tpr_gb, _ = roc_curve(y_test, y_score_gb)
auc_gb = auc(fpr_gb, tpr_gb)
res_gb= pd.Series(gb.predict_proba(dftest)[:, 1], index=tid_test)
print("RF: {}".format(auc_rf))
print("LR: {}".format(auc_lr))
print("MLP: {}".format(auc_mlp))
print("GB: {}".format(auc_gb))

def blight_model():
    return res
