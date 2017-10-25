import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler

# from datetime import datetime
from dateutil import parser

X = pd.read_csv('train.csv', encoding='ISO-8859-1')
dftest = pd.read_csv('test.csv', encoding='ISO-8859-1')

latlons = pd.read_csv('latlons.csv')
addr = pd.read_csv('addresses.csv')


def parse_date(s):
    return parser.parse(s)


def get_year(dt):
    return dt.year


def get_month(dt):
    return dt.month


def get_day(dt):
    return dt.day


X['ticket_issued_date'] = X['ticket_issued_date'].apply(parse_date)
X['ticket_issued_year'] = X['ticket_issued_date'].apply(get_year)
X['ticket_issued_month'] = X['ticket_issued_date'].apply(get_month)
X['ticket_issued_quartal'] = (X['ticket_issued_month'] + 2) / 3
X['ticket_issued_day'] = X['ticket_issued_date'].apply(get_day)
X['judgment_amount_squared'] = X['judgment_amount'].apply(np.square)

dftest['ticket_issued_date'] = dftest['ticket_issued_date'].apply(parse_date)
dftest['ticket_issued_year'] = dftest['ticket_issued_date'].apply(get_year)
dftest['ticket_issued_month'] = dftest['ticket_issued_date'].apply(get_month)
dftest['ticket_issued_quartal'] = (dftest['ticket_issued_month'] + 2) / 3
dftest['ticket_issued_day'] = dftest['ticket_issued_date'].apply(get_day)
dftest['judgment_amount_squared'] = dftest['judgment_amount'].apply(np.square)

# Merge geo coordinates
X = X.merge(addr, on='ticket_id').merge(latlons, on='address')
dftest = dftest.merge(addr, on='ticket_id').merge(latlons, on='address')

latlons = None
addr = None

feature_sets = {0: ['judgment_amount'],
                1: ['judgment_amount', 'ticket_issued_year'],
                2: ['judgment_amount', 'ticket_issued_year', 'ticket_issued_month'],
                3: ['lon', 'lat'],
                4: ['judgment_amount', 'lon', 'lat'],
                5: ['judgment_amount', 'lon', 'lat', 'ticket_issued_month'],
                6: ['judgment_amount', 'lon', 'lat', 'ticket_issued_quartal', 'ticket_issued_year'],
                7: ['judgment_amount', 'lon', 'lat', 'clean_up_cost'],
                8: ['judgment_amount', 'lon', 'lat', 'clean_up_cost', 'ticket_issued_month'],
                9: ['judgment_amount', 'lon', 'lat', 'clean_up_cost', 'ticket_issued_month', 'ticket_issued_year']}

fs = feature_sets[6].copy()
fs.append('compliance')
X = X[fs]

# Remove NaN rows
X.dropna(axis=0, how='any', inplace=True)

y = X['compliance'].copy()
X = X.drop('compliance', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y)

# Prepare evaluation input/output
tid_test = dftest['ticket_id'].copy()
dftest = dftest[X_test.columns]

# Some entries contain no lat/lon values
dftest.fillna(value=0, inplace=True)

# RandomForest
if True:
    rf = RandomForestClassifier().fit(X_train, y_train)
    gs_rf = GridSearchCV(rf, n_jobs=-1, verbose=5,
                         param_grid={'n_estimators': [100, 200, 500],
                                     'max_depth': [10, 20, 40, 100],
                                     'max_features': [1, 2, 3, 4, 5]},
                         scoring='roc_auc').fit(X_train, y_train)
    print('Best params RF: {}'.format(gs_rf.best_params_))

    y_score_rf = rf.predict_proba(X_test)[:, 1]
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_score_rf)
    auc_rf = auc(fpr_rf, tpr_rf)
    res_rf = pd.Series(rf.predict_proba(dftest)[:, 1], index=tid_test)
    print("RF: {}".format(auc_rf))

# Logistic Regression
if False:
    lr = LogisticRegression(C=8, penalty='l1').fit(X_train_scaled, y_train)
    gs_lr = GridSearchCV(lr, n_jobs=-1, verbose=5,
                         param_grid={'C': [1, 5, 10]},
                         scoring='roc_auc').fit(X_train_scaled, y_train)
    y_score_lr = lr.predict_proba(X_test_scaled)[:, 1]
    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_score_lr)
    auc_lr = auc(fpr_lr, tpr_lr)
    res_lr = pd.Series(lr.predict_proba(dftest_scaled)[:, 1], index=tid_test)
    print("LR: {}".format(auc_lr))

# Neural Network
if False:
    mlp = MLPClassifier(hidden_layer_sizes=[200, 200], alpha=1.0, activation='logistic',
                        random_state=0, solver='lbfgs').fit(X_train_scaled, y_train)
    gs_mlp = GridSearchCV(mlp, n_jobs=-1, verbose=5,
                          param_grid={'hidden_layer_sizes': [[10, 10], [50, 50], [20, 20]],
                                      'alpha': [.001, 0.01, 0.1, 1., 10.]},
                          scoring='roc_auc').fit(X_train, y_train)
    y_score_mlp = mlp.predict_proba(X_test_scaled)[:, 1]
    fpr_mlp, tpr_mlp, _ = roc_curve(y_test, y_score_mlp)
    auc_mlp = auc(fpr_mlp, tpr_mlp)
    print("MLP: {}".format(auc_mlp))

# GradientBoost
if True:
    gb = GradientBoostingClassifier()
    #gs_gb = GridSearchCV(gb, n_jobs=-1, verbose=5,
    #                     param_grid={'learning_rate': [0.001, 0.01, 0.1, 0.5, 1.0],
    #                                 'n_estimators': [100, 400, 1000]},
    #                     scoring='roc_auc').fit(X_train, y_train)
    #print('Best params GB: {}'.format(gs_gb.best_params_))

    gb = GradientBoostingClassifier(learning_rate=0.1, n_estimators=1000).fit(X_train, y_train)
    y_score_gb = gb.predict_proba(X_test)[:, 1]
    fpr_gb, tpr_gb, _ = roc_curve(y_test, y_score_gb)
    auc_gb = auc(fpr_gb, tpr_gb)
    res_gb = pd.Series(gb.predict_proba(dftest)[:, 1], index=tid_test)
    print("GB: {}".format(auc_gb))


def blight_model():
    return res_gb
