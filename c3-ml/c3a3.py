import numpy as np
import pandas as pd

def answer_one():
    df = pd.read_csv('fraud_data.csv')
    return df[df['Class'] == 1].count()['Class'] / df['Class'].count()

# Use X_train, X_test, y_train, y_test for all of the following questions
from sklearn.model_selection import train_test_split

df = pd.read_csv('fraud_data.csv')

X = df.iloc[:,:-1]
y = df.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

def answer_two():
    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import recall_score, accuracy_score
    dummy_majority = DummyClassifier(strategy = 'most_frequent')\
    .fit(X_train, y_train)
    y_pred = dummy_majority.predict(X_test)
    
    return (accuracy_score(y_test, y_pred), recall_score(y_test, y_pred))

def answer_three():
    from sklearn.metrics import recall_score, precision_score, accuracy_score
    from sklearn.svm import SVC

    clf = SVC()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return (accuracy_score(y_test, y_pred), recall_score(y_test, y_pred), precision_score(y_test, y_pred))

def answer_four():
    from sklearn.metrics import confusion_matrix
    from sklearn.svm import SVC

    clf = SVC(C=1e9, gamma=1e-07)

    clf.fit(X_train, y_train)

    y_scores = clf.decision_function(X_test)
    y_pred = y_scores*0 + (y_scores > -220)
    return confusion_matrix(y_test, y_pred)
answer_four()


def answer_five():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import precision_recall_curve, roc_curve
    lr = LogisticRegression().fit(X_train, y_train)
    prec, rec, thresh = precision_recall_curve(y_test, lr.decision_function(X_test))
    fpr, tpr, _ = roc_curve(y_test, lr.decision_function(X_test))
    import matplotlib.pyplot as plt
    plt.plot(fpr, tpr)
    plt.show()
    i, = np.where(prec == 0.75)
    j, = np.where(abs(fpr - 0.16) < .0005)
    return (rec[i[0]], tpr[j[0]])


def answer_six():    
    from sklearn.model_selection import GridSearchCV
    from sklearn.linear_model import LogisticRegression

    lr = LogisticRegression()
    gs = GridSearchCV(lr, param_grid={'C': [0.01, 0.1, 1, 10, 100], 'penalty': ['l1', 'l2']}, scoring='recall')\
                                      .fit(X_train, y_train)
    
    return gs.cv_results_['mean_test_score'].reshape(5,2)
