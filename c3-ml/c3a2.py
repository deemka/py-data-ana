import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


np.random.seed(0)
n = 15
x = np.linspace(0, 10, n) + np.random.randn(n) / 5
y = np.sin(x) + x / 6 + np.random.randn(n) / 10


X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)


# You can use this function to help you visualize the dataset by
# plotting a scatterplot of the data points
# in the training and test sets.
def part1_scatter():
    import matplotlib.pyplot as plt
    # %matplotlib notebook
    plt.figure()
    plt.scatter(X_train, y_train, label='training data')
    plt.scatter(X_test, y_test, label='test data')
    plt.legend(loc=4)

# NOTE: Uncomment the function below to visualize the data, but be sure
# to **re-comment it before submitting this assignment to the autograder**.
# part1_scatter()


# ############### Question 1 #####################
# Write a function that fits a polynomial LinearRegression model on the training data X_train for
# degrees 1, 3, 6, and 9. (Use PolynomialFeatures in sklearn.preprocessing to create the polynomial
# features and then fit a linear regression model) For each model, find 100 predicted values over
# the interval x = 0 to 10 (e.g. np.linspace(0,10,100)) and store this in a numpy array. The first
# row of this array should correspond to the output from the model trained on degree 1, the second
# row degree 3, the third row degree 6, and the fourth row degree 9.


def answer_one():
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures

    # Your code here
    res = np.zeros([4, 100])
    x_ = np.linspace(0, 10, 100)

    for g in [1, 3, 6, 9]:

        # transform training input to polynomial features
        X_train_pf = PolynomialFeatures(degree=g).fit_transform(X_train.reshape(-1, 1))

        # transform prediction input to polynomial features
        x_pf = PolynomialFeatures(degree=g).fit_transform(x_.reshape(-1, 1))

        # train linear model using polynomial features
        lr = LinearRegression().fit(X_train_pf, y_train)

        # predict
        pred = lr.predict(x_pf)
        res[int(g / 3), :] = pred
    return res


answer_one()


# ############### Question 2 #####################
# Write a function that fits a polynomial LinearRegression model on the training data X_train for
# degrees 0 through 9. For each model compute the R2R2 (coefficient of determination) regression
# score on the training data as well as the the test data, and return both of these arrays in a
# tuple. This function should return one tuple of numpy arrays (r2_train, r2_test). Both arrays
# should have shape (10,)


def answer_two():
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.metrics.regression import r2_score

    # Your code here
    r_train = np.empty(10)
    r_test = np.empty(10)
    for g in range(0, 10):
        # transform training input to polynomial features
        X_train_pf = PolynomialFeatures(degree=g).fit_transform(X_train.reshape(-1, 1))

        # transform prediction input to polynomial features
        X_test_pf = PolynomialFeatures(degree=g).fit_transform(X_test.reshape(-1, 1))

        lr = LinearRegression().fit(X_train_pf, y_train)
        r_train[g] = r2_score(y_train, lr.predict(X_train_pf))
        r_test[g] = r2_score(y_test, lr.predict(X_test_pf))
    return (r_train, r_test)


# ############### Question 3 #####################
# Based on the R2 scores from question 2 (degree levels 0 through 9), what degree level
# corresponds to a model that is underfitting? What degree level corresponds to a model that is
# overfitting? What choice of degree level would provide a model with good generalization
# performance on this dataset? Note: there may be multiple correct solutions to this question.
# (Hint: Try plotting the R2 scores from question 2 to visualize the relationship between degree
# level and R2 )
# This function should return one tuple with the degree values in this order:
# (Underfitting, Overfitting, Good_Generalization)


def answer_three():
    amax = np.argmax(answer_two()[1])
    a = np.arange(10)
    return (a[0], a[-1], amax)


answer_three()


# ############### Question 4 #####################
# Training models on high degree polynomial features can result in overly complex models that
# overfit, so we often use regularized versions of the model to constrain model complexity, as we saw
# with Ridge and Lasso linear regression.
# For this question, train two models: a non-regularized
# LinearRegression model (default parameters) and a regularized Lasso Regression model (with
# parameters alpha=0.01, max_iter=10000) on polynomial features of degree 12. Return the R2R2 score
# for both the LinearRegression and Lasso model's test sets.
# This function should return one tuple (LinearRegression_R2_test_score, Lasso_R2_test_score)


def answer_four():
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import Lasso, LinearRegression
    from sklearn.metrics.regression import r2_score

    g = 12
    # transform training input to polynomial features
    X_train_pf = PolynomialFeatures(degree=g).fit_transform(X_train.reshape(-1, 1))

    # transform prediction input to polynomial features
    X_test_pf = PolynomialFeatures(degree=g).fit_transform(X_test.reshape(-1, 1))

    # train regular linear
    lr = LinearRegression().fit(X_train_pf, y_train)

    # train Lasso
    lass = Lasso(alpha=0.1, max_iter=10000).fit(X_train_pf, y_train)

    # compute R2 scores
    sc_lr = r2_score(y_test, lr.predict(X_test_pf))
    sc_lass = r2_score(y_test, lass.predict(X_test_pf))
    return (sc_lr, sc_lass)


answer_four()



import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


mush_df = pd.read_csv('mushrooms.csv')
mush_df2 = pd.get_dummies(mush_df)

X_mush = mush_df2.iloc[:,2:]
y_mush = mush_df2.iloc[:,1]

# use the variables X_train2, y_train2 for Question 5
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_mush, y_mush, random_state=0)

# For performance reasons in Questions 6 and 7, we will create a smaller version of the
# entire mushroom dataset for use in those questions.  For simplicity we'll just re-use
# the 25% test split created above as the representative subset.
#
# Use the variables X_subset, y_subset for Questions 6 and 7.
X_subset = X_test2
y_subset = y_test2


# ############### Question 4 #####################
# Using X_train2 and y_train2 from the preceeding cell, train a DecisionTreeClassifier with default
# parameters and random_state=0. What are the 5 most important features found by the decision tree?

# As a reminder, the feature names are available in the X_train2.columns property, and the order of
# the features in X_train2.columns matches the order of the feature importance values in the
# classifier's feature_importances_ property.

# This function should return a list of length 5 containing the feature names in descending order of
# importance.

def answer_five():
    from sklearn.tree import DecisionTreeClassifier
    clf = DecisionTreeClassifier(random_state=0).fit(X_train2, y_train2)

    # Your code here
    f = clf.feature_importances_
    f5_idx = np.argsort(f)[::-1][:5]

    return X_train2.columns[f5_idx].tolist()


def answer_six():
    from sklearn.svm import SVC
    from sklearn.model_selection import validation_curve

    train_scores, test_scores = validation_curve(SVC(random_state=0), X_subset, y_subset,
                                                 param_name='gamma',
                                                 param_range=np.logspace(-4, 1, 6), cv=3)
    train_scores = train_scores.mean(1)
    test_scores = test_scores.mean(1)
    return (train_scores, test_scores)


def answer_seven():
    gmax = np.argmax(answer_six()[1])
    return(float(10.**(0.0 - 4.0)), float(10.0**(5.0 - 4.0)), float(10.**(gmax - 4.0)))
    # return float(10**(float(gmax-4)))


answer_seven()
