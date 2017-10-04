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


def answer_one():
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures

    # Your code here
    res = np.zeros([4, 100])
    x_ = np.linspace(0, 10, 100)
    for g in [1, 3, 6, 9]:
        # transform features to polynomial
        X_train_pf = PolynomialFeatures(degree=g).fit_transform(X_train.reshape(-1, 1))
        x_pf = PolynomialFeatures(degree=g).fit_transform(x_.reshape(-1, 1))
        # train the linear model using the polynomial features
        lr = LinearRegression().fit(X_train_pf, y_train)

        # predict
        pred = lr.predict(x_pf)
        res[int(g / 3), :] = pred
    return res


answer_one()
