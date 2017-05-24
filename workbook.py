import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import ShuffleSplit
import numpy as np
import matplotlib.pyplot as plt


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring=rmsle_est)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

def rmsle(h, y):
    return np.sqrt(np.square(np.log(h + 1) - np.log(y + 1)).mean())

def rmsle_est(est, x, y):
    return np.sqrt(np.square(np.log(est.predict(x) + 1) - np.log(y + 1)).mean())

def main():
    df_train = pd.read_csv('train.csv')
    drop_tmp_list = []

    # scoring = make_scorer(rmsle, greater_is_better=False)
    # loss = make_scorer(rmsle, greater_is_better=True)

    for i in df_train.columns:
        if df_train[i].dtype == 'object' and df_train[i].unique().shape[0] > 10:
            drop_tmp_list.append(i)

    df_train.drop(drop_tmp_list, axis=1, inplace=True)
    df_train.dropna(axis=0, inplace=True)
    df_train = pd.get_dummies(df_train)

    dx, dxt, dy, dyt = train_test_split(df_train.drop(['price_doc'], axis=1), df_train.price_doc, test_size=.2)

    model = LinearRegression(normalize=True)
    #model = ExtraTreesRegressor()

    title = "Learning Curves"
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
    plot_learning_curve(model, title, dx, dy, cv=cv, n_jobs=3)

    plt.show()

    #model.fit(dx, dy)

    pred = model.predict(dx)
    pred_t = model.predict(dxt)

    print(rmsle(dy,pred))
    print(rmsle(dyt, pred_t))

if __name__ == '__main__':
    main()