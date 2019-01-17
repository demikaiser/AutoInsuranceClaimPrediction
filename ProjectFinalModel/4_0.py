import pandas as pd
import numpy as np
import xgboost as xgb
import winsound
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from timeit import default_timer as timer


def xgboost_classifier(X_train, X_test, Y_train):

    Y_train = np.where(Y_train > 0, 1, 0)

    model = xgb.XGBClassifier(learning_rate=0.19, n_estimators=800, max_depth=20, min_child_weight=1, gamma=.1,
                              subsample=.8, colsample_bytree=.8, objective='binary:logistic',
                              booster='gbtree', seed=27, reg_alpha=.005)

    sw = ((Y_train * 15) - Y_train) + np.ones(len(Y_train))

    model.fit(X_train, Y_train, sample_weight=sw)
    preds = model.predict_proba(X_test)[:,1]

    preds = np.where(preds > .6158, 1, 0)

    return preds


def xgboost_regressor(X_train, X_test, Y_train):

    model = xgb.XGBRegressor(objective='reg:linear', colsample_bytree=.7, colsample_bylevel=.8, eta=0.0001,
                             max_depth=20, alpha=1, n_estimators=1, booster='gbtree', min_child_weight=1, gamma=0.1,
                             subsample=1, reg_alpha=70)

    model.fit(X_train, Y_train)
    preds = model.predict(X_test)

    return preds


def run_class_regress(df):

    X = df.drop(['ClaimAmount', 'rowIndex'], axis=1)
    Y = df.ClaimAmount

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, shuffle=False, random_state=0)

    regress_preds = xgboost_regressor(X_train, X_test, Y_train)
    class_preds = xgboost_classifier(X_train, X_test, Y_train)

    result = regress_preds * class_preds

    print("Test set Mae:", mean_absolute_error(Y_test, result))

    result = np.where(result > 0, 1, 0)
    y_test = np.where(Y_test > 0, 1, 0)

    print("Test set F1:", f1_score(y_test, result))
    print("Test set Confusion:\n ", confusion_matrix(y_test, result))


def main():

    df = pd.read_csv("trainingset.csv")

    start = timer()
    run_class_regress(df)
    end = timer()
    print("Took", end - start, "seconds")

    winsound.Beep(250, 1000)


if __name__ == "__main__":
    main()

