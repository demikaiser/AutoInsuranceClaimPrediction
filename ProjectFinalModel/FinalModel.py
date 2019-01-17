import pandas as pd
import numpy as np
import xgboost as xgb
import winsound
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib


################################################################################
#                         Experiments with 7-3 Training Set                    #
################################################################################

def experiment_xgboost_classifier(X_train, X_test, Y_train):

    Y_train = np.where(Y_train > 0, 1, 0)

    model = xgb.XGBClassifier(max_depth=20, learning_rate=.15, n_estimators=400)
    sw = ((Y_train * 15) - Y_train) + np.ones(len(Y_train))
    model.fit(X_train, Y_train, sample_weight=sw)

    preds = model.predict(X_test)
    return preds


def experiment_xgboost_regressor(X_train, X_test, Y_train):

    model = xgb.XGBRegressor(max_depth=20, learning_rate=0.001)
    model.fit(X_train, Y_train)

    preds = model.predict(X_test)
    return preds


def run_experiment_70_30():

    df = pd.read_csv("trainingset.csv")

    X = df.drop(['ClaimAmount', 'rowIndex'], axis=1)
    Y = df.ClaimAmount

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, shuffle=False, random_state=0)

    regress_preds = experiment_xgboost_regressor(X_train, X_test, Y_train)
    class_preds = experiment_xgboost_classifier(X_train, X_test, Y_train)

    result = regress_preds * class_preds

    print("Test set Mae:", mean_absolute_error(Y_test, result))

    result = np.where(result > 0, 1, 0)
    Y_test = np.where(Y_test > 0, 1, 0)

    print("Test set F1:", f1_score(Y_test, result))
    print("Test set Confusion:\n ", confusion_matrix(Y_test, result))


################################################################################
#                         Train and Assess for Competition                     #
################################################################################

def train_classification_model(X_train, Y_train):

    # Convert the labels for the classification.
    Y_train = np.where(Y_train > 0, 1, 0)

    # The XGBoost model for classification.
    model_classification = xgb.XGBClassifier(max_depth=20,
                                             learning_rate=.15,
                                             n_estimators=400)
    sw = ((Y_train * 15) - Y_train) + np.ones(len(Y_train))
    model_classification.fit(X_train, Y_train, sample_weight=sw)

    # Save the model for the competition.
    joblib.dump(model_classification, 'model_classification')


def train_regression_model(X_train, Y_train):

    # The XGBoost model for regression.
    model_regression = xgb.XGBRegressor(max_depth=20, learning_rate=0.001)
    model_regression.fit(X_train, Y_train)

    # Save the model for the competition.
    joblib.dump(model_regression, 'model_regression')


def train_all():

    df = pd.read_csv("trainingset.csv")

    # Convert the data for the training.
    X_ENTIRE_TRAINING = df.drop(['ClaimAmount', 'rowIndex'], axis=1)
    Y_ENTIRE_TRAINING = df.ClaimAmount

    # Train two models for the classification and the regression.
    train_classification_model(X_ENTIRE_TRAINING, Y_ENTIRE_TRAINING)
    train_regression_model(X_ENTIRE_TRAINING, Y_ENTIRE_TRAINING)


def assess_competition_set():

    df = pd.read_csv("competitionset.csv")
    X_ENTIRE_TEST = df.drop(['rowIndex'], axis=1)

    # Load all the trained models
    model_classification = joblib.load('model_classification')
    model_regression = joblib.load('model_regression')

    # Predict each sections and generate the final prediction.
    class_preds = model_classification.predict(X_ENTIRE_TEST)
    regress_preds = model_regression.predict(X_ENTIRE_TEST)

    y_pred_final = regress_preds * class_preds

    # Make the submission file.
    submission = pd.DataFrame(y_pred_final, columns=['ClaimAmount'])
    submission.to_csv("predictedclaimamount.csv", index=True, index_label='rowIndex')

    # Print out success message.
    print("COMPLETE: predictedclaimamount.csv created!")


def compare_two_submission(df1_name, df2_name):
    """ Compare submissions, this is an internal utility.
    """
    df1 = pd.read_csv(df1_name)
    df2 = pd.read_csv(df2_name)

    print("Result:", np.allclose(df1.values[:, 1], df2.values[:, 1]))

    a = np.where(df1.values[:, 1] > 0, 1, 0)
    b = np.where(df2.values[:, 1] > 0, 1, 0)
    print(confusion_matrix(a, b))


if __name__ == "__main__":

    run_experiment_70_30()
    # train_all()
    # assess_competition_set()

    # Beep when finished.
    winsound.Beep(250, 100)

























