
# Default Libraries
import math
import inspect
import itertools

# Data Science Libraries
import numpy as np
import pandas as pd
import sklearn_pandas

# sklearn Library (https://scikit-learn.org)
import sklearn.model_selection
import sklearn.linear_model
import sklearn.neighbors
import sklearn.metrics
import sklearn.tree
import sklearn.ensemble
import sklearn.naive_bayes
import sklearn.neural_network
from sklearn.externals import joblib

class DataScienceModeler:

    r_X = None
    r_Y = None
    r_x_train = None
    r_x_test = None
    r_y_train = None
    r_y_test = None

    c_X = None
    c_Y = None
    c_x_train = None
    c_x_test = None
    c_y_train = None
    c_y_test = None

    TESTSET_X = None

    COMPETITIONSET_X = None

    ############################################################################
    #                              Data Pre-Processing                         #
    ############################################################################

    def __init__(self):
        pass

    def load_trainingset(self, shuffle):
        print("<==== ====", inspect.stack()[0][3], "==== ====>")

        df = pd.read_csv("trainingset.csv")

        continuous_features = ["1", "2", "6", "8", "10"]
        categorical_features = ["3", "4", "5", "7", "9", "11", "12",
                                "13", "14", "15", "16", "17", "18"]

        df['Claimed'] = np.where(df['ClaimAmount'] > 0, 1, 0)

        col_list = list(df.columns)
        col_list.remove('rowIndex')
        col_list.remove('Claimed')
        col_list.remove('ClaimAmount')
        col_list.insert(0, 'ClaimAmount')
        col_list.insert(0, 'Claimed')
        col_list.insert(0, 'rowIndex')
        df = df[col_list]

        df = pd.get_dummies(
            df, columns=["feature" + n for n in categorical_features],
            dtype=np.int64
        )

        transform_mapper = sklearn_pandas.DataFrameMapper([
            ('rowIndex', None),
            ('Claimed', None),
            ('ClaimAmount', None),
        ], default=sklearn.preprocessing.StandardScaler())
        standardized = transform_mapper.fit_transform(df.copy())
        df = pd.DataFrame(standardized, columns=df.columns)





        print("0. Prepare the Final Data Sets (Classification)")
        self.c_X = df.drop(['rowIndex', 'Claimed', 'ClaimAmount'], axis=1)
        self.c_Y = df.Claimed
        self.c_x_train, self.c_x_test, self.c_y_train, self.c_y_test = sklearn.model_selection\
            .train_test_split(self.c_X, self.c_Y, test_size=0.30, shuffle=shuffle)

        print("0. Prepare the Final Data Sets (Regression)")
        self.r_X = df.drop(['rowIndex', 'Claimed', 'ClaimAmount'], axis=1)
        self.r_Y = df.ClaimAmount
        self.r_x_train, self.r_x_test, self.r_y_train, self.r_y_test = sklearn.model_selection\
            .train_test_split(self.r_X, self.r_Y, test_size=0.30, shuffle=shuffle)





    def load_testset(self, shuffle):

        print("<==== ====", inspect.stack()[0][3], "==== ====>")

        df = pd.read_csv("competitionset.csv")

        continuous_features = ["1", "2", "6", "8", "10"]
        categorical_features = ["3", "4", "5", "7", "9", "11", "12",
                                "13", "14", "15", "16", "17", "18"]

        col_list = list(df.columns)
        col_list.remove('rowIndex')
        col_list.insert(0, 'rowIndex')
        df = df[col_list]

        df = pd.get_dummies(
            df, columns=["feature" + n for n in categorical_features],
            dtype=np.int64
        )

        transform_mapper = sklearn_pandas.DataFrameMapper([
            ('rowIndex', None),
        ], default=sklearn.preprocessing.StandardScaler())
        standardized = transform_mapper.fit_transform(df.copy())
        df = pd.DataFrame(standardized, columns=df.columns)

        print("0. Prepare the Final Data Sets (Regression)")
        self.TESTSET_X = df.drop(['rowIndex'], axis=1)

    def load_competitionset(self, shuffle):
        pass

    def select_features_regression(self, num_features):
        print("<==== ====", inspect.stack()[0][3], "==== ====>")

        feature_select_model = sklearn.tree.DecisionTreeRegressor()

        trans = sklearn.feature_selection.RFE(feature_select_model, n_features_to_select=num_features)
        self.r_x_train = trans.fit_transform(self.r_x_train, self.r_y_train)
        self.r_x_test = trans.fit_transform(self.r_x_test, self.r_y_test)

    def select_features_classification(self, num_features):
        print("<==== ====", inspect.stack()[0][3], "==== ====>")

        feature_select_model = sklearn.tree.DecisionTreeClassifier()

        trans = sklearn.feature_selection.RFE(feature_select_model, n_features_to_select=num_features)
        self.c_x_train = trans.fit_transform(self.c_x_train, self.c_y_train)
        self.c_x_test = trans.fit_transform(self.c_x_test, self.c_y_test)

    ############################################################################
    #                                 Utilities                                #
    ############################################################################

    def print_regression_performance_metrics(self, y_test, y_pred):

        label_prediction_difference = np.subtract(y_test, y_pred)
        MAE = np.mean(np.absolute(label_prediction_difference))
        print("MAE: ", MAE)

    def print_classification_performance_metrics(self, y_test, y_pred):

        confusion_matrix = sklearn.metrics.confusion_matrix(y_test, y_pred)
        print("Confusion Matrix:")
        print(confusion_matrix)

        tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y_test, y_pred).ravel()
        print("TN:", tn, "FP", fp, "FN", fn, "TP", tp)

        f1_score = sklearn.metrics.f1_score(y_test, y_pred)
        print("F1 Performance Score: %.6f%%" % (f1_score * 100))

    ############################################################################
    #                               Model Execution                            #
    ############################################################################

    def experiment(self):

        self.load_trainingset(False)

        #### #### #### #### Classification Model #### #### #### ####
        model_classification = sklearn.tree.ExtraTreeClassifier(
            class_weight=None, criterion='gini', max_depth=None,
            max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, random_state=None,
            splitter='random')
        model_classification = model_classification.fit(self.c_x_train, self.c_y_train)

        #### #### #### #### Regression Model #### #### #### ####
        model_regression = sklearn.svm.SVR(
            C=0.001, cache_size=200, coef0=0.0, degree=3, epsilon=0.1,
            gamma='auto_deprecated', kernel='sigmoid', max_iter=-1, shrinking=True,
            tol=0.001, verbose=False)
        model_regression = model_regression.fit(self.r_x_train, self.r_y_train)

        #### #### #### #### Prediction Process #### #### #### ####
        TAU = 0.2
        y_pred_prob = model_classification.predict_proba(self.c_x_test)
        y_pred_prob = pd.DataFrame(y_pred_prob)[1]
        y_pred_classification = \
            y_pred_prob.apply(
                lambda x: 1
                if x > TAU else 0
            ).values
        y_pred_regression = model_regression.predict(self.r_x_test)
        y_pred_regression = y_pred_classification * y_pred_regression

        self.print_classification_performance_metrics(self.c_y_test, y_pred_classification)
        self.print_regression_performance_metrics(self.r_y_test, y_pred_regression)

    def train(self):

        # Load the training set.
        self.load_trainingset(False)

        #### #### #### #### Classification Model #### #### #### ####
        model_classification = sklearn.tree.ExtraTreeClassifier(
            class_weight=None, criterion='gini', max_depth=None,
            max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, random_state=None,
            splitter='random')
        model_classification = model_classification.fit(self.c_X, self.c_Y)

        #### #### #### #### Regression Model #### #### #### ####
        model_regression = sklearn.svm.SVR(
            C=0.001, cache_size=200, coef0=0.0, degree=3, epsilon=0.1,
            gamma='auto_deprecated', kernel='sigmoid', max_iter=-1, shrinking=True,
            tol=0.001, verbose=False)
        model_regression = model_regression.fit(self.r_X, self.r_Y)

        #### #### #### #### Save Models #### #### #### ####
        joblib.dump(model_classification, 'model_classification')
        joblib.dump(model_regression, 'model_regression')

    def assess(self):

        # Load the test set.
        self.load_testset(False)

        #### #### #### #### Prediction Process #### #### #### ####
        model_classification = joblib.load('model_classification')
        model_regression = joblib.load('model_regression')

        #### #### #### #### Prediction Process #### #### #### ####
        TAU = 0.2
        y_pred_prob = model_classification.predict_proba(self.TESTSET_X)
        y_pred_prob = pd.DataFrame(y_pred_prob)[1]
        y_pred_classification = \
            y_pred_prob.apply(
                lambda x: 1
                if x > TAU else 0
            ).values
        y_pred_regression = model_regression.predict(self.TESTSET_X)
        y_pred_final = y_pred_classification * y_pred_regression

        # Make the submission file.
        submission = pd.DataFrame(y_pred_final, columns=['ClaimAmount'])
        submission.to_csv("submission.csv", index=True, index_label='rowIndex')

        # Print out success message.
        print("COMPLETE: submission.csv created!")

################################################################################
#                                 Main                                         #
################################################################################

if __name__ == "__main__":

    # DataScienceModeler().experiment()
    DataScienceModeler().train()
    # DataScienceModeler().assess()


