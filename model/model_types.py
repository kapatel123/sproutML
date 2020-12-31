from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
from utils import constant


class ModelTypes:
    """
    This class contains methods to retrieve the models and hyperparameters
    for all supported model algorithms (currently - LR, RF, XGB)
    Assumption --> LR is for binary classification. RF (binary or multi). XGB (only multi).
    """

    @staticmethod
    def logistic_regression_model():
        """
        Gets Logistic Regression model and dictionary of hyperparameters.
        :return: model (sklearn.linear_model.LogisticRegression),
            dict of hyperparameters in the format {param (str) : param_distribution}.
        """
        lr_parameter_dict = {
            "logisticregression__penalty": 'elasticnet'
        }
        lr_model = LogisticRegression()
        return lr_model, lr_parameter_dict

    @staticmethod
    def random_forest_model():
        """
        Gets RandomForest model and dictionary of tunable hyperparameters.
        :return: model (sklearn.ensemble.RandomForestClassifier),
            dict of tunable hyperparameters in the format {param (str) : param_distribution}.
        """
        rf_parameter_dict = {
            "randomforestclassifier__n_estimators": sp_randint(100, 1001),
            "randomforestclassifier__max_depth": sp_randint(10, 101),
            "randomforestclassifier__min_samples_split": sp_randint(2, 11),
            "randomforestclassifier__min_samples_leaf": sp_randint(1, 5),
        }
        rf_model = RandomForestClassifier(n_jobs=-1)
        return rf_model, rf_parameter_dict

    @staticmethod
    def xgboost_model(objective=None):
        """
        Gets XGBoost model and dictionary of tunable hyperparameters.
        :param objective (str): learning objective for XGBoost model
        :return: model (xgboost.XGBClassifier),
            dict of tunable hyperparameters in the format {param (str) : param_distribution}.
        """
        xgb_parameter_dict = {
            "xgbclassifier__min_child_weight": sp_randint(1, 11),
            "xgbclassifier__max_depth": sp_randint(3, 16),
            "xgbclassifier__subsample": sp_uniform(0.5, 0.5),
            "xgbclassifier__colsample_bytree": sp_uniform(0.5, 0.5),
            "xgbclassifier__n_estimators": sp_randint(200, 501),
            "xgbclassifier__gamma": sp_uniform(0.5, 1.5),
        }
        xgb_model = XGBClassifier(
            learning_rate=0.02, objective=objective, n_jobs=-1)
        return xgb_model, xgb_parameter_dict

    @staticmethod
    def get_model(model_name):
        """
        Gets model and dictionary of tunable hyperparameters for supported model_name.
        """
        if model_name == 'logistic_regression':
            model, params = ModelTypes.random_forest_model()
        elif model_name == 'randomforest':
            model, params = ModelTypes.random_forest_model()
        elif model_name == 'xgboost':
            model, params = ModelTypes.xgboost_model(objective="multi:softmax")
        return model, params
