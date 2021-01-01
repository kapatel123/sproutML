from model.model_types import get_model
from model.hyperparameter_tuning import random_search_tuning, random_search_fit


class Driver:
    def generate_model(self, x_train, y_train):
        # TODO insert preprocessing steps
        # TODO build score_metric appropriate for chosen prediction problem
        # TODO set all inputs within constant.py
        model, model_parameters = get_model('xgboost')
        random_search_params = random_search_tuning(model, model_parameters,
                                                    iterations=10, folds=5, score_metric='f1_score')
        best_model, best_params = random_search_fit(
            random_search_params)
