from sklearn.model_selection import RandomizedSearchCV
import time

# TODO explore whether it's worthile to include bayesian opt. Random Search works quite well already. Grid Search for exhaustive option?


class HyperparameterTuning:
    """
    This class contains the methods used to perform hyperparameter tuning for models that require/have 2+ hyperparameters
    """

    @staticmethod
    def random_search_tuning(model, parameters, iterations, folds, score_metric):
        """
        Creates RandomizedSearchCV object using passed in parameters.
        n_jobs = -1 will leverage max CPU cores. -2 is likely safer
        """
        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=parameters,
            scoring=score_metric,
            verbose=1,
            n_jobs=-1,
            n_iter=iterations,
            cv=folds,
            return_train_score=True,
        )
        return random_search

    @staticmethod
    def random_search_fit(random_search, train_x, train_y):
        """
        Fits and returns model with best parameters found 
        """
        start = time.time()
        best_params = random_search.fit(train_x, train_y)
        end = time.time()

        best_param_model = best_params.best_estimator_

        print("Runtime:", end - start)
        print("Best Score", best_params.best_score_)
        return best_param_model, best_params
