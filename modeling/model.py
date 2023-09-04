from typing import Union

import pandas as pd
from hyperopt import Trials, fmin, space_eval, tpe
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from helpers.ancillary import save_cv_scores, save_model


def train_model_with_hyperopt(estimator: Union[Pipeline, RegressorMixin, ClassifierMixin],
                              x_train: pd.DataFrame, y_train: Union[pd.DataFrame, pd.Series],
                              model_uid: str, param_space: dict, iterations: int,
                              cv_strategy: Union[int, iter],
                              cv_scoring: str) -> Union[Pipeline, RegressorMixin, ClassifierMixin]:
    """
    Trains a model using Hyperopt.

    :param estimator: an estimator, such as a regression model, a classification model, or a fuller modeling pipeline
    :param x_train: x train
    :param y_train: y train
    :param model_uid: model uid
    :param param_space: parameter search space to optimize
    :param iterations: number of iterations to run the optimization
    :param cv_strategy: cross validation strategy
    :param cv_scoring: how to score cross validation folds
    :return: optimized estimator
    """
    cv_scores_df = pd.DataFrame()

    def _model_objective(params):
        estimator.set_params(**params)
        score = cross_val_score(estimator, x_train, y_train, cv=cv_strategy, scoring=cv_scoring, n_jobs=-1)
        temp_cv_scores_df = pd.DataFrame(score)
        temp_cv_scores_df = temp_cv_scores_df.reset_index()
        temp_cv_scores_df['index'] = 'fold_' + temp_cv_scores_df['index'].astype(str)
        temp_cv_scores_df = temp_cv_scores_df.T
        temp_cv_scores_df = temp_cv_scores_df.add_prefix('fold_')
        temp_cv_scores_df = temp_cv_scores_df.iloc[1:]
        temp_cv_scores_df['mean'] = temp_cv_scores_df.mean(axis=1)
        temp_cv_scores_df['std'] = temp_cv_scores_df.std(axis=1)
        temp_params_df = pd.DataFrame(params, index=list(range(0, len(params) + 1)))
        temp_cv_scores_df = pd.concat([temp_params_df, temp_cv_scores_df], axis=1)
        temp_cv_scores_df = temp_cv_scores_df.dropna()
        nonlocal cv_scores_df
        cv_scores_df = pd.concat([cv_scores_df, temp_cv_scores_df], axis=0)
        return 1 - score.mean()

    trials = Trials()
    best = fmin(_model_objective, param_space, algo=tpe.suggest, max_evals=iterations, trials=trials)
    best_params = space_eval(param_space, best)
        
    cv_scores_df = cv_scores_df.sort_values(by=['mean'], ascending=False)
    cv_scores_df = cv_scores_df.reset_index(drop=True)
    cv_scores_df = cv_scores_df.reset_index()
    cv_scores_df = cv_scores_df.rename(columns={'index': 'ranking'})
    save_cv_scores(cv_scores_df, model_uid)

    estimator.set_params(**best_params)
    estimator.fit(x_train, y_train)

    save_model(estimator, model_uid)
    return estimator
