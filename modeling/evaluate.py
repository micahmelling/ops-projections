import os
from typing import List, Union

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.pipeline import Pipeline

from helpers.ancillary import make_directories_if_not_exists
from modeling.config import evaluation_named_tuple

plt.switch_backend('Agg')


def make_predict_vs_actual_dataframe(estimator: Union[Pipeline, RegressorMixin], x_df: pd.DataFrame,
                                     target_series: Union[pd.Series]) -> pd.DataFrame:
    """
    Creates a dataframe of predictions vs. actuals.

    :param estimator: estimator object
    :param x_df: predictor dataframe
    :param target_series: target values
    :return: dataframe of predictions and actuals, with the predictions stored in the 'pred' column
    """
    return pd.concat(
        [
            pd.DataFrame(estimator.predict(x_df), columns=['pred']),
            target_series.reset_index(drop=True)
        ],
        axis=1)


def make_full_predictions_dataframe(estimator: Union[Pipeline, RegressorMixin], model_uid: str, x_df: pd.DataFrame,
                                    target_series: Union[pd.Series]) -> pd.DataFrame:
    """
    Produces a dataframe consisting of a point estimate, a lower bound, an upper bound, and the actual value.

    :param estimator: estimator object
    :param model_uid: model uid
    :param x_df: predictor dataframe
    :param target_series: target values
    :returns: pandas dataframe of predictions
    """
    df = make_predict_vs_actual_dataframe(estimator, x_df, target_series)
    df = df[['pred', target_series.name]]
    x_df = x_df.reset_index(drop=True)
    df = pd.concat([df, x_df], axis=1)
    save_path = os.path.join('modeling', 'model_results', model_uid, 'diagnostics', 'predictions')
    make_directories_if_not_exists([save_path])
    df.to_csv(os.path.join(save_path, 'predictions_vs_actuals.csv'), index=False)
    return df


def _evaluate_model(target_series: pd.Series, prediction_series: pd.Series, scorer: callable,
                    metric_name: str) -> pd.DataFrame:
    """
    Applies a scorer function to evaluate predictions against the ground-truth labels.

    :param target_series: target series
    :param prediction_series: prediction series
    :param scorer: scoring function to evaluate the predictions; expected to be like a scikit-learn metrics' callable
    :param metric_name: name of the metric we are using to score our model
    :returns: pandas dataframe reflecting the scoring results for the metric of interest
    """
    score = scorer(target_series, prediction_series)
    df = pd.DataFrame({metric_name: [score]})
    return df


def run_and_save_evaluation_metrics(target_series: pd.Series, prediction_series: pd.Series, model_uid: str,
                                    evaluation_list: List[evaluation_named_tuple]) -> None:
    """
    Runs a series of evaluations metrics on a model's predictions and writes the results locally.

    :param target_series: target series
    :param prediction_series: prediction series
    :param model_uid: model uid
    :param evaluation_list: list of named tuples, which each tuple having the ordering of: the column with the
    predictions, the scoring function callable, and the name of the metric
    """
    main_df = pd.DataFrame()
    for evaluation_config in evaluation_list:
        temp_df = _evaluate_model(target_series, prediction_series,
                                  evaluation_config.scorer_callable, evaluation_config.metric_name)
        main_df = pd.concat([main_df, temp_df], axis=1)
    main_df = main_df.T
    main_df.reset_index(inplace=True)
    main_df.columns = ['scoring_metric', 'holdout_score']
    save_path = os.path.join('modeling', 'model_results', model_uid, 'diagnostics', 'evaluation_files')
    make_directories_if_not_exists([save_path])
    main_df.to_csv(os.path.join(save_path, 'evaluation_scores.csv'), index=False)


def run_omnibus_model_evaluation(estimator: Union[Pipeline, RegressorMixin],
                                 x_df: pd.DataFrame, target_series: pd.Series, model_uid: str,
                                 evaluation_list: List[evaluation_named_tuple], target: str) -> None:
    """
    Runs a series of model evaluation techniques. Namely, providing scores of various metrics on the entire dataset
    and on segments of the dataset.

    :param estimator: trained regresion model or pipeline with regression model
    :param x_df: x predictor dataframe
    :param target_series: target series
    :param model_uid: model uid
    :param evaluation_list: list of named tuples, which each tuple having the ordering of: the column with the
    predictions, the scoring function callable, and the name of the metric
    :param target: name of the target
    """
    predictions_df = make_full_predictions_dataframe(estimator, model_uid, x_df, target_series)
    run_and_save_evaluation_metrics(predictions_df[target], predictions_df['pred'], model_uid, evaluation_list)
