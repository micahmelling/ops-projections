from typing import List, Tuple, Union

import pandas as pd

from data.data import load_appearance_data, load_batting_data
from helpers.wrangling import (create_custom_ts_cv_splits,
                               create_data_splits_on_year, create_x_y_split,
                               prep_appearances_data, prep_batting_data,
                               prep_modeling_data)
from modeling.config import (CV_FOLDS, CV_SCORER, DATA_START_YEAR, MIN_AB,
                             MODEL_EVALUATION_LIST, MODEL_TRAINING_LIST,
                             TARGET, TEST_YEAR, evaluation_named_tuple,
                             model_named_tuple)
from modeling.evaluate import run_omnibus_model_evaluation
from modeling.explain import produce_shap_values_and_plots
from modeling.model import train_model_with_hyperopt
from modeling.pipeline import get_pipeline


def prep_data(data_start_year: int, min_ab: int) -> pd.DataFrame:
    """
    Prepares dataframe for predicting OPS for the next season.

    :param data_start_year: year in which to start the data
    :param min_ab: minimum number of at-bats in a season to qualify
    :return: dataframe for modeling
    """
    batting_df = load_batting_data()
    appearances_df = load_appearance_data()
    batting_df = prep_batting_data(batting_df, min_ab=min_ab)
    appearances_df = prep_appearances_data(appearances_df)
    modeling_df = prep_modeling_data(batting_df, appearances_df, data_start_year=data_start_year)
    modeling_df = modeling_df.dropna()
    return modeling_df


def create_training_testing_data(df: pd.DataFrame, test_start_year: int) -> Tuple[pd.DataFrame]:
    """
    Creates training and testing splits.

    :param df: modeling dataframe
    :param test_start_year: year in which the test set starts
    :return: x_train, y_train, x_test, y_test
    """
    train_df, test_df = create_data_splits_on_year(df, test_start_year)
    x_train, y_train = create_x_y_split(train_df, 'target')
    x_test, y_test = create_x_y_split(test_df, 'target')
    x_train = x_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    x_test = x_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    return x_train, y_train, x_test, y_test


def train_and_evaluate_explain_models(x_train: pd.DataFrame, y_train: Union[pd.Series, pd.DataFrame],
                                      x_test: pd.DataFrame, y_test: Union[pd.Series, pd.DataFrame],
                                      model_training_list: List[model_named_tuple], cv_scoring: str,
                                      data_start_year: int, test_year: int,
                                      model_evaluation_list: List[evaluation_named_tuple], target: str,
                                      cv_folds: int) -> None:
    """
    Trains, evaluates, and explains a series of models to predict OPS for the next season.

    :param x_train: x_train
    :param y_train: y_train
    :param x_test: x_test
    :param y_test: y_test
    :param model_training_list: list of models to training
    :param cv_scoring: cv scoring
    :param data_start_year: year in which to start the data
    :param test_year: year in which the test set starts
    :param model_evaluation_list: list of model evaluation metrics
    :param target: name of the target
    :param cv_folds: number of cv folds
    """
    cv_splits = create_custom_ts_cv_splits(x_train, start_year=data_start_year, end_year=test_year - 1,
                                           cv_folds=cv_folds)
    for model in model_training_list:
        pipe = get_pipeline(model.model)
        best_model = train_model_with_hyperopt(
            estimator=pipe,
            x_train=x_train,
            y_train=y_train,
            model_uid=model.model_name,
            param_space=model.param_space,
            iterations=model.iterations,
            cv_strategy=cv_splits,
            cv_scoring=cv_scoring
        )
        run_omnibus_model_evaluation(
            estimator=pipe,
            x_df=x_test,
            target_series=y_test,
            model_uid=model.model_name,
            evaluation_list=model_evaluation_list,
            target=target
        )
        produce_shap_values_and_plots(
            pipeline=best_model,
            x_df=x_test,
            model_uid=model.model_name
        )


def main(data_start_year: int, min_ab: int, test_start_year: int, model_training_list: List[model_named_tuple],
         cv_scoring: str, model_evaluation_list: List[evaluation_named_tuple], target: str, cv_folds: int) -> None:
    """
    Execution function for our OPS modeling problem

    :param data_start_year: year in which to start the data
    :param min_ab: minimum number of at-bats in a season to qualify
    :param test_start_year: year in which the test set starts
    :param model_training_list: list of models to training
    :param cv_scoring: cv scoring
    :param model_evaluation_list: list of model evaluation metrics
    :param target: name of the target
    :param cv_folds: number of cv folds
    """
    modeling_df = prep_data(data_start_year, min_ab)
    x_train, y_train, x_test, y_test = create_training_testing_data(modeling_df, test_start_year)
    train_and_evaluate_explain_models(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        model_training_list=model_training_list,
        cv_scoring=cv_scoring,
        data_start_year=data_start_year,
        test_year=test_start_year,
        model_evaluation_list=model_evaluation_list,
        target=target,
        cv_folds=cv_folds
    )


if __name__ == "__main__":
    main(
        data_start_year=DATA_START_YEAR,
        min_ab=MIN_AB,
        test_start_year=TEST_YEAR,
        model_training_list=MODEL_TRAINING_LIST,
        cv_scoring=CV_SCORER,
        model_evaluation_list=MODEL_EVALUATION_LIST,
        target=TARGET,
        cv_folds=CV_FOLDS
    )
