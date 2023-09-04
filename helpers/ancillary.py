import os
from typing import Union

import joblib
import pandas as pd
from sklearn.base import ClassifierMixin, RegressorMixin


def make_directories_if_not_exists(directories_list: list) -> None:
    """
    Makes directories in the current working directory if they do not exist.

    :param directories_list: list of directories to create
    """
    for directory in directories_list:
        if not os.path.exists(directory):
            os.makedirs(directory)


def save_cv_scores(df: pd.DataFrame, model_uid: str) -> None:
    """
    Saves cross-validation scores into a model-specific directory.

    :param df: dataframe of cross-validation scores
    :param model_uid: model uid
    """
    save_directory = os.path.join('modeling', 'model_results', model_uid, 'cv_scores')
    make_directories_if_not_exists([save_directory])
    df.to_csv(os.path.join(save_directory, 'cv_scores.csv'), index=False)


def save_model(model: Union[RegressorMixin, ClassifierMixin], model_uid: str,
               model_append_name: str = None, ) -> None:
    """
    Saves a pickled model into a model-specific directory.
    :param model: regression or classification model
    :param model_uid: model uid
    :param model_append_name: name to append to the model name
    """
    model_name = 'model'
    if model_append_name:
        model_name = f'{model_name}_{model_append_name}'
    save_directory = os.path.join('modeling', 'model_results', model_uid, 'model')
    make_directories_if_not_exists([save_directory])
    joblib.dump(model, os.path.join(save_directory, f'{model_name}.pkl'))


def save_modeling_data_in_model_directory(x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame,
                                          y_test: pd.Series, model_uid: str) -> None:
    """
    Saves modeling data into a model-specific directory.

    :param x_train: x train
    :param y_train: y train
    :param x_test: x test
    :param y_test: y test
    :param model_uid: model uid
    """
    save_directory = os.path.join('modeling', 'model_results', model_uid, 'data')
    make_directories_if_not_exists([save_directory])
    joblib.dump(x_train, os.path.join(save_directory, 'x_train.pkl'))
    joblib.dump(y_train, os.path.join(save_directory, 'y_train.pkl'))
    joblib.dump(x_test, os.path.join(save_directory, 'x_test.pkl'))
    joblib.dump(y_test, os.path.join(save_directory, 'y_test.pkl'))

