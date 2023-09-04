from copy import deepcopy
from typing import Union

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


def subset_based_on_comparison(df: pd.DataFrame, column: str, comparison_value: Union[int, float],
                               greater: bool) -> pd.DataFrame:
    """
    Subsets a dataframe based on a comparison value.

    :param df: dataframe
    :param column: column for comparison
    :param comparison_value: comparison value
    :param greater: if a greater than or equal to comparison; if false, it will be a less than or equal to comparison
    :return: subsetted dataframe
    """
    if greater:
        df = df.loc[df[column] >= comparison_value]
    else:
        df = df.loc[df[column] <= comparison_value]
    return df


def collapse_yearly_appearance_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapses multiple of the same yearID by player into a single row. This occurs when a player plays for multiple
    teams in the same season.

    :param df: appearance dataframe
    :return: collapsed appearance dataframe
    """
    df = df[['yearID', 'playerID', 'G_p', 'G_c', 'G_1b', 'G_2b', 'G_3b', 'G_ss', 'G_lf', 'G_cf', 'G_rf', 'G_of',
             'G_dh']]
    df = df.groupby(['yearID', 'playerID']).agg({
        'G_p': 'sum', 'G_c': 'sum', 'G_1b': 'sum', 'G_2b': 'sum', 'G_3b': 'sum', 'G_ss': 'sum', 'G_lf': 'sum',
        'G_cf': 'sum', 'G_rf': 'sum', 'G_of': 'sum',
        'G_dh': 'sum'
    })
    df = df.reset_index(drop=False)
    return df


def collapse_yearly_batting_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapses yearly batting data into a single row, in cases where an individual played for multiple teams in a season.

    :param df: batting dataframe
    :return: batting dataframe with collapsed rows
    """
    df = df.groupby(['yearID', 'playerID']).agg({
        'AB': 'sum', 'H': 'sum', '2B': 'sum', '3B': 'sum', 'HR': 'sum', 'BB': 'sum', 'HBP': 'sum', 'SF': 'sum'
    })
    df = df.reset_index(drop=False)
    return df


def find_column_with_highest_value_across_row(df: pd.DataFrame, col_list: list, output_col_name: str) -> pd.DataFrame:
    """
    Adds a column with a value that is the highest across a list of columns in a row.

    :param df: pandas dataframe
    :param col_list: list of columns to consider
    :param output_col_name: the output columnn name
    :return: dataframe with column identifying the highest value
    """
    df[output_col_name] = df[col_list].idxmax(axis=1)
    return df


def remove_rows_not_equal_to(df: pd.DataFrame, column: str,
                             comparison_value: Union[float, int, str]) -> pd.DataFrame:
    """
    Removes rows where a column value does not equal some specified value.

    :param df: pandas dataframe
    :param column: column to check
    :param comparison_value: value to check against
    :return: dataframe with rows removed as desired
    """
    return df.loc[df[column] != comparison_value]


def create_grouped_lags(df: pd.DataFrame, feature: str, group_feature: str, n_lags: int) -> pd.DataFrame:
    """
    Creates n_lags number of lag variables by group for the desired feature.

    :param df: dataframe
    :param feature: feature to lag
    :param group_feature: group ID feature
    :param n_lags: number of lags to create
    :return: dataframe with lag features
    """
    lags = [i for i in range(1, n_lags + 1)]
    for lag in lags:
        feature_name = f'{feature}_lag_{lag}'
        df[feature_name] = df.groupby(group_feature)[feature].shift(lag)
        df[feature_name] = df[feature].fillna(value=0)
    return df


def create_group_shift(df: pd.DataFrame, output_feature_name: str, group_feature: str, shift_feature: str,
                       shift_amount: int, drop_resulting_null: bool) -> pd.DataFrame:
    """
    Creates a column of values shifted by group.

    :param df: pandas dataframe
    :param output_feature_name: name of output feature
    :param group_feature: column identifying the feature
    :param shift_feature: feature to shift
    :param shift_amount: the amount of rows to shift; can be positive or negative
    :param drop_resulting_null: whether or not to drop shifts the result in nulls
    :return: dataframe with output_feature_name as a new column
    """
    df[output_feature_name] = df.groupby(group_feature)[shift_feature].shift(shift_amount)
    if drop_resulting_null:
        df = df.loc[~df[output_feature_name].isnull()]
    return df


def prep_batting_data(df: pd.DataFrame, min_ab: int) -> pd.DataFrame:
    """
    Preps batting data via subsetting and collapsing as needed.

    :param df: batting dataframe
    :param min_ab: minimum number of at bats in a season to be considered
    :return: processed batting dataframe
    """
    df = subset_based_on_comparison(df, column='AB', comparison_value=min_ab, greater=True)
    df = collapse_yearly_batting_data(df)
    return df


def prep_appearances_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preps appearances data via subsetting and collapsing as needed.

    :param df: appearances dataframe
    :return: processed appearances dataframe
    """
    df = collapse_yearly_appearance_data(df)
    df = find_column_with_highest_value_across_row(df, col_list=['G_p', 'G_c', 'G_1b', 'G_2b', 'G_3b', 'G_ss', 'G_lf',
                                                                 'G_cf', 'G_rf', 'G_of', 'G_dh'],
                                                   output_col_name='primary_position')
    df = remove_rows_not_equal_to(df, column='primary_position', comparison_value='G_p')
    return df


def calculate_ops(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates OPS per row.

    :param df: dataframe with batting stats
    :return: dataframe with ops column
    """
    df['obp'] = (df['H'] + df['BB'] + df['HBP']) / (df['AB'] + df['BB'] + df['HBP'] + df['SF'])
    df['slg'] = ((df['HR'] * 4 + df['3B'] * 3 + df['2B'] * 2) + (df['H'] - (df['HR'] + df['3B'] + df['2B']))) / df['AB']
    df['ops'] = df['obp'] + df['slg']
    return df


def create_expanding_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tightly-coupled function to calculate expanding OPS metrics: mean, median, max, min, and std. Calculates number
    of seasons played as well. To note, this only considers eligible seasons defined by the foregoing filtering.

    :param df: dataframe with columns for ops, playerID, and yearID
    :return: dataframe with expanding features
    """
    df['career_avg_ops'] = df.groupby('playerID')['ops'].expanding().mean().reset_index(level=0, drop=True)
    df['career_median_ops'] = df.groupby('playerID')['ops'].expanding().median().reset_index(level=0, drop=True)
    df['career_max_ops'] = df.groupby('playerID')['ops'].expanding().max().reset_index(level=0, drop=True)
    df['career_min_ops'] = df.groupby('playerID')['ops'].expanding().min().reset_index(level=0, drop=True)
    df['career_std_ops'] = df.groupby('playerID')['ops'].expanding().std().reset_index(level=0, drop=True)
    df['career_std_ops'] = df['career_std_ops'].fillna(value=0)
    df['season_count'] = df.groupby('playerID')['yearID'].expanding().count().reset_index(level=0, drop=True)
    return df


def prep_modeling_data(batting_df: pd.DataFrame, appearances_df: pd.DataFrame, data_start_year: int) -> pd.DataFrame:
    """
    Prepares the necessary modeling dataframe for our problem.

    :param batting_df: batting dataframe
    :param appearances_df: appearances dataframe
    :param data_start_year: year we want to start considering data
    :return: modeling dataframe
    """
    df = pd.merge(batting_df, appearances_df, how='inner', on=['yearID', 'playerID'])
    df = calculate_ops(df)
    df = df[['playerID', 'yearID', 'ops', 'primary_position']]
    df = df.sort_values(by=['yearID'], ascending=True)
    df = create_grouped_lags(df, feature='ops', group_feature='playerID', n_lags=3)
    df = create_expanding_features(df)
    df = create_group_shift(df, output_feature_name='target', group_feature='playerID', shift_feature='ops',
                            shift_amount=-1, drop_resulting_null=True)
    df = subset_based_on_comparison(df, column='yearID', comparison_value=data_start_year, greater=True)
    return df
    

def drop_features(df: Union[pd.DataFrame, np.ndarray], features_list: list) -> pd.DataFrame:
    """
    Drops features (columns) from a dataframe.

    :param df: pandas dataframe
    :param features_list: list of features (columns) to drop
    :return: dataframe with dropped columns
    """
    df = df.drop(features_list, axis=1, errors='ignore')
    return df


def create_data_splits_on_year(df: pd.DataFrame, test_year_start: int) -> tuple:
    """
    Creates two splits of data: training and testing for model training and evaluation based on a year-wise split.

    :param df: pandas dataframe on which we can subset based on a year column
    :param test_year_start: year to start the testing data
    :return: training dataframe and testing dataframe
    """
    train_df = df.loc[df['yearID'] < test_year_start]
    test_df = df.loc[df['yearID'] >= test_year_start]
    return train_df, test_df


def create_x_y_split(df: pd.DataFrame, target: str) -> tuple:
    """
    Creates an x-y split for modeling.

    :param df: dataframe of modeling data
    :param target: name of the target column
    :return: x predictor dataframe and y target series
    """
    y_series = df[target]
    x_df = df.drop(target, axis=1)
    return x_df, y_series


def create_custom_ts_cv_splits(df: pd.DataFrame, start_year: int, end_year: int, cv_folds: int) -> list:
    """
    Creates a set of custom cross validation splits based on year. The function takes a start year and an end year along
    with a number of cv folds. Based on the number of years between the provided years, it will create an equal number
    of array splits based on cv folds. The folds are arranged as a time-series cross validation problem. That is,
    in the first split, the first split is the training data and the second split is the testing data. In the second
    split, the first two splits are the training data, and the third split is the testing data. And so on.

    :param df: pandas dataframe of training data
    :param start_year: start year of the cross validation folds
    :param end_year: end year of the cross validation folds
    :param cv_folds: number of cv folds
    :return: list of tuples, with each tuple containing two items - the first is the index of the training observations
    and the second is the index of the testing observations
    """
    cv_splits = []
    years = list(np.arange(start_year, end_year + 1, 1))
    year_splits = np.array_split(years, cv_folds)
    for n, year_split in enumerate(year_splits):
        if n != cv_folds - 1:
            train_ids = year_splits[:n + 1]
            train_ids = np.concatenate(train_ids)
            test_ids = year_splits[n + 1]
            train_indices = df.loc[df['yearID'].isin(train_ids)].index.values.astype(int)
            test_indices = df.loc[df['yearID'].isin(test_ids)].index.values.astype(int)
            cv_splits.append((train_indices, test_indices))
    return cv_splits


def transform_data_with_pipeline(pipeline: Pipeline, x_df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms x_df with the pre-processing steps defined in the pipeline.
    :param pipeline: scikit-learn pipeline
    :param x_df: x predictor dataframe for transformation
    :return: transformed x dataframe
    """
    pipeline_ = deepcopy(pipeline)
    pipeline_.steps.pop(len(pipeline_) - 1)
    x_df = pipeline_.transform(x_df)
    return x_df
