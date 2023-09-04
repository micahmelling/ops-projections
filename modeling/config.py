from collections import namedtuple

from hyperopt import hp
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             median_absolute_error)
from xgboost import XGBRegressor

FEATURES_TO_DROP = ['year', 'playerID']
CATEGORICAL_FEATURES = ['primary_position']
DATA_START_YEAR = 1970
TEST_YEAR = 2021
MIN_AB  = 200

TARGET = 'target'
CV_SCORER = 'neg_mean_squared_error'
CV_FOLDS = 5


FOREST_PARAM_GRID = {
    'model__max_depth': hp.uniformint('model__max_depth', 3, 16),
    'model__min_samples_leaf': hp.uniform('model__min_samples_leaf', 0.001, 0.01),
    'model__max_features': hp.choice('model__max_features', ['log2', 'sqrt']),
}

XGBOOST_PARAM_GRID = {
    'model__learning_rate': hp.uniform('model__learning_ratee', 0.01, 0.5),
    'model__n_estimators': hp.randint('model__n_estimators', 75, 150),
    'model__max_depth': hp.randint('model__max_depth', 3, 16),
    'model__min_child_weight': hp.uniformint('model__min_child_weight', 2, 16),
}



model_named_tuple = namedtuple('model_config', {'model_name', 'model', 'param_space', 'iterations'})
MODEL_TRAINING_LIST = [
    model_named_tuple(model_name='random_forest', model=RandomForestRegressor(n_estimators=500),
                      param_space=FOREST_PARAM_GRID, iterations=25),
    model_named_tuple(model_name='xgb', model=XGBRegressor(),
                      param_space=XGBOOST_PARAM_GRID, iterations=25),
]


evaluation_named_tuple = namedtuple('model_evaluation', {'scorer_callable', 'metric_name'})
MODEL_EVALUATION_LIST = [
    evaluation_named_tuple(scorer_callable=mean_squared_error, metric_name='mse'),
    evaluation_named_tuple(scorer_callable=median_absolute_error, metric_name='mdae'),
    evaluation_named_tuple(scorer_callable=mean_absolute_error, metric_name='mae'),
]
