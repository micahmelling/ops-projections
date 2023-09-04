from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from helpers.wrangling import drop_features
from modeling.config import CATEGORICAL_FEATURES, FEATURES_TO_DROP
from modeling.embedding import EmbeddingsEncoder


def get_pipeline(model: RegressorMixin or ClassifierMixin) -> Pipeline:
    """
    Creates a scikit-learn modeling pipeline for our modeling problem. In this case, a set of features can be dropped
    per the FEATURES_TO_DROP global defined in modeling.config. A model is then applied.

    :param model: regression or classification model
    :return: scikit-learn pipeline
    """
    pipeline = Pipeline(steps=[
        ('dropper', FunctionTransformer(drop_features, validate=False,
                                        kw_args={
                                            'features_list': FEATURES_TO_DROP
                                        })),
        ('embedder', EmbeddingsEncoder(columns=CATEGORICAL_FEATURES)),
        ('model', model)
        ])
    return pipeline
