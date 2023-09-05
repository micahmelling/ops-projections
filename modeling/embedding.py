from copy import deepcopy

import pandas as pd
from keras import models
from keras.layers import Dense, Embedding, Flatten
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, StandardScaler


class EmbeddingsEncoder(BaseEstimator, TransformerMixin):
    """
    Encodes categorical features as a numeric embedding space.
    """
    def __init__(self, columns, epochs=5, batch_size=3):
        self.mapping_dict = {}
        self.columns = columns
        self.epochs = epochs
        self.batch_size = batch_size

    # this assumes category levels are unique across columns. if this assumotion does not hold, the mapping_dict can be adjusted to handle
    def fit(self, X, Y):
        for col in self.columns:
            le = LabelEncoder()
            X_ = deepcopy(X)
            Y_ = deepcopy(Y)
            Y_ = Y_.values
            Y_ = StandardScaler().fit_transform(Y_.reshape(-1, 1))
            X_[col] = le.fit_transform(X_[col])

            embedding_size = 1
            input_dim = len(X_[col].unique())

            model = models.Sequential()
            model.add(Embedding(input_dim=input_dim, output_dim=embedding_size, input_length=1, name="embedding"))
            model.add(Flatten())
            model.add(Dense(50, activation="relu"))
            model.add(Dense(15, activation="relu"))
            model.add(Dense(1))
            model.compile(loss="mse", optimizer="adam", metrics=["accuracy"])
            model.fit(x=X_[[col]].values, y=Y_, epochs=self.epochs, batch_size=self.batch_size)

            layer = model.get_layer('embedding')
            output_embeddings = layer.get_weights()

            output_embeddings_df = pd.DataFrame(output_embeddings[0])
            output_embeddings_df = output_embeddings_df.reset_index()
            output_embeddings_df.columns = [col, 'embedding']
            output_embeddings_df[col] = le.inverse_transform(output_embeddings_df[col])
            feature_dict = dict(zip(output_embeddings_df[col], output_embeddings_df['embedding']))
            self.mapping_dict.update(feature_dict)
        return self

    def transform(self, X, Y=None):
        for col in self.columns:    
            X[col] = X[col].map(self.mapping_dict)
        return X
