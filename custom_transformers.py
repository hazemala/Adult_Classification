# custom_transformers.py

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class LogTransfomer(BaseEstimator, TransformerMixin):
    def __init__(self, col):
        self.col = col

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X, y=None):
        assert self.n_features_in_ == X.shape[1]
        return np.log1p(X)

    def get_feature_names_out(self, X=None, y=None):
        return self.col

class LogTransfomer_0(BaseEstimator, TransformerMixin):
    def __init__(self, col):
        self.col = col

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X, y=None):
        assert self.n_features_in_ == X.shape[1]
        return np.sign(X) * np.log1p(np.abs(X))

    def get_feature_names_out(self, input_features=None):
        return self.col

class Handle_Ub_Lb(BaseEstimator, TransformerMixin):
    def __init__(self, col):
        self.col = col

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        q1 = np.percentile(X, 25)
        q2 = np.percentile(X, 50)
        q3 = np.percentile(X, 75)
        iqr = q3 - q1
        self.ub_train = q3 + 1.5 * iqr
        self.lb_train = q1 - 1.5 * iqr
        return self

    def transform(self, X, y=None):
        assert self.n_features_in_ == X.shape[1]
        X[X > self.ub_train] = self.ub_train
        X[X < self.lb_train] = self.lb_train
        return X

    def get_feature_names_out(self, X=None, y=None):
        return self.col
