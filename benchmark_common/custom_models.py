import warnings
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
import numpy as np

def _logit(a, b): return a / (b-a)

class FixedOneLogisticRegression(LogisticRegression):
    def __init__(self, **kwargs):
        # Ensure no penalty (L2/L1 regularization would change coefficients)
        kwargs['penalty'] = None
        kwargs['random_state'] = 0
        super().__init__(max_iter=1, solver='sag', **kwargs)
    def fit(self, X, y):
        super().fit(X, y)
        n_features = X.shape[1]
        self.coef_ = np.ones((1, n_features))  # For binary classification
        self.intercept_ = np.zeros(1)          # No bias
        return self

class FixedZeroLogisticRegression(LogisticRegression):
    def __init__(self, **kwargs):
        # Ensure no penalty (L2/L1 regularization would change coefficients)
        kwargs['penalty'] = None
        kwargs['random_state'] = 0
        super().__init__(max_iter=1, solver='sag', **kwargs)
    def fit(self, X, y):
        super().fit(X, y)
        n_features = X.shape[1]
        self.coef_ = np.zeros((1, n_features))  # For binary classification
        self.intercept_ = np.log(_logit(np.sum(y), len(y)))
        return self

class GroupEffectLogisticRegression(LogisticRegression):
    def __init__(self, group_name='ln_NumTested', **kwargs):
        # Ensure no penalty (L2/L1 regularization would change coefficients)
        kwargs['penalty'] = None
        kwargs['random_state'] = 0
        super().__init__(max_iter=1, solver='sag', **kwargs)
        self.lr = LogisticRegression(solver='liblinear', random_state=0)
        self.group_name = group_name
    def fit(self, X, y):
        idx = -1
        for i, c in enumerate(X.columns):
            if c == self.group_name: idx = i
        X = np.array(X)
        super().fit(X, y)
        self.lr.fit(X[:,[idx]], y)
        n_features = X.shape[1]
        self.coef_ = np.zeros((1, n_features))  # For binary classification
        self.coef_[idx] = self.lr.coef_[0]
        self.intercept_ = self.lr.intercept_
        return self

class HardThresholdClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, columns=[], op=-1, thres=0, **kwargs):
        assert op in [-2,-1,0,1,2], f'The operator {op} is invalid!'
        self.columns = columns
        self.op = op
        self.thres = thres
    def fit(self, X, y, *args, **kwargs):
        if hasattr(X, 'columns'):
            self.fit_columns_ = X.columns
            self.feature_names_in_ = X.columns.tolist()
        else:
            self.fit_columns_ = []
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        return self
    def predict(self, X):
        def bin_operator(a, b):
            if self.op ==-2: return a <  b
            if self.op ==-1: return a <= b
            if self.op == 0: return a == b
            if self.op == 1: return a >= b
            if self.op == 2: return a >  b
        if hasattr(X, 'columns'):
            predict_columns = X.columns
            if len(self.fit_columns_) > 0:
                assert len(self.fit_columns_) == len(predict_columns), F'{len(self.fit_columns_)} == {len(predict_columns)} failed!'
                assert (self.fit_columns_ == predict_columns).all(), F'{self.fit_columns} == {predict_columns} failed!'
        else:
            predict_columns = self.fit_columns_
        X2 = np.array(X)
        for col_idx, col_name in enumerate(predict_columns):
            if col_name in self.columns:
                return np.where(bin_operator(X2[:,col_idx], self.thres), 1, 0)
        warnings.warn(f'The array-like {X} has no column named {self.columns}!')
        return np.zeros(len(X))
    def predict_proba(self, X):
        ret = self.predict(X)
        return np.array([(1-p, p) for p in ret])

