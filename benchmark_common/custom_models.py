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
        self.intercept_ = np.zeros(1)           # No bias
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

