from sklearn.linear_model import LogisticRegression
import numpy as np

class FixedOneLogisticRegression(LogisticRegression):
    def __init__(self, **kwargs):
        # Ensure no penalty (L2/L1 regularization would change coefficients)
        kwargs['penalty'] = None  
        super().__init__(**kwargs)
    def fit(self, X, y):
        super().fit(X, y)
        n_features = X.shape[1]
        self.coef_ = np.ones((1, n_features))  # For binary classification
        self.intercept_ = np.zeros(1)           # No bias
        return self

