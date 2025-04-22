import copy
import os
import sys

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification

script_dir = os.path.dirname(os.path.abspath(__file__))

sys.path.append(F'{script_dir}/../')
import IsotonicLogisticRegression

# Set a seed for reproducibility
np.random.seed(42)

X1, y1 = make_classification(n_samples=10000, n_features=5, n_informative=5, n_redundant=0, random_state=42)

X2, y2 = copy.deepcopy(X1), copy.deepcopy(y1)
x3 = X2[:,-1]

# insert a new feature which is equal to (i.e., 100% depends on and is 100% correlated with) the last feature of X1
X2 = np.insert(X2, -1, x3, axis=1)

log_reg = LogisticRegression(max_iter=1000, random_state=42)

# check that the coefficients before and after adding the new feature do not change, 
# except that the two features that are equal to each other have their coefficients halved

log_reg.fit(X1, y1)
coef_1 = log_reg.coef_
log_reg.fit(X2, y2)
coef_2 = log_reg.coef_
print(F'LogisticRegression coefficients before and after adding the 100% correlated feature (when not using our feature transformation)')
print([coef_1, coef_2])

ilr = IsotonicLogisticRegression.IsotonicLogisticRegression()
X1 = ilr.fit_transform(X1, y1)
X2 = ilr.fit_transform(X2, y2)    
log_reg.fit(X1, y1)
coef_1 = log_reg.coef_
log_reg.fit(X2, y2)
coef_2 = log_reg.coef_
print(F'LogisticRegression coefficients before and after adding the 100% correlated feature (when using our feature transformation)')
print([coef_1, coef_2])

