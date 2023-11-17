#!/usr/bin/env python

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

pseudocount = 0.5
# This is some example code for an implementation of the logistic regression with odds ratios estimated by isotonic regressions.
# In the future, we may 
# - implement the L0 and/or L1 norm versions of isotonic regressions
# - better estimate the odds ratios
# - optimize both the isotonic curve and the logistic curve together so the the overall cross-entropy loss is minimized  (maybe with some EM algorithm)

class IsotonicLogisticRegression:
    
    X0 = None
    X1 = None
    irs = [] # regressions
    ivs = [] # predicted values
    orX = None
    lr = LogisticRegression()
    
    def __init__(self):
        self.X0 = None
        self.X1 = None
        self.irs = []
        self.ivs = []
        self.orX = None
        self.lr = LogisticRegression()
        
    def fit(self, X, y):
        for label in y: assert label in [0, 1], F'Label {label} is not binary'
        
        X0 = self.X0 = X[y==0,:]
        X1 = self.X1 = X[y==1,:]
        assert X.shape[1] == X0.shape[1] and X0.shape[1] == X1.shape[1], 'InternalError'
        assert X0.shape[0] > 1, 'At least two negative examples should be provided'
        assert X1.shape[0] > 1, 'At least two positive examples should be provided'
        assert X1.shape[0] < X0.shape[0], 'There should be more positive examples than negative examples'
        
        irs = self.irs = [IsotonicRegression(increasing = 'auto', out_of_bounds = 'clip') for _ in range(X.shape[1])]
        irv = self.ivs = [None for _ in range(X.shape[1])]
        for colidx in range(X.shape[1]):
            x0 = sorted(X0[:,colidx])
            x1 = sorted(X1[:,colidx])
            x1open = [-float('inf')] + x1 + [float('inf')]
            x1to0s = [[] for _ in range(len(x1))]
            min_atleast_idx = 1
            for xp0 in x0:
                while x1open[min_atleast_idx] < xp0: min_atleast_idx += 1
                dist2lower = (xp0 - x1open[min_atleast_idx-1])
                dist2upper = (x1open[min_atleast_idx] - xp0)
                assert dist2lower >= 0
                assert dist2upper >= 0
                if dist2lower + dist2lower < 1e-99:
                    frac2lower = frac2upper = 0.5
                else:
                    frac2lower = (dist2upper / float(dist2lower + dist2upper) if dist2upper < 1e99 else 1)
                    frac2upper = (dist2lower / float(dist2lower + dist2upper) if dist2lower < 1e99 else 1)
                if frac2lower > 1e-99: x1to0s[min_atleast_idx-2].append((xp0, frac2lower))
                if frac2upper > 1e-99: x1to0s[min_atleast_idx-1].append((xp0, frac2upper))
            print(F'colidx={colidx} : x1to0s={x1to0s}')
            def sumup(xs): return sum(x[1] for x in xs)
            raw_1to0_oddsratios = [(1.0 / len(x1)) / (float(sumup(xs) + pseudocount) / (len(x0) + pseudocount * len(x1))) for xs in x1to0s]
            self.ivs[colidx] = irs[colidx].fit(x1 ,raw_1to0_oddsratios)
            print(F'colidx={colidx} : oddsratios={raw_1to0_oddsratios}')
        self.orX = np.array([irs[colidx].predict(X[:,colidx]) for colidx in range(X.shape[1])]).transpose()
        print(self.orX)
        return self.lr.fit(self.orX, y)
    def predict(self, X):
        irs = self.irs
        test_orX = np.array([irs[colidx].predict(X[:,colidx]) for colidx in range(X.shape[1])]).transpose()
        return self.lr.predict(test_orX)
    def predict_proba(self, X):
        irs = self.irs
        test_orX = np.array([irs[colidx].predict(X[:,colidx]) for colidx in range(X.shape[1])]).transpose()
        return self.lr.predict_proba(test_orX)
def test1():
    X = np.array([
        [ 1, 10],
        [ 2, 30],
        [ 5, 60],
        [ 7,100],
        [ 9,150],
        [11,210]
    ])
    y = np.array([1,0,1,0,0,0])
    ilr = IsotonicLogisticRegression()
    ilr.fit(X, y)
    Xtest = np.array([
        [0,0],
        [0,999],
        [6,0],
        [6,999],
        [99,0],
        [99,999],
        [5,30],
        [5,40],
        [5,50],
        [5,60]
    ])
    #print(F'test_pred_labels={ilr.predict(Xtest)}')
    testres = np.concatenate([Xtest, ilr.predict_proba(Xtest)], axis=1)
    print(F'test_X_probas=\n{testres}')
    #print(F'test_out_probas=\n{ilr.predict_proba(Xtest)}')
    
def main():
    pass

if __name__ == '__main__':
    test1()

