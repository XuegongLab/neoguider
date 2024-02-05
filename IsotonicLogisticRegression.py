#!/usr/bin/env python

import collections,logging,math,pprint,random
import numpy as np
import scipy
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

# This is some example code for an implementation of the logistic regression with odds ratios estimated by isotonic regressions.
# In the future, we may:
#   1. optimize both the isotonic curve and the logistic curve together so the the overall cross-entropy loss is minimized
#   2. perform additional isotonic regression on the sum of each pair of fitted isotonic functions

class IsotonicLogisticRegression:
    
    def __init__(self, excluded_cols = [], pseudocount=0.5, **kwargs):
        """ Initialize """
        self.X0 = None
        self.X1 = None
        self.prevalence_odds = -1
        self.raw_log_odds = []
        self.irs0 = []
        self.ixs1 = []
        self.irs1 = []
        self.ivs1 = []
        self.ixs2 = []
        self.irs2 = []
        self.ivs2 = []
        self.logORX = None
        self.excluded_cols = excluded_cols
        self.logr = LogisticRegression(**kwargs)
        # Probability can be calibrated with:
        # n_splits=5, random_state=1, cccv_n_jobs=-1,
        # sklearn.calibration.CalibratedClassifierCV(estimator=None, *, method='sigmoid', cv=KFold(n_splits=5, shuffle=True, random_state=1), n_jobs=-1, ensemble=True)
        # sklearn.model_selection.KFold(n_splits=5, *, shuffle=False, random_state=None)
        # self.cccv = CalibratedClassifierCV(estimator=self.logr, method='isotonic', cv=KFold(n_splits=n_splits, shuffle=True, random_state=random_state), n_jobs=cccv_n_jobs, ensemble=True)
        # We tested calibration and confirmed that LogisticRegression is already well-calibrated, which is as expected from theory.
        self.pseudocount = pseudocount
    
    def _abbrevshow(alist, anum=5):
        if len(alist) <= anum*2: return [alist]
        else: return [alist[0:anum], alist[(len(alist)-anum):len(alist)]]
         
    def get_params(self):
        """ Recursively get the params of this model """
        ret = [self.logORX, self.logr.get_params()]
        for ir in self.irs0: ret.append(ir.get_params())
        return ret
    
    def get_info(self):
        """ Recursively get the fitted params of this model """
        logr = self.logr
        logr_info = [logr.classes_, logr.coef_, logr.intercept_, logr.n_features_in_, logr.n_iter_]
        isor_info = []
        for i in range(len(self.irs0)):
            ir = self.irs0[i]
            isor_info.append([ir.X_min_, ir.X_max_, ir.X_thresholds_, ir.y_thresholds_, ir.f_, ir.increasing_])
        return [logr_info, isor_info]
    
    def _split(self, X, is_already_splitted = False):
        X1 = np.array(X)
        if not is_already_splitted:
            excluded_cols = set(self.excluded_cols)
            ex_colidxs = []            
            for colidx in range(X1.shape[1]):
                if colidx in self.excluded_cols:
                    ex_colidxs.append(i)
            if hasattr(X, 'columns'):
                for colidx, colname in enumerate(X.columns):
                    if colname in self.excluded_cols:
                        ex_colidxs.append(colidx)
            self.ex_colidxs = sorted(list(set(ex_colidxs)))
        ex_colidxs = self.ex_colidxs
        in_colidxs = [colidx for colidx in range(X1.shape[1]) if (not colidx in ex_colidxs)]
        if hasattr(X, 'iloc'):
            return X.iloc[:,in_colidxs], X.iloc[:,ex_colidxs]
        else:
            return X[:,in_colidxs], X[:,ex_colidxs]
    
    def _center(self, x, y, epsilon = 1e-6):
        """ Implement the centering step of the centered isotonic regression at https://arxiv.org/pdf/1701.05964.pdf """
        assert len(x) == len(y)
        L = len(x)
        x2 = []
        y2 = []
        idx1 = 0
        idx2 = 0
        while idx2 < L:
            while idx2 < L and abs(y[idx1] - y[idx2]) < epsilon: idx2 += 1
            xsum = 0
            ysum = 0
            for i in range(idx1, idx2):
                xsum += x[i]
                ysum += y[i]
            x2.append(xsum / float(idx2 - idx1))
            y2.append(ysum / float(idx2 - idx1))
            idx1 = idx2
        return (x2, y2)
    
    def _assert_input(self, X, y, is_num_asserted=True):
        for label in y: assert label in [0, 1], F'Label {label} is not binary'
        for rowit in range(X.shape[0]):
            for colit in range(X.shape[1]):
                if is_num_asserted: 
                    assert not math.isnan(X[rowit][colit]), F'Nan value encountered in row {rowit} col {colit} ({X[rowit]})'
                    assert (-1e50 < X[rowit][colit]), F'Number too small (< -1e50)  at row {rowit} col {colit} ({X[rowit]})'
                    assert ( 1e50 > X[rowit][colit]), F'Number too large (>  1e50)  at row {rowit} col {colit} ({X[rowit]})'
        X0 = self.X0 = X[y==0,:]
        X1 = self.X1 = X[y==1,:]
        assert X.shape[1] == X0.shape[1] and X0.shape[1] == X1.shape[1], 'InternalError'
        assert X0.shape[0] > 1, 'At least two negative examples should be provided'
        assert X1.shape[0] > 1, 'At least two positive examples should be provided'
        assert X1.shape[0] < X0.shape[0], 'The number of positive examples should be less than the number of negative examples'
    
    def total_order(self, xs, ys, random_state=0):
        ret = []
        local_rand = random.Random(random_state)
        zs = list(range(len(xs)))
        shuf_ret = local_rand.shuffle(zs)
        assert shuf_ret == None
        xzylist = sorted(zip(xs, zs, ys))
        xsetlist = sorted(set(xs))
        xsetlist_idx = 0
        for x, z, y in xzylist:
            while xsetlist[xsetlist_idx] < x: xsetlist_idx += 1
            assert xsetlist[xsetlist_idx] == x
            
            xlower = (xsetlist[xsetlist_idx-1] if (xsetlist_idx-1 >= 0)            
                    else xsetlist[ 0] - (xsetlist[ 1] - xsetlist[ 0]))
            
            xupper = (xsetlist[xsetlist_idx+1] if (xsetlist_idx+1 < len(xsetlist)) 
                    else xsetlist[-1] + (xsetlist[-1] - xsetlist[-2]))
            
            xnew = xlower + ((xupper - xlower) * z + 0.5) / len(xzylist)
            ret.append((xnew, y))
        return ret
    def partition(self, xyarr):
        contigs = []
        prev_y = None
        for x,y in xyarr:
            if y == prev_y:
                contig.append((x,y))
            else:
                if prev_y != None: contigs.append(contig)
                contig = [(x,y)]
                prev_y = y
        contigs.append(contig)
        return contigs
        
    def fit(self, X1, y1, is_centered=True, random_state=0, **kwargs):
        """ scikit-learn fit
            is_centered : using centered isotonic regression or not
        """
        #def triangular_kernel(val, mid, lo, hi): return max((0, ((val-lo) / (mid-lo) if (val < mid) else (hi-val) / (hi-mid))))
        #def heaviside_rectangular_kernel(val, mid, lo, hi): return (1 if (lo < val and val < hi) else (0.5 if (val == lo or val == hi) else 0))
        def powermean(arr, p=1): return 1.0/len(arr) * (sum(ele**p for ele in arr))**p
        inX, exX = self._split(X1)
        X = np.array(inX)
        y = np.array(y1)
        self._assert_input(X, y)
        X0 = self.X0 = X[y==0,:]
        X1 = self.X1 = X[y==1,:]
        raw_log_odds = self.raw_log_odds = [None for _ in range(X.shape[1])]
        irs0 = self.irs0 = [None for _ in range(X.shape[1])]
        ixs1 = self.ixs1 = [None for _ in range(X.shape[1])]
        irs1 = self.irs1 = [IsotonicRegression(increasing = 'auto', out_of_bounds = 'clip') for _ in range(X.shape[1])]
        ivs1 = self.ivs1 = [None for _ in range(X.shape[1])]
        ixs2 = self.ixs2 = [None for _ in range(X.shape[1])]
        irs2 = self.irs2 = [IsotonicRegression(increasing = 'auto', out_of_bounds = 'clip') for _ in range(X.shape[1])]
        ivs2 = self.ivs2 = [None for _ in range(X.shape[1])]
        self.prevalence_odds = (len(X1) / float(len(X0)))
        for colidx in range(X.shape[1]):
            x = X[:,colidx]
            xylist = (self.total_order(x, y, random_state) if (len(set(x)) > 1) else zip(x,y))
            xylistlist = self.partition(xylist)
            xcenters = []
            xodds = []
            prev_ylabel = None
            for i, curr_xylist in enumerate(xylistlist):
                prev_len = (len(xylistlist[i-1]) if (i-1 >= 0)               else len(xylistlist[i+1]))
                next_len = (len(xylistlist[i+1]) if (i+1 <  len(xylistlist)) else len(xylistlist[i-1]))
                pre2_len = (len(xylistlist[i-2]) if (i-2 >= 0)               else len(curr_xylist))
                nex2_len = (len(xylistlist[i+2]) if (i+2 <  len(xylistlist)) else len(curr_xylist))
                assert prev_len > 0
                assert next_len > 0
                assert len(curr_xylist) > 0
                yset = set(xy[1] for xy in curr_xylist)
                assert len(yset) == 1
                ylabel = list(yset)[0]
                assert prev_ylabel != ylabel
                xcenter = sum(xy[0] for xy in curr_xylist) / float(len(curr_xylist))
                #if len(curr_xylist) * 2 < (prev_len + next_len):
                #    odds = len(curr_xylist) / powermean((prev_len, next_len))
                #else:
                odds = (powermean((len(curr_xylist), powermean((pre2_len, nex2_len)))) + self.pseudocount) / (powermean((prev_len, next_len)) + self.pseudocount)
                xcenters.append(xcenter)
                xodds.append((odds) if (ylabel == 1) else (1/odds))
                prev_ylabel = ylabel
            raw_log_odds = np.log(xodds)
            center_log_odds = np.log(self.prevalence_odds)
            relative_log_odds = raw_log_odds - center_log_odds
            self.raw_log_odds[colidx] = raw_log_odds
            self.ixs1[colidx] = xcenters
            self.ivs1[colidx] = 1*center_log_odds + self.irs1[colidx].fit_transform(xcenters, relative_log_odds)
            self.irs0[colidx] = self.irs1[colidx]
            if is_centered:
                x2, y2 = self._center(xcenters, self.irs1[colidx].predict(xcenters))
                self.ixs2[colidx] = x2
                self.ivs2[colidx] = 1*center_log_odds + self.irs2[colidx].fit_transform(x2, y2)
                self.irs0[colidx] = self.irs2[colidx]
        log_ratios = self._transform(X)
        self.logr.fit(np.hstack([log_ratios, exX]), y, **kwargs)
        return self
    
    def get_odds_offset(self):
        return self.prevalence_odds
    def get_density_estimated_X(self):
        return self.ixs1
    def get_density_estimated_log_odds(self):
        return self.raw_log_odds
    def get_isotonic_X(self):
        return self.ixs1
    def get_isotonic_log_odds(self):
        return self.ivs1
    def get_centered_isotonic_X(self):
        return self.ixs2
    def get_centered_isotonic_log_odds(self):
        return self.ivs2

    def _transform(self, X):
        return np.array([self.irs0[colidx].predict((X[:,colidx])) for colidx in range(X.shape[1])]).transpose()
    
    def transform(self, X1):
        """ scikit-learn transform """
        X = np.array(X1)
        return self._transform(X)

    def fit_transform(self, X1, y1):
        """ scikit-learn fit_transform """
        self.fit(X1, y1)
        return self.transform(X1)
    
    def _extract_features(self, X1):
        inX, exX = self._split(X1, True)
        test_orX = self.transform(inX)
        return np.hstack([test_orX, exX])
    
    def predict(self, X1):
        """ scikit-learn predict using logistic regression built on top of isotonic scaler """
        allfeatures = self._extract_features(X1)
        return self.logr.predict(allfeatures)

    def predict_proba(self, X1):
        """ scikit-learn predict_proba using logistic regression built on top of isotonic scaler """
        allfeatures = self._extract_features(X1)
        return self.logr.predict_proba(allfeatures)

def test_fit_and_predict_proba():
    import pandas as pd
    
    pp = pprint.PrettyPrinter(indent=4)
    logging.basicConfig(format='test_fit_and_predict_proba %(asctime)s - %(message)s', level=logging.DEBUG)
    X = np.array([
        [ 1,  10, 0],
        [ 3,  30, 0],
        [ 5,  60, 0],
        [ 7, 100, 0],
        [ 9, 150, 0],
        [11, 210, 0],
        [13, 280, 0],
        [15, 360, 0],
        [17, 450, 0],
        [19, 550, 0],
        [21, 660, 0],
        [23, 780, 0],
        [25, 910, 0],
        [27,1050, 0],
        [29,1300, 0],
    ])
    X = pd.DataFrame(X, columns = ['col1', 'col2', 'col3'])
    y = np.array([1,1,0,1,1,0,0,0,0,0,1,0,0,0,0])
    ilr = IsotonicLogisticRegression(excluded_cols = ['col3'])
    ilr.fit(X, y)
    Xtest = np.array([
        [0,    0, 0],
        [0,  999, 0],
        [6,    0, 0],
        [6,  999, 0],
        [99,   0, 0],
        [99, 999, 0],
        [5,   30, 0],
        [5,   40, 0],
        [5,   50, 0],
        [5,   60, 0]
    ])
    testres = np.concatenate([Xtest, ilr.predict_proba(Xtest)], axis=1)
    print(F'test_X_probas=\n{testres}')
    pp.pprint(ilr.get_info())
    pp.pprint(np.hstack((X,y[:,None])))

if __name__ == '__main__':
    test_fit_and_predict_proba()

