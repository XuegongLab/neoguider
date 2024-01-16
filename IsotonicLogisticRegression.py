#!/usr/bin/env python
import collections,bisect,logging,math,pprint
import numpy as np

from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

# sklearn.calibration.CalibratedClassifierCV(estimator=None, *, method='sigmoid', cv=KFold(n_splits=5, shuffle=True, random_state=1), n_jobs=-1, ensemble=True)
# sklearn.model_selection.KFold(n_splits=5, *, shuffle=False, random_state=None)

# This is some example code for an implementation of the logistic regression with odds ratios estimated by isotonic regressions.
# In the future, we may 
# - implement the L0 and/or L1 norm versions of isotonic regressions
# - optimize both the isotonic curve and the logistic curve together so the the overall cross-entropy loss is minimized  (maybe with some EM algorithm)
# - DONE: better estimate the odds ratios (with the centered isotonic regression at https://arxiv.org/pdf/1701.05964.pdf)

class IsotonicLogisticRegression:
    
    def __init__(self, excluded_cols = [], pseudocount=0.5, penalty=None, n_splits=5, random_state=1, cccv_n_jobs=-1, **kwargs):
        """ Initialize """
        self.X0 = None
        self.X1 = None
        self.raw_log_odds = []
        self.ixs = []
        self.irs = []
        self.ivs = []
        self.ixs2 = []
        self.irs2 = []
        self.ivs2 = []
        self.logORX = None
        self.excluded_cols = excluded_cols
        self.logr = LogisticRegression(penalty=penalty, **kwargs)
        self.cccv = CalibratedClassifierCV(estimator=self.logr, method='isotonic', 
                cv=KFold(n_splits=n_splits, shuffle=True, random_state=random_state), n_jobs=cccv_n_jobs, ensemble=True)
        self.pseudocount = pseudocount
    
    def get_params(self):
        """ Recursively get the params of this model """
        ret = [self.logORX, self.logr.get_params()]
        for ir in self.irs: ret.append(ir.get_params())
        return ret
    
    def get_info(self):
        """ Recursively get the fitted params of this model """
        logr = self.logr
        logr_info = [logr.classes_, logr.coef_, logr.intercept_, logr.n_features_in_, logr.n_iter_]
        isor_info = []
        for i in range(len(self.irs)):
            ir = self.irs[i]
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

    def encode(self, X1, y1, prior_func='constant', prior_strength=2):
        """
            X1: categorical features
            y1: binary response
            return: numerically encoded features using the WOE (weight of evidence) encoding (https://letsdatascience.com/target-encoding/)
        """
        assert prior_func in ['constant', 'sqrt'], F'The prior_func param {prior_func} is invalid!'
        X = np.array(X1)
        y = np.array(y1)
        self._assert_input(X, y, False)
        X0 = self.X0 = X[y==0,:]
        X1 = self.X1 = X[y==1,:]
        baseratio = (float(len(X1)) / float(len(X0)))
        logbase = np.log(baseratio)
        prior_w = prior_strength * self.pseudocount
        if prior_func == 'sqrt': prior_w = prior_strength * self.pseudocount * (float(len(X1))**0.5)
        ret = []
        for colidx in range(X.shape[1]):
            elements = np.unique(X[:,colidx])
            x0counter = collections.Counter(sorted(X0[:,colidx]))
            x1counter = collections.Counter(sorted(X1[:,colidx]))
            ele2or = {ele : ((x1counter[ele] + prior_w) / (x0counter[ele] + prior_w / baseratio)) for ele in elements}
            ret.append([(np.log(ele2or[ele]) - logbase) for ele in X[:,colidx]])
        return np.array(ret).transpose()
    
    def encode1d(self, x, y, **kwargs):
        """ self.encode with 1D x and returning a dictionary mapping categories to numbers """
        logging.debug(x)
        logging.debug(y)
        vals = self.encode([[v] for v in x], y, **kwargs)
        ret = {}
        for v1, v2 in zip(x, vals): 
            assert len(v2) == 1
            ret[v1] = v2[0]
        return ret
    
    def fit(self, X1, y1, is_centered=True, **kwargs):
        """ scikit-learn fit
            is_centered : using centered isotonic regression or not
        """
        inX, exX = self._split(X1)
        X = np.array(inX)
        y = np.array(y1)
        self._assert_input(X, y)
        X0 = self.X0 = X[y==0,:]
        X1 = self.X1 = X[y==1,:]
        raw_log_odds = self.raw_log_odds = [None for _ in range(X.shape[1])]
        ixs  = self.ixs  = [None for _ in range(X.shape[1])]
        irs  = self.irs  = [IsotonicRegression(increasing = 'auto', out_of_bounds = 'clip') for _ in range(X.shape[1])]
        ivs  = self.ivs  = [None for _ in range(X.shape[1])]
        ixs2 = self.ixs2 = [None for _ in range(X.shape[1])]
        irs2 = self.irs2 = [IsotonicRegression(increasing = 'auto', out_of_bounds = 'clip') for _ in range(X.shape[1])]
        ivs2 = self.ivs2 = [None for _ in range(X.shape[1])]
        for colidx in range(X.shape[1]):
            landmark_x_vals = [-1e99] + sorted(list(set(X1[:,colidx]))) + [1e99]
            idxto0s = [[] for _ in range(len(landmark_x_vals))]
            idxto1s = [[] for _ in range(len(landmark_x_vals))]
            for xi, yi in sorted(zip(X[:,colidx], y)):
                idx = bisect.bisect_left(landmark_x_vals, xi)
                assert idx > 0
                assert idx < len(landmark_x_vals)
                lower = landmark_x_vals[idx-1]
                upper = landmark_x_vals[idx]
                sim2lower = (xi - lower) # / (upper - lower)
                sim2upper = (upper - xi) # / (upper - lower)
                assert sim2lower >= 0, F'sim2lower {sim2lower} >= 0 failed for xi={xi} yi={yi} colidx={colidx} idx={idx}'
                assert sim2upper >= 0, F'sim2upper {sim2upper} >= 0 failed for xi={xi} yi={yi} colidx={colidx} idx={idx}'
                frac2lower = sim2upper / (sim2lower + sim2upper)
                frac2upper = sim2lower / (sim2lower + sim2upper)
                if 0 == yi:
                    idxto0s[idx-1].append((xi, frac2lower))
                    idxto0s[idx-0].append((xi, frac2upper))
                else:
                    idxto1s[idx-1].append((xi, frac2lower))
                    idxto1s[idx-0].append((xi, frac2upper))
            
            def sumup(xs): return sum(x[1] for x in xs)
            idxto0sum = [ sumup(idxto0s[i]) for i in range(len(landmark_x_vals)) ]
            idxto1sum = [ sumup(idxto1s[i]) for i in range(len(landmark_x_vals)) ]
            idxs = [ i for i in range(len(landmark_x_vals)) if (idxto0sum[i] > 0.5 and idxto1sum[i] > 0.5) ]
            '''
            # This part cannot handle duplicated feature values of positive examples
            x0 = sorted(X0[:,colidx])
            x1 = sorted(X1[:,colidx])
            x1open = [-float('inf')] + x1 + [float('inf')]
            x1to0s = [[] for _ in range(len(x1))]
            min_atleast_idx = 1
            for xp0 in x0:
                while x1open[min_atleast_idx] < xp0: min_atleast_idx += 1
                dist2lower = (xp0 - x1open[min_atleast_idx-1])
                dist2upper = (x1open[min_atleast_idx] - xp0)
                assert dist2lower >= 0, F'dist2lower {dist2lower} >= 0 failed for xp0={xp0}'
                assert dist2upper >= 0, F'dist2upper {dist2lower} >= 0 failed for xp0={xp0}'
                if dist2lower + dist2lower < 1e-99:
                    frac2lower = frac2upper = 0.5
                else:
                    frac2lower = (dist2upper / float(dist2lower + dist2upper) if dist2upper < 1e99 else 1)
                    frac2upper = (dist2lower / float(dist2lower + dist2upper) if dist2lower < 1e99 else 1)
                if frac2lower > 1e-99: x1to0s[min_atleast_idx-2].append((xp0, frac2lower))
                if frac2upper > 1e-99: x1to0s[min_atleast_idx-1].append((xp0, frac2upper))
            logging.debug(F'colidx={colidx} : x1to0s={x1to0s}')
            raw_1to0_oddsratios = [(1.0 / len(x1)) / (float(sumup(xs) + self.pseudocount) / (len(x0) + self.pseudocount * len(x1))) for xs in x1to0s]
            fitted_x_vals = x1
            '''
            idxto0sum2 = [idxto0sum[i] for i in idxs]
            num0s = sum(idxto0sum2)
            idxto1sum2 = [idxto1sum[i] for i in idxs]
            num1s = sum(idxto1sum2)
            fitted_x_vals = [landmark_x_vals[i] for i in idxs]
            raw_1to0_oddsratios = [
                    ( ( (sumup(idxto1s[i])                   ) / (num1s) ) / 
                      ( (sumup(idxto0s[i]) + self.pseudocount) / (num0s + self.pseudocount * num1s / sumup(idxto1s[i])) ) )
                    for i in idxs]
            self.raw_log_odds[colidx] = np.log(raw_1to0_oddsratios)
            self.ixs[colidx] = fitted_x_vals
            self.ivs[colidx] = self.irs[colidx].fit_transform(fitted_x_vals, np.log(raw_1to0_oddsratios), sample_weight = idxto1sum2)
            FV_W_LOR_x1_x0_list = list(zip(fitted_x_vals, idxto1sum2, np.log(raw_1to0_oddsratios), self.irs[colidx].predict(fitted_x_vals),
                   [sumup(idxto1s[i]) for i in idxs], [sumup(idxto0s[i]) for i in idxs]))
            logging.debug(F'colidx={colidx} : len(FV_W_obsLOR_expLOR_x1_x0_list)={len(FV_W_LOR_x1_x0_list)}')
            for fv_record in FV_W_LOR_x1_x0_list:
                vs = '\t'.join(F'{v:.4G}'.rjust(10) for v in fv_record)
                logging.debug(F'  fv_rec = {vs}')
            if is_centered:
                x2, y2 = self._center(fitted_x_vals, self.irs[colidx].predict(fitted_x_vals)) # fitted_x_vals instead of x1
                self.ixs2[colidx] = x2
                self.ivs2[colidx] = self.irs2[colidx].fit_transform(x2, y2)
            
        self.logORX = self._transform(X) # np.array([self.irs[colidx].predict((X[:,colidx])) for colidx in range(X.shape[1])]).transpose()
        logging.debug(self.logORX)
        #self.cccv = CalibratedClassifierCV(estimator=self.logr, method='isotonic', 
        #        cv=KFold(n_splits=min([sum(y), 5]), shuffle=True, random_state=1), n_jobs=-1, ensemble=True)
        #self.cccv.fit(self.logORX, y)
        self.logr.fit(np.hstack([self.logORX, exX]), y, **kwargs)
        return self
    
    def get_density_estimated_log_odds(self):
        return self.raw_log_odds
    def get_isotonic_X(self):
        return self.ixs
    def get_isotonic_log_odds(self):
        return self.ivs
    def get_centered_isotonic_X(self):
        return self.ixs2
    def get_centered_isotonic_log_odds(self):
        return self.ivs2

    def _transform(self, X):
        return np.array([self.irs[colidx].predict((X[:,colidx])) for colidx in range(X.shape[1])]).transpose()
    
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
        test_orX = self.transform(inX) # np.array([self.irs[colidx].predict(X[:,colidx]) for colidx in range(X.shape[1])]).transpose()
        return np.hstack([test_orX, exX])
    
    def predict(self, X1):
        """ scikit-learn predict using logistic regression built on top of isotonic scaler """
        allfeatures = self._extract_features(X1)
        return self.logr.predict(allfeatures)
        #return self.cccv.predict(test_orX)

    def predict_proba(self, X1):
        """ scikit-learn predict_proba using logistic regression built on top of isotonic scaler """
        allfeatures = self._extract_features(X1)
        return self.logr.predict_proba(allfeatures)
        #return self.cccv.predict_proba(test_orX)
    
    def get_ivs(): return self.ivs

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
        [19, 550, 0]
    ])
    X = pd.DataFrame(X, columns = ['col1', 'col2', 'col3'])
    y = np.array([1,1,0,1,1,0,0,0,0,0])
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

def test_encode_and_encode1d():
    pp = pprint.PrettyPrinter(indent=4)
    logging.basicConfig(format='test_encode_and_encode1d %(asctime)s - %(message)s', level=logging.DEBUG)
    X = np.array([
        [ 1, 'A'],
        [ 1, 'B'],
        [ 1, 'B'],
        [ 1, 'B'],
        [ 1, 'B'],
        [ 2, 'C'],
        [ 2, 'C'],
        [ 2, 'C'],
        [ 2, 'C'],
        [ 2, 'D']
    ])
    y = np.array([1,1,0,1,1,0,0,0,0,0])
    ilr = IsotonicLogisticRegression()
    X2 = ilr.encode(X, y)
    logging.info(F'Encoded={X2} ')
    v3 = ilr.encode1d([v[1] for v in X], y)
    logging.info(F'Encode1d={v3} ')

if __name__ == '__main__':
    test_fit_and_predict_proba()
    test_encode_and_encode1d()

