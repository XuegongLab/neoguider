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

#logging.basicConfig(level = logging.DEBUG)

def abbrevshow(alist, anum=5):
    if len(alist) <= anum*2: return [alist]
    else: return [alist[0:anum], alist[(len(alist)-anum):len(alist)]]

class IsotonicLogisticRegression:
    
    def __init__(self, excluded_cols = [], pseudocount=1.0, penalty=None, n_splits=5, random_state=1, cccv_n_jobs=-1, **kwargs):
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
        self.logr = LogisticRegression(penalty=penalty, **kwargs)
        self.cccv = CalibratedClassifierCV(estimator=self.logr, method='isotonic', 
                cv=KFold(n_splits=n_splits, shuffle=True, random_state=random_state), n_jobs=cccv_n_jobs, ensemble=True)
        self.pseudocount = pseudocount
         
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

    def encode(self, X1, y1, prior_func='constant', prior_strength=1):
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
        # The triangular kernel has unbound mean value, so it is not used
        def triangular_kernel(val, mid, lo, hi): return max((0, ((val-lo) / (mid-lo) if (val < mid) else (hi-val) / (hi-mid))))
        def heaviside_rectangular_kernel(val, mid, lo, hi): return (1 if (lo < val and val < hi) else (0.5 if (val == lo or val == hi) else 0))
        mykernel = triangular_kernel    

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
            ''' # This code, which uses prefix sum, is supposed to be faster than the code using binary search in terms of big-O (n versus nlogn) if there is no duplicated values
                # but there is such a big constant in it that it is probably not worth it (especially if duplicated values have to be handled, resulting in nlogn running time anyway). 
                # also the sorted built-in function is nlogn, so the overall runntime time cannot be changed
            xysorted = sorted(zip(X[:,colidx], y))
            x0_previdx_arr = np.zeros(len(y) + 2)
            x1_previdx_arr = np.zeros(len(y) + 2)
            x0_nextidx_arr = np.zeros(len(y) + 2)
            x1_nextidx_arr = np.zeros(len(y) + 2)
            x0_previdx_arr[0] = x0_previdx_arr[-1] = x0_nextidx_arr[0] = x0_nextidx_arr[-1] = -1
            x1_previdx_arr[0] = x1_previdx_arr[-1] = x1_nextidx_arr[0] = x1_nextidx_arr[-1] = -1
            x0count_prefixsum = np.zeros(len(y) + 2)
            y0count_prefixsum = np.zeros(len(y) + 2)            
            x0_previdx = x1_previdx = 0
            for i, (xi, yi) in enumerate([-1e100, 0.5] + xysorted + [1e100, 0.5]):
                if i == 0: continue
                if i == len(xysorted) + 1: continue
                x0_previdx_arr[i] = x0_previdx
                x1_previdx_arr[i] = x1_previdx
                x0count_prefixsum[i] = x0count_prefixsum[i-1] #if i > 0 else 0
                x0count_prefixsum[i] = x0count_prefixsum[i-1] #if i > 0 else 0
                if yi == 0:
                    for j in range(x0_previdx-1, i, 1):
                        if j>=0: x0_nextidx[j] = i
                    x0_previdx = i
                    x0count_prefixsum[i] += 1
                if yi == 1: 
                    for j in range(x1_previdx-1, i, 1):
                        if j >=0: x1_nextidx[j] = i
                    x1_previdx = i
                    x1count_prefixsum[i] += 1
            x0count_prefixsum[-1] = x0count_prefixsum[-2]
            x1count_prefixsum[-1] = x1count_prefixsum[-2]
            
            for i, (xi, yi) in enumerate(xysorted):
                i += 1
                if yi == 0 and x1_nextidx_arr[i] - x1_previdx_arr[i] <= x0_nextidx_arr[i] - x0_previdx_arr[i]: # compute prefix sum
                if yi == 1 and x1_nextidx_arr[i] - x1_previdx_arr[i] >= x0_nextidx_arr[i] - x0_previdx_arr[i]: # compute prefix sum
                # compared prefix sum
            '''
            
            # Each estimated odds is approx inv-gamma distributed with alpha=2 and beta=1, https://en.wikipedia.org/wiki/Inverse-gamma_distribution
            xyarr = sorted((xi, yi) for (xi, yi) in zip(X[:,colidx], y))
            x0_arr = np.array([-1e100] + [xi for (xi, yi) in xyarr if yi == 0] + [1e100])
            x1_arr = np.array([-1e100] + [xi for (xi, yi) in xyarr if yi == 1] + [1e100])
            windows = []
            
            for xi, yi in xyarr:
                prev_winlen = len(windows)
                # assert yi == 0 or yi == 1
                x0lo_idx1 = bisect.bisect_left (x0_arr, xi) - 1
                x0lo = x0_arr[x0lo_idx1]
                x0hi_idx1 = bisect.bisect_right(x0_arr, xi)
                x0hi = x0_arr[x0hi_idx1]
                x1lo_idx1 = bisect.bisect_left (x1_arr, xi) - 1
                #print(F'bisect_left({x1_arr}, {xi}) + 1 == {x1lo_idx1}')
                x1lo = x1_arr[x1lo_idx1]
                x1hi_idx1 = bisect.bisect_right(x1_arr, xi)
                x1hi = x1_arr[x1hi_idx1]
                assert (x0lo < xi)
                assert x0_arr[x0lo_idx1+1] >= xi
                assert (x0hi > xi)
                assert x0_arr[x0hi_idx1-1] <= xi
                assert (x1lo < xi)
                assert x1_arr[x1lo_idx1+1] >= xi
                assert (x1hi > xi)
                x1_arr[x1hi_idx1-1] <= xi
                if 0 == yi and x0lo <= x1lo and x1hi <= x0hi:
                    x1lo_idx2 = bisect.bisect_left (x1_arr, x0lo)
                    x1hi_idx2 = bisect.bisect_right(x1_arr, x0hi) - 1
                    if (x1hi_idx2 - x1lo_idx2) >= (x0hi_idx1 - x0lo_idx1):
                        x0vals = x0_arr[x0lo_idx1:(x0hi_idx1+1)]
                        x1vals = x1_arr[x1lo_idx2:(x1hi_idx2+1)]
                        x0kernels = [mykernel(val, xi, x0lo, x0hi) for val in x0vals]
                        x1kernels = [mykernel(val, xi, x0lo, x0hi) for val in x1vals]
                        x0pcw = sum([heaviside_rectangular_kernel(val, xi, x0lo, x0hi) for val in x0vals])
                        x1pcw = sum([heaviside_rectangular_kernel(val, xi, x0lo, x0hi) for val in x1vals])
                        if min([len(x0vals), len(x1vals)]) != 3: 
                              logging.debug(F'y=0 kernel size==3: xi={xi} yi={yi} x0lo={x1lo} x0hi={x0hi} x0vals={abbrevshow(x0vals)} x1vals={abbrevshow(x1vals)}')
                        else: logging.debug(F'y=0 kernel size!=3: xi={xi} yi={yi} x0lo={x1lo} x0hi={x0hi} x0vals={abbrevshow(x0vals)} x1vals={abbrevshow(x1vals)}')
                        windows.append((xi, yi, x0lo, x0hi, x0vals, x1vals, x0kernels, x1kernels, x0pcw, x1pcw))
                if 1 == yi and x0lo >= x1lo and x1hi >= x0hi:
                    x0lo_idx2 = bisect.bisect_left (x0_arr, x1lo)
                    x0hi_idx2 = bisect.bisect_right(x0_arr, x1hi) - 1
                    if (x0hi_idx2 - x0lo_idx2) >= (x1hi_idx1 - x1lo_idx1):
                        x0vals = x0_arr[x0lo_idx2:(x0hi_idx2+1)]
                        x1vals = x1_arr[x1lo_idx1:(x1hi_idx1+1)]                    
                        x0kernels = [mykernel(val, xi, x1lo, x1hi) for val in x0vals]
                        x1kernels = [mykernel(val, xi, x1lo, x1hi) for val in x1vals]
                        x0pcw = sum([heaviside_rectangular_kernel(val, xi, x1lo, x1hi) for val in x0vals])
                        x1pcw = sum([heaviside_rectangular_kernel(val, xi, x1lo, x1hi) for val in x1vals])
                        if min([len(x0vals), len(x1vals)]) != 3:
                            logging.debug(F'y=1 kernel size==3: xi={xi} yi={yi} x1lo={x1lo} x1hi={x1hi} x0vals={abbrevshow(x0vals)} x1vals={abbrevshow(x1vals)}')
                        else:
                            logging.debug(F'y=1 kernel size!=3: xi={xi} yi={yi} x1lo={x1lo} x1hi={x1hi} x0vals={abbrevshow(x0vals)} x1vals={abbrevshow(x1vals)}')
                        if sum(x0kernels) <= 0 or sum(x1kernels) <= 0:
                            logging.warning(F'For the {colidx}-th feature, '
                                    F'the feature value xi={xi} (with yi={yi} x1lo={x1lo} x1hi={x1hi} x0vals={abbrevshow(x0vals)} x1vals={abbrevshow(x1vals)}) '
                                    F'has estimated densities of {sum(x1kernels)} and {sum(x0kernels)} for positive and negative examples, respectively '
                                    F'(WARNING_ZERO_VALUED_DENSITY). ')
                        windows.append((xi, yi, x1lo, x1hi, x0vals, x1vals, x0kernels, x1kernels, x0pcw, x1pcw))
                if len(windows) == prev_winlen:
                    if 1 == yi: logging.debug(F'y=1 kernel skipped: x1val={xi} range1=({x1lo}, {x1hi}) range0=({x0lo}, {x0hi})')
            if not windows:
                logging.warning(F'Failed to perform kernel density estimation '
                        F'because there is no majority label to the left-or-right part of the window at every feature value. ')
                fitted_x_vals = np.array([np.mean([x for (x,y) in xyarr])])
                raw_1to0_oddsratios = np.array([self.prevalence_odds]) # np.array([sum(y for (x,y) in xyarr) / len(xyarr)])
                sample_weight = np.array([1])
            else:
                #if self.prevalence_odds < 1:
                #    positive_pseudocount = self.pseudocount * self.prevalence_odds
                #    negative_pseudocount = self.pseudocount
                #else:
                #    positive_pseudocount = self.pseudocount
                #    negative_pseudocount = self.pseudocount / self.prevalence_odds
                #positive_pseudocount = self.pseudocount / (1 + self.prevalence_odds) * self.prevalence_odds
                #negative_pseudocount = self.pseudocount / (1 + self.prevalence_odds) * 1

                fitted_x_vals = np.array([w[0] for w in windows])
                raw_1to0_oddsratios = np.array([
                        ((sum(w[7]) + self.pseudocount*w[9]/(w[8]+w[9])) / (sum(w[6]) + self.pseudocount*w[8]/(w[8]+w[9])))
                        for w in windows])
                #sample_weight = np.array([min((len(w[7]), len(w[6]))) for w in windows])
                #logging.warning(F'sample_weight={collections.Counter(sample_weight)}' )
                sample_weight = np.array([1] * len(windows))
            '''
            This part cannot handle the situation with 
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
            idxto0sum2 = [idxto0sum[i] for i in idxs]
            num0s = sum(idxto0sum2)
            idxto1sum2 = [idxto1sum[i] for i in idxs]
            num1s = sum(idxto1sum2)
            fitted_x_vals = [landmark_x_vals[i] for i in idxs]
            raw_1to0_oddsratios = [
                    ( ( (sumup(idxto1s[i])                   ) / (num1s) ) / 
                      ( (sumup(idxto0s[i]) + self.pseudocount) / (num0s + self.pseudocount * num1s / sumup(idxto1s[i])) ) )
                    for i in idxs]

            '''
            
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
            sample_weight = idxto1sum2
            '''
            # central_log_odds = avg_log_odds = np.mean(self.raw_log_odds[colidx])
             #assert not math.isnan(avg_log_odds)
            raw_log_odds = np.log(raw_1to0_oddsratios)
            central_log_odds = np.log(self.prevalence_odds)
            relative_log_odds = raw_log_odds - central_log_odds
            self.raw_log_odds[colidx] = raw_log_odds
            self.ixs1[colidx] = fitted_x_vals
            self.ivs1[colidx] = 1*central_log_odds + self.irs1[colidx].fit_transform(fitted_x_vals, relative_log_odds, sample_weight = sample_weight)
            self.irs0[colidx] = self.irs1[colidx]
            if is_centered:
                x2, y2 = self._center(fitted_x_vals, self.irs1[colidx].predict(fitted_x_vals)) # fitted_x_vals instead of x1
                self.ixs2[colidx] = x2
                self.ivs2[colidx] = 1*central_log_odds + self.irs2[colidx].fit_transform(x2, y2)
                self.irs0[colidx] = self.irs2[colidx]
        self.logORX = self._transform(X) # np.array([self.irs[colidx].predict((X[:,colidx])) for colidx in range(X.shape[1])]).transpose()
        logging.debug(self.logORX)
        #self.cccv = CalibratedClassifierCV(estimator=self.logr, method='isotonic', 
        #        cv=KFold(n_splits=min([sum(y), 5]), shuffle=True, random_state=1), n_jobs=-1, ensemble=True)
        #self.cccv.fit(self.logORX, y)
        self.logr.fit(np.hstack([self.logORX, exX]), y, **kwargs)
        return self
    
    def get_odds_offset(self):
        return self.prevalence_odds
    def get_density_estimated_log_odds(self):
        return  self.raw_log_odds
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

