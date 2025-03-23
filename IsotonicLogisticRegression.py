#!/usr/bin/env python

import collections,copy,logging,math,pprint,random,warnings
import numpy as np
import scipy
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression, LogisticRegression
#from sklearn.neighbors import KernelDensity, KNeighborsRegressor
#from sklearn.preprocessing import QuantileTransformer

from sklearn.utils.validation import check_is_fitted

def moving_average(x, w):
    return np.convolve(x, np.ones(w), "valid") / w

class AlwaysConstantRegressor(BaseEstimator, ClassifierMixin, RegressorMixin):
    def __init__(self, predicted_value=0):
        self.predicted_value = predicted_value
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return np.full(X.shape[0], self.predicted_value)
    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)
    def predict(self, X):
        return np.full(X.shape[0], self.predicted_value)
 
class ConvexRegression(BaseEstimator, ClassifierMixin, RegressorMixin):
    def __init__(self, shape='auto'):
        super().__init__()
        self.pivotlo = None
        self.pivothi = None
        self.pivotlo2 = None
        self.pivothi2 = None
        self.shape = shape
    def compute_pivots(self, x, y, random_state=0):
        assert len(x) == len(y)
        mov_avg_width = int(math.ceil(len(x)**0.5)) # int(math.ceil(1.06 * sigma * len(x)**(-1.0/5.0))) #int(math.ceil(len(x)**0.5/4.0))
        regression_width = mov_avg_width  #int(math.ceil(len(x)**0.5/4.0))
        prediction_width = (regression_width + 1) // 2 # int(math.ceil(len(x)**0.5/8.0))
        #qt = QuantileTransformer(random_state=random_state)
        #x1 = qt.fit_transform([[v] for v in x])
        y1 = moving_average(y, mov_avg_width)
        assert len(y1) == len(y) - (mov_avg_width - 1), F'{len(y1)} == {len(y)} - ({mov_avg_width}-1) failed!'
        #kd = KernelDensity(bandwidth=bandwidth)
        #kd.fit(x1, y)
        #y1 = kd.predict(x1)
        #x2 = [v[0] for v in x1]
        idxmax = np.argmax(y1)
        idxmin = np.argmin(y1)
        if (y1[0] + y1[-1]) / 2.0 > np.mean(y1):
            idx = idxmin
        else:
            idx = idxmax
        idxlo1 = max((int(idx + (mov_avg_width // 2) - regression_width), 0))
        idxhi1 = min((int(idx + (mov_avg_width // 2) + regression_width), len(x)-1))
        idxlo2 = max((int(idx + (mov_avg_width // 2) - prediction_width), 0))
        idxhi2 = min((int(idx + (mov_avg_width // 2) + prediction_width), len(x)-1))

        #pivots = qt.inverse_transform([[x2[idxlo]], [x2[idxhi]]])
        #return pivots[0][0], pivots[1][0]
        return x[idxlo1], x[idxhi1], x[idxlo2], x[idxhi2]
    def fit(self, x, y):
        incs = ('auto','auto')
        if self.shape == 'convex': incs = True, False
        if self.shape == 'concave': incs = False, True
        self.irlo = IsotonicRegression(increasing = incs[0], out_of_bounds = 'clip')
        self.irhi = IsotonicRegression(increasing = incs[1], out_of_bounds = 'clip')

        x, y = zip(*sorted(zip(x,y)))
        #print(x)
        #print(y)
        self.pivotlo, self.pivothi, self.pivotlo2, self.pivothi2 = self.compute_pivots(x, y)
        logging.debug(F'pivots={self.pivotlo},{self.pivothi}')
        xlo, ylo = zip(*[v for v in zip(x,y) if v[0] <= self.pivothi])
        xhi, yhi = zip(*[v for v in zip(x,y) if v[0] >= self.pivotlo])
        self.irlo.fit(xlo, ylo)
        self.irhi.fit(xhi, yhi)
    def transform(self, x):
        return self.predict(x)
    def predict(self, x):
        ret = []
        ylo = self.irlo.predict(x)
        yhi = self.irhi.predict(x)
        for i in range(len(x)):
            if x[i] < self.pivotlo2:
                v = ylo[i]
            if x[i] > self.pivothi2:
                v = yhi[i]
            if x[i] >= self.pivotlo2 and x[i] <= self.pivothi2:
                v = (ylo[i] + yhi[i]) / 2
            ret.append(v)
        return np.array(ret)
    def fit2d(self, X, y):
        X = np.array(X)
        for colidx in range(X.shape[1]):
            self.fit1d(X[:,colidx], y)
        return self
    def transform2d(X):
        X = np.array(X)
        ret = []
        for colidx in range(X.shape[1]):
            y = self.transform1d(X[:,colidx])
            ret.append(y)
        return np.array(ret).transpose()
    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)
    
# This is some example code for an implementation of the logistic regression with odds ratios estimated by isotonic regressions.
# In the future, we may:
#   1. optimize both the isotonic curve and the logistic curve together so the the overall cross-entropy loss is minimized
#   2. perform additional isotonic regression on the sum of each pair of fitted isotonic functions

class IsotonicLogisticRegression(BaseEstimator, ClassifierMixin, RegressorMixin):
    
    def __init__(self, 
            excluded_cols = [], 
            convex_cols=[],
            task='classification',
            final_predictor = None, # (ElasticNetCV() if taks=='regression' else LogisticRegression()),
            pseudocount=0.5, 
            random_state=0,
            fit_add_measure_error=None, 
            transform_add_measure_error=None,
            ft_fit_add_measure_error=None, 
            ft_transform_add_measure_error=None,
            fit_data_clear=False,
            spearman_r_pvalue_thres=0.05, spearman_r_pvalue_warn=True, spearman_r_pvalue_drop_irrelevant_feature=True, spearman_r_pvalue_correction='none',
            increasing = 'auto',
            **kwargs):
        """ 
        Initialize
        @param excluded_cols: columns to remain untransformed
        @param convex_cols: columns that are subject to convex regression (modeling by convex functions) instead of isotonic regression (monotonic functions)
        @param task: classification (default) or regression
        @param final_predictor: the final predictor to be used afer feature transformations, 
            defaults to LogisticRegression and LinearRegression with default params for classification and regression, respectively. 
        
        @param pseudocount: deprecated and not used (has no effect whatsoever)
        
        @param random_state: the state for generating random numbers (just like the random_state from sklearn)
        
        @param fit_add_measure_error         : introduce noise to the fit                      method to prevent overfitting. Empirical evidence supports its use for plain decision trees. 
        @param transform_add_measure_error   : introduce noise to the transform                method to prevent overfitting. This option is advanced. 
        @param ft_fit_add_measure_error      : introduce noise to the fit part of       fit_transform to prevent overfitting. This option is advanced. 
        @param ft_transform_add_measure_error: introduce noise to the transform part of fit_transform to prevent overfitting. This option is advanced. 
        
        @param fit_data_clear: let the fit method perform clear_intermediate_internal_data at its end
        
        @param spearman_r_pvalue_thres: the p-value for the null hypothesis that the label as a function of a feature is neither increasing nor decreasing
        @param spearman_r_pvalue_warn: when set to True, gives warning with warnings.warn if the null hypothesis fails to hold at the given p-value threshold of spearman_r_pvalue_thres
        @param spearman_r_pvalue_drop_irrelevant_feature: zero out the feature if the null hypothesis fails to hold at the p-value threshold of spearman_r_pvalue_thres for the feature. 
            If the p-value threshold is <0 and >1, then always zero out and and keep unchanged the feature, respectively.
            It is highly recommended to use sklearn.feature_selection.VarianceThreshold to remove the features that are zeroed-out. 
        @param spearman_r_pvalue_correction: can be "bonferroni" (which tends to over-correct p-values) or anything else (which does not correct) for correcting p-values
        
        @param increasing: True/False if the label as a function of each feature is increasing/decreasing, where 'auto' means inferred from the data.
            Typically, when this value is set to either True or False (i.e., not 'auto'), then spearman_r_pvalue_drop_irrelevant_feature should be set to False
            because this True/False provides prior info to the relationship between the label and the feature.
        
        @return the initialized instance
        """
        super().__init__()

        self.excluded_cols = excluded_cols
        self.convex_cols =   convex_cols
        
        self.task = task
        self.final_predictor = final_predictor
        # Probability can be calibrated with:
        # n_splits=5, random_state=1, cccv_n_jobs=-1,
        # sklearn.calibration.CalibratedClassifierCV(estimator=None, *, method='sigmoid', cv=KFold(n_splits=5, shuffle=True, random_state=1), n_jobs=-1, ensemble=True)
        # sklearn.model_selection.KFold(n_splits=5, *, shuffle=False, random_state=None)
        # self.cccv = CalibratedClassifierCV(estimator=self._internal_predictor, method='isotonic', cv=KFold(n_splits=n_splits, shuffle=True, random_state=random_state), n_jobs=cccv_n_jobs, ensemble=True)
        # We tested calibration and confirmed that LogisticRegression is already well-calibrated, which is as expected from theory.
        self.pseudocount = pseudocount
        self.random_state = random_state
        self.fit_add_measure_error = fit_add_measure_error
        self.transform_add_measure_error = transform_add_measure_error
        self.ft_fit_add_measure_error = ft_fit_add_measure_error
        self.ft_transform_add_measure_error = ft_transform_add_measure_error
        
        self.fit_data_clear = fit_data_clear
        self.spearman_r_pvalue_thres = spearman_r_pvalue_thres
        self.spearman_r_pvalue_warn = spearman_r_pvalue_warn
        self.spearman_r_pvalue_drop_irrelevant_feature = spearman_r_pvalue_drop_irrelevant_feature
        self.spearman_r_pvalue_correction = spearman_r_pvalue_correction
        self.increasing = increasing
        self.kwargs = kwargs

    def __sklearn_is_fitted__(self):
        """
        Check fitted status and return a Boolean value.
        """
        return hasattr(self, "_is_fitted") and self._is_fitted

    def check_increasing(self, x, y, spearman_r_pvalue_thres, effective_n_samples):
        """Determine whether y is monotonically correlated with x.

        y is found increasing or decreasing with respect to x based on a Spearman
        correlation test.

        Parameters
        ----------
        x : array-like of shape (n_samples,)
                Training data.

        y : array-like of shape (n_samples,)
            Training target.
        
        spearman_r_pvalue_thres : float
            The P value threshold for the Spearman correlation test
        
        effective_n_samples : number of independent datapoints sampled, this is generally smaller than n_samples
        
        Returns
        -------
        A tuple (a, b, c) of three numbers
        
        a : float 
            Spearman correlation coefficient
        
        b : float between 0 and 1
            P value of the coefficient
        
        c : -1, 0, or 1
            Whether the relationship is increasing (1), decreasing (-1) or undetermined (0).

        Notes
        -----
        The Spearman correlation coefficient is estimated from the data, and the
        sign of the resulting estimate is used as the result.

        References
        ----------
        Fisher transformation. Wikipedia.
        https://en.wikipedia.org/wiki/Fisher_transformation
        """
        
        # Calculate Spearman rho estimate and set return accordingly.
        rho, _ = spearmanr(x, y)
        rho_sgn = (0 if rho == 0 else (1 if rho > 0 else -1))
        
        if effective_n_samples > 3:
            # Run Fisher transform to get the rho CI, but handle rho=+/-1
            if rho not in [-1.0, 1.0]:
                F = 0.5 * math.log((1.0 + rho) / (1.0 - rho))
                F_se = 1 / math.sqrt(effective_n_samples - 3)

                # Use a 95% CI, i.e., +/-1.96 S.E.
                # https://en.wikipedia.org/wiki/Fisher_transformation
                
                # assert 0.95-1e-9 < 1-scipy.stats.norm.sf(scipy.stats.norm.ppf(0.95)) < 0.95+1e-9
                # assert 1.96-1e-9 < scipy.stats.norm.ppf(1-scipy.stats.norm.sf(1.96)) < 1.96+1e-9
                std_observed = abs(F) / F_se
                pvalue_observed = scipy.stats.norm.sf(std_observed) * 2
            else:
                pvalue_observed = 0.0
        else:
            pvalue_observed = 1.0
        return rho, pvalue_observed, (rho_sgn if pvalue_observed < spearman_r_pvalue_thres else 0)

    def _abbrevshow(alist, anum=5):
        if len(alist) <= anum*2: return [alist]
        else: return [alist[0:anum], alist[(len(alist)-anum):len(alist)]]
    def set_random_state(self, random_state):
        self.random_state = random_state
    def custom_get_params(self):
        """ Recursively get the params of this model """
        return {'log_OR': self.logORX, 'LogisticRegression.params': self._internal_predictor.get_params(), 'IsotonicRegression.params' : [ir.get_params() for ir in self.irs0_]}
        return ret
    def clear_intermediate_internal_data(self, steps=[0,1,2]):
        if 0 in steps:
            self.X0_ = None
            self.X1_ = None
            self.raw_log_odds_ = None
        if 1 in steps:
            self.ixs1_ = None
            self.irs1_ = None
            self.ivs1_ = None
        if 2 in steps:
            self.ixs2_ = None
            self.irs2_ = None
            self.ivs2_ = None
    def get_info(self):
        """ Recursively get the fitted params of this model """
        int_pred = self._internal_predictor
        if self.task == 'regression':
            int_pred_info = [int_pred.coef_, int_pred.intercept_, int_pred.n_features_in_]
        else:
            int_pred_info = [int_pred.classes_, int_pred.coef_, int_pred.intercept_, int_pred.n_features_in_, int_pred.n_iter_]
        isor_info = []
        for i in range(len(self.irs0_)):
            ir = self.irs0_[i]
            isor_info.append([ir.X_min_, ir.X_max_, ir.X_thresholds_, ir.y_thresholds_, ir.f_, ir.increasing_])
        return [int_pred_info, isor_info]
    
    def _split(self, X, is_already_splitted=False):
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
            return X.iloc[:,in_colidxs], X.iloc[:,ex_colidxs], in_colidxs, ex_colidxs
        else:
            return X[:,in_colidxs], X[:,ex_colidxs], in_colidxs, ex_colidxs
    
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
    
    def _assert_input(self, X, y, is_num_asserted=True, is_binary_clf_asserted=True):
        for rowit in range(X.shape[0]):
            for colit in range(X.shape[1]):
                if is_num_asserted: 
                    assert not math.isnan(X[rowit][colit]), F'Nan value encountered in row {rowit} col {colit} ({X[rowit]})'
                    assert (-1e50 < X[rowit][colit]), F'Number too small (< -1e50)  at row {rowit} col {colit} ({X[rowit]})'
                    assert ( 1e50 > X[rowit][colit]), F'Number too large (>  1e50)  at row {rowit} col {colit} ({X[rowit]})'
        assert X.shape[0] == len(y), F'{X.shape[0]} == {len(y)} failed for the input X={X} and y={y}'
        if is_binary_clf_asserted:
            for label in y: assert label in [0, 1], F'Label {label} is not binary'
            X0 = X[y==0,:]
            X1 = X[y==1,:]
            assert X.shape[1] == X0.shape[1] and X0.shape[1] == X1.shape[1], 'InternalError'
            assert X0.shape[0] > 1, 'At least two negative examples should be provided'
            assert X1.shape[0] > 1, 'At least two positive examples should be provided'
            #assert X1.shape[0] < X0.shape[0], 'The number of positive examples should be less than the number of negative examples'
    
    def ensure_total_order(self, xs):
        xsetlist1 = sorted(set(xs))
        if len(xsetlist1) <= 1: return [x for x in xs]
        ret = []
        local_rand = random.Random(self.random_state)
        zs = list(range(len(xs)))
        shuf_ret = local_rand.shuffle(zs)
        assert shuf_ret == None        
        xsetlist = [xsetlist1[0] - (xsetlist1[1]-xsetlist1[0])] + xsetlist1 + [xsetlist1[-1] + (xsetlist1[-1]-xsetlist1[-2])]
        x2prev = {}
        x2next = {}
        for i,x in enumerate(xsetlist):
            if i == 0 or i == len(xsetlist) - 1: continue
            x2prev[x] = xsetlist[i-1]
            x2next[x] = xsetlist[i+1]
        for x, z in zip(xs, zs):
            lower = (x2prev[x] + x) / 2.0
            upper = (x2next[x] + x) / 2.0
            ret.append(lower + (upper - lower) * (z+0.5) / len(xs))
        return ret
    '''
    def total_order(self, xs, ys):
        ret = []
        local_rand = random.Random(self.random_state)
        zs = list(range(len(xs)))
        shuf_ret = local_rand.shuffle(zs)
        assert shuf_ret == None
        xzylist = sorted(zip(xs, zs, ys))
        xsetlist = sorted(set(xs))
        xsetlist_idx = 0
        for x, z, y in xzylist:
            while xsetlist[xsetlist_idx] < x: xsetlist_idx += 1
            assert xsetlist[xsetlist_idx] == x
            
            xmid = xsetlist[xsetlist_idx]
            xlower = (xsetlist[xsetlist_idx-1] if (xsetlist_idx-1 >= 0)            
                    else xsetlist[ 0] - (xsetlist[ 1] - xsetlist[ 0]))
            xlo2 = (xlower + xmid)/2.0
            xupper = (xsetlist[xsetlist_idx+1] if (xsetlist_idx+1 < len(xsetlist)) 
                    else xsetlist[-1] + (xsetlist[-1] - xsetlist[-2]))
            xup2 = (xupper + xmid)/2.0
            
            assert z + 0.5 < len(xzylist)
            xnew = xlo2 + (xup2 - xlo2) * (z + 0.5) / len(xzylist)
            ret.append((xnew, y))
        return ret
    '''
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
    
    def get_default(self, *args):
        for arg in args:
            if arg != None: return arg
        return None
   
    def get_feature_names(self):
        check_is_fitted(self)
        return self.feature_names_in_

    def get_feature_importances(self, analysis='f2l2f'):
        """ get feature importance scores
            analysis: one of the following 1-D array in which each number corresponds to a feature
                "spearman_r": returning the observed spearman correlation coefficient of each single feature by itself with the label when ignoring all other features;
                "pvalue": returning the P-value of each single feature by itself when ignoring all other features, with the null hypothesis of having zero spearman correlation coefficient;
                "trend" : returning one of (-1,0,+1) for each single feature by itself, where the label is (decreasing,constant,increasing) as a function of the feature, when ignoring all other features;
                "f2l"   : feature-to-label, returning the average log odds contributed by each single feature by itself when ignoring all other features;
                "f2f"   : feature-to-features, returning the percent contribution of each feature in the ensemble of all features;
                "f2l2f" : feature-to-label multiplied by feature-to-feature, returning the average log odds contributed by each feature in the ensemble of all features.                
        """
        check_is_fitted(self)
        if analysis=='f2l2f':
            return self.feature_importances_
        elif analysis=='f2l':
            return self.feature_importances_to_label_
        elif analysis=='f2f':
            return self.feature_importances_to_features_
        elif analysis=='spearman_r':
            return self.feature_spearman_rs_
        elif analysis=='pvalue':
            return self.feature_pvalues_
        elif analysis=='trend':
            return self.feature_trends_
        else:
            raise TypeError(F'The importance type "{importance_type}" is invalid, it must be either "f2l", "f2f", "f2l2f", "spearman_r", "pvalue", or "trend"!')
    
    def _set_feature_importances(self):
        assert len(self._internal_predictor.coef_.shape) <= 2, F'The shape {self._internal_predictor.coef_.shape} of the coef_ {self._internal_predictor.coef_} of {self._internal_predictor} has more than two dimensions!'
        assert len(self._internal_predictor.coef_.shape) == 1 or self._internal_predictor.coef_.shape[0] == 1, F'The shape {self._internal_predictor.coef_.shape} of the coef_ {self._internal_predictor.coef_} of {self._internal_predictor} is invalid!'
        self.feature_importances_to_label_       = np.array([np.mean([abs(x - np.log(self.get_odds_offset())) for x in self.ivs0_[colidx]]) for colidx, colname in enumerate(self.feature_names_in_)])
        self.feature_importances_to_features_ = np.array([
            self._internal_predictor.coef_[colidx] if 1==len(self._internal_predictor.coef_.shape) else self._internal_predictor.coef_[0][colidx] 
            for colidx, colname in enumerate(self.feature_names_in_)])
        self.feature_importances_ = self.feature_importances_to_label_ * self.feature_importances_to_features_
        return self.feature_importances_

    def fit(self, X1, y1, is_centered=True, add_measure_error=None, data_clear=None, data_clear_steps=[0,1,2], set_feature_importances=True,
            spearman_r_pvalue_thres=None, spearman_r_pvalue_warn=None, spearman_r_pvalue_drop_irrelevant_feature=None, **kwargs):
        """ scikit-learn fit
            is_centered : using centered isotonic regression or not
            set_feature_importances: will set feature importances during the fit
            spearman_r_pvalue_thres:                   if set then will override the one from __init__
            spearman_r_pvalue_warn:                    if set then will override the one from __init__
            spearman_r_pvalue_drop_irrelevant_feature: if set then will override the one from __init__
            kwargs: keyword arguments to the LinearRegression() or LogisticRegression() that is internally used for the regression or classification task and the fit() method used for the task
        """
        if spearman_r_pvalue_thres                   == None: spearman_r_pvalue_thres                   = self.spearman_r_pvalue_thres
        if spearman_r_pvalue_warn                    == None: spearman_r_pvalue_warn                    = self.spearman_r_pvalue_warn
        if spearman_r_pvalue_drop_irrelevant_feature == None: spearman_r_pvalue_drop_irrelevant_feature = self.spearman_r_pvalue_drop_irrelevant_feature
        
        if self.final_predictor:
            self._internal_predictor = self.final_predictor
        else:
            if self.task == 'regression':
                self._internal_predictor = LinearRegression(**self.kwargs)
                #self._internal_predictor = ElasticNetCV(**kwargs)
            else:
                self._internal_predictor = LogisticRegression(**self.kwargs)
        self.irrelevant_feature_indexes_ = []
        
        #def triangular_kernel(val, mid, lo, hi): return max((0, ((val-lo) / (mid-lo) if (val < mid) else (hi-val) / (hi-mid))))
        #def heaviside_rectangular_kernel(val, mid, lo, hi): return (1 if (lo < val and val < hi) else (0.5 if (val == lo or val == hi) else 0))
        def powermean(arr, p=1): return 1.0/len(arr) * (sum(ele**p for ele in arr))**p
        # NOTE: Setting add_measure_error=True may improve the performance of some ML methods. 
        add_measure_error = self.get_default(add_measure_error, self.fit_add_measure_error, False)
        data_clear = self.get_default(data_clear, self.fit_data_clear, False)
        inX, exX, inIdxs, exIdxs = self._split(X1)
        X = np.array(inX)
        y = np.array(y1)
        
        self.feature_spearman_rs_  = [None for _ in range(X.shape[1])]
        self.feature_pvalues_     = [None for _ in range(X.shape[1])]
        self.feature_trends_      = [None for _ in range(X.shape[1])]
        if hasattr(inX, 'columns'):
            self.feature_names_in_ = [(colname if colname != '' else colidx) for colidx, colname in enumerate(inX.columns)]
        else:
            self.feature_names_in_ = [i for i in range(X.shape[1])]
        
        assert X.shape[0] > 0, F'The input {X1} does not have any rows'
        assert X.shape[1] > 0, F'The input {X1} does not have any columns'
        
        if self.spearman_r_pvalue_correction == 'bonferroni': spearman_r_pvalue_thres = spearman_r_pvalue_thres / float(X.shape[1])

        raw_log_odds = self.raw_log_odds_ = [None for _ in range(X.shape[1])]
        inv0 = self.inv0_ = [IsotonicRegression(increasing=self.increasing, out_of_bounds='clip') for _ in range(X.shape[1])]
        ixs0 = self.ixs0_ = [None for _ in range(X.shape[1])]
        ivs0 = self.ivs0_ = [None for _ in range(X.shape[1])]
        irs0 = self.irs0_ = [None for _ in range(X.shape[1])]
        ixs1 = self.ixs1_ = [None for _ in range(X.shape[1])]
        irs1 = self.irs1_ = [IsotonicRegression(increasing=self.increasing, out_of_bounds='clip') for _ in range(X.shape[1])]
        ivs1 = self.ivs1_ = [None for _ in range(X.shape[1])]
        ixs2 = self.ixs2_ = [None for _ in range(X.shape[1])]
        irs2 = self.irs2_ = [IsotonicRegression(increasing=self.increasing, out_of_bounds='clip') for _ in range(X.shape[1])]
        ivs2 = self.ivs2_ = [None for _ in range(X.shape[1])]
        
        self.convex_regressions_0_ = [None for _ in range(X.shape[1])]
        self.convex_regressions_1_ = [ConvexRegression() for _ in range(X.shape[1])]
        self.convex_regressions_2_ = [ConvexRegression() for _ in range(X.shape[1])]
        
        if self.task == 'regression':
            self._assert_input(X, y, is_binary_clf_asserted=False)
            X0 = self.X0_ = X[y<=np.mean(y),:]
            X1 = self.X1_ = X[y> np.mean(y),:]
            self.prevalence_odds_ = np.nan
        else:
            self._assert_input(X, y)
            X0 = self.X0_ = X[y==0,:]
            X1 = self.X1_ = X[y==1,:]
            self.prevalence_odds_ = (len(X1) / float(len(X0)))
        for colidx in range(X.shape[1]): 
            x = X[:,colidx]
            if self.task == 'regression':                
                xylist = sorted(zip(x,y))
                x1 = np.array([x for (x,y) in xylist])
                y1 = np.array([y for (x,y) in xylist])
                center_log_odds = 0
                self.raw_log_odds_[colidx] = y1
                n_effective_examples = len(xylist)
            else:
                x = X[:,colidx]
                xord = self.ensure_total_order(x)
                xylist = sorted(zip(xord,y)) #xylist = (self.total_order(x, y) if (len(set(x)) > 1) else sorted(zip(x,y)))
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
                    odds = (powermean((len(curr_xylist), powermean((pre2_len, nex2_len)))) + 0*self.pseudocount) / (powermean((prev_len, next_len)) + 0*self.pseudocount)
                    xcenters.append(xcenter)
                    xodds.append((odds) if (ylabel == 1) else (1/odds))
                    prev_ylabel = ylabel
                raw_log_odds = np.log(xodds)
                center_log_odds = np.log(self.prevalence_odds_)                
                self.raw_log_odds_[colidx] = raw_log_odds
                x1 = np.array(xcenters)
                y1 = relative_log_odds = raw_log_odds - center_log_odds
                n_effective_examples = (len(xylistlist) - 1) / 2.0

            X_in = inX
            self.ixs1_[colidx] = x1
            spearman_r, pvalue_observed, is_inc_or_dec = self.check_increasing(x1, y1, spearman_r_pvalue_thres, n_effective_examples)
            self.feature_spearman_rs_[colidx] = spearman_r
            self.feature_pvalues_[colidx] = pvalue_observed
            self.feature_trends_[colidx] = is_inc_or_dec

            if colidx in self.convex_cols or (hasattr(X_in, 'columns') and X_in.columns[colidx] in self.convex_cols):
                if not colidx in self.convex_cols: self.convex_cols.append(colidx)
                self.ivs1_[colidx] = self.convex_regressions_1_[colidx].fit_transform(x1, y1)
                self.convex_regressions_0_[colidx] = self.convex_regressions_1_[colidx]
            else:
                if is_inc_or_dec == 0:
                    self.irrelevant_feature_indexes_.append(colidx)
                    if spearman_r_pvalue_warn:
                        colname = (X_in.columns[colidx] if hasattr(X_in, 'columns') else 'Unnamed column')
                        if spearman_r_pvalue_drop_irrelevant_feature:
                            warnings.warn(F'The feature {colname} at column index {colidx} seems to be irrelevant and is dropped (not kept). ')
                        else:
                            warnings.warn(F'The feature {colname} at column index {colidx} seems to be irrelevant but is still kept (not dropped). ')
                    if spearman_r_pvalue_drop_irrelevant_feature:
                        self.irs1_[colidx] = AlwaysConstantRegressor(0)
                self.ivs1_[colidx] = 1*center_log_odds + self.irs1_[colidx].fit_transform(x1, y1)                
                self.irs0_[colidx] = self.irs1_[colidx]
                self.ivs0_[colidx] = self.ivs1_[colidx]
            self.ixs0_[colidx] = self.ixs1_[colidx]
            if is_centered:
                if colidx in self.convex_cols or (hasattr(X_in, 'columns') and X_in.columns[colidx] in self.convex_cols):
                    x2, y2 = self._center(x1, self.convex_regressions_1_[colidx].predict(x1))
                    self.ixs2_[colidx] = x2
                    self.ivs2_[colidx] = 1*center_log_odds + self.convex_regressions_2_[colidx].fit_transform(x2, y2)
                    self.convex_regressions_0_[colidx] = self.convex_regressions_2_[colidx]
                else:
                    x2, y2 = self._center(x1, self.irs1_[colidx].predict(x1))
                    self.ixs2_[colidx] = x2
                    self.ivs2_[colidx] = 1*center_log_odds + self.irs2_[colidx].fit_transform(x2, y2)
                    self.irs0_[colidx] = self.irs2_[colidx]                
                self.ivs0_[colidx] = self.ivs2_[colidx]
                self.ixs0_[colidx] = self.ixs2_[colidx]
                self.inv0_[colidx].fit_transform(y2, x2)
                _x2 = self.inv0_[colidx].predict(y2)
            else:
                self.inv0_[colidx].fit_transform(y1, x1)
        log_ratios = self._transform(X, add_measure_error=add_measure_error, is_inverse=False)
        self._internal_predictor.fit(np.hstack([log_ratios, exX]), y, **kwargs)
        if data_clear: self.clear_intermediate_internal_data(data_clear_steps)
        self.n_features_in_ = X1.shape[1]
        self._is_fitted = True
        self._set_feature_importances()
        return self
    
    def get_odds_offset(self):
        return self.prevalence_odds_
    def get_density_estimated_X(self):
        return self.ixs1_
    def get_density_estimated_log_odds(self):
        return self.raw_log_odds_
    def get_isotonic_X(self):
        return self.ixs1_
    def get_isotonic_log_odds(self):
        return self.ivs1_
    def get_centered_isotonic_X(self):
        return self.ixs2_
    def get_centered_isotonic_log_odds(self):
        return self.ivs2_
        
    def _transform(self, X, add_measure_error, is_inverse):
        if add_measure_error:
            XT = np.array([self.ensure_total_order(X[:,colidx]) for colidx in range(X.shape[1])])
        else:
            XT = np.array([(X[:,colidx]) for colidx in range(X.shape[1])])
        return np.array([(
            self.inv0_[colidx].transform(xT) if is_inverse else (
            self.convex_regressions_0_[colidx].transform(xT)
            if (colidx in self.convex_cols or (hasattr(X, 'columns') and X.columns[colidx] in self.convex_cols))
            else self.irs0_[colidx].predict(xT)
            )) for colidx,xT in enumerate(XT)]).transpose()
        #return np.array([self.irs0_[colidx].predict(xT) for colidx,xT in enumerate(XT)]).transpose()
        #return np.array([self.irs0_[colidx].predict((X[:,colidx])) for colidx in range(X.shape[1])]).transpose()
    
    def transform(self, X1, add_measure_error=None, is_inverse=False):
        """ scikit-learn transform 
            add_measure_error: set to True to add measurement error to prevent overfitting
            is_inverse: set to True to perform inverse transform. Please use the inverse_transform method instead if possible.
        """
        check_is_fitted(self)
        add_measure_error = self.get_default(add_measure_error, self.transform_add_measure_error, False)
        X = np.array(X1)
        inX, exX, inIdxs, exIdxs = self._split(X, True)
        test_orX = self._transform(inX, add_measure_error=add_measure_error, is_inverse=is_inverse)
        X2 = np.zeros((X.shape[0], X.shape[1]))
        X2[:,inIdxs] = test_orX
        X2[:,exIdxs] = exX
        return X2
    
    def inverse_transform(self, X1):
        """
        scikit-learn inverse_transform
        caveat: inverse_transform(transform(X)) != X and transform(inverse_transform(X)) != X for a column x of X 
                if at least one scalar value of x is not within the range in which the transform function of x is monotonically increasing
        """
        return self.transform(X1, is_inverse=True)
    
    def fit_transform(self, X1, y1, fit_add_measure_error=None, transform_add_measure_error=None):
        """ scikit-learn fit_transform 
            fit_add_measure_error: set to True to add measurement error to prevent overfitting (it may work for plain decision trees)
            transform_add_measure_error: set to True to add measurement error to prevent overfitting
        """
        # NOTE: Setting add_measure_error=True may improve the performance of some ML methods 
        #   such as DecisionTreeClassifier (DT) presumably because DT without regularization tends to overfit.
        fit_add_measure_error = self.get_default(fit_add_measure_error, self.ft_fit_add_measure_error, False)
        transform_add_measure_error = self.get_default(transform_add_measure_error, self.ft_transform_add_measure_error, True)
        self.fit(X1, y1, add_measure_error=fit_add_measure_error)
        return self.transform(X1, add_measure_error=transform_add_measure_error)
    
    def _extract_features(self, X):
        inX, exX, inIdxs, exIdxs = self._split(X, True)
        test_orX = self._transform(inX, add_measure_error=False, is_inverse=False)
        return np.hstack([test_orX, exX])
    
    def predict(self, X1):
        """ scikit-learn predict using logistic regression built on top of isotonic scaler """
        check_is_fitted(self)
        X = np.array(X1)
        allfeatures = self._extract_features(X)
        return self._internal_predictor.predict(allfeatures)

    def predict_proba(self, X1):
        """ scikit-learn predict_proba using logistic regression built on top of isotonic scaler """
        check_is_fitted(self)
        X = np.array(X1)
        allfeatures = self._extract_features(X)
        if self.task == 'regression':
            ret = self._internal_predictor.predict(allfeatures)
            return np.array([(x,x) for x in ret])
        return self._internal_predictor.predict_proba(allfeatures)

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
    ilr = IsotonicLogisticRegression(spearman_r_pvalue_thres=2.0, excluded_cols=['col3'])
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

def test_fit_and_predict_with_dups(task='classification'):
    import pandas as pd
    
    pp = pprint.PrettyPrinter(indent=4)
    logging.basicConfig(format='test_fit_and_predict_with_dups %(asctime)s - %(message)s', level=logging.DEBUG)
    X = np.array([
        [ 0,  10, 0],
        [ 0,  30, 0],
        [ 0,  60, 0],
        [ 0, 100, 0],
        [ 0, 150, 0],

        [ 1, 210, 0],
        [ 1, 280, 0],
        [ 1, 360, 0],
        [ 1, 450, 0],
        [ 1, 550, 0],
            
        [ 2, 660, 0],
        [ 2, 780, 0],
        [ 2, 910, 0],
        [ 2,1050, 0],
        [ 2,1300, 0],

        [ 2,1300, 0],
    ])
    X = pd.DataFrame(X, columns = ['col1', 'col2', 'col3'])
    y = np.array([1,1,0,0,1, 0,0,1,0,1, 0,0,1,0,0, 0])
    ilr = IsotonicLogisticRegression(spearman_r_pvalue_thres=2.0, excluded_cols=['col3'], task=task)
    ilr.set_random_state(42+0)
    X2 = ilr.fit_transform(X, y)
    X3 = ilr.transform(X)
    y1 = ilr.predict(X)
    ilr.set_random_state(42+1)
    x2 = ilr.ensure_total_order(X.iloc[:,0])
    ordered_xs = list(zip(X.iloc[:,0],x2))
    print(F'train_ordered_X={ordered_xs}')
    print(F'train_transformed_X=\n{X2}')
    print(F'test_transformed_X=\n{X3}')

def test_fit_and_predict_with_convex(task='classification'):
    import pandas as pd

    pp = pprint.PrettyPrinter(indent=4)
    logging.basicConfig(format='test_fit_and_predict_with_convex %(asctime)s - %(message)s', level=logging.DEBUG)
    X = np.array([
        [ 0,  10, 0],
        [ 0,  30, 0],
        [ 0,  60, 0],
        [ 0, 100, 0],
        [ 0, 150, 0],

        [ 1, 210, 0],
        [ 1, 280, 0],
        [ 1, 360, 0],
        [ 1, 450, 0],
        [ 1, 550, 0],

        [ 2, 660, 0],
        [ 2, 780, 0],
        [ 2, 910, 0],
        [ 2,1050, 0],
        [ 2,1300, 0],

        [ 2,1300, 0],
    ])
    X = pd.DataFrame(X, columns = ['col1', 'col2', 'col3'])
    y = np.array([9,8,8,6,6,4,3,2,1,0,1,2,3,4,5,6])
    ilr = IsotonicLogisticRegression(spearman_r_pvalue_thres=2.0, excluded_cols=['col3'],convex_cols=['col2'], task='regression')
    ilr.set_random_state(42+0)
    X2 = ilr.fit_transform(X, y)
    X3 = ilr.transform(X)
    y1 = ilr.predict(X)
    ilr.set_random_state(42+1)
    x2 = ilr.ensure_total_order(X.iloc[:,0])
    ordered_xs = list(zip(X.iloc[:,0],x2))
    print(F'train_ordered_X={ordered_xs}')
    print(F'train_transformed_X=\n{X2}')
    print(F'test_transformed_X=\n{X3}')
    print(F'test_predicted_X=\n{y1}')

def test_inverse_transform(
        task='classification',
        #task='regression'
        ):
    Xtrain = np.array([
        [-1, -10, 0],
        [ 0,  10, 0],
        [ 1,  30, 0],
        [ 2,  60, 0],
        [ 3, 100, 0],
        [ 4, 150, 0],
        [ 5, 210, 0],
        [ 6, 280, 0],
    ])
    #ytrain = [1e0,1e1,1e2,1e3,1e4,1e5,1e6]
    ytrain = [0, 0, 1, 0, 1, 0, 1, 1]
    Xtest = np.array([
        [ 0,  10, 0],
        [ 1,  30, 0],
        [ 2,  60, 0],
        [ 3, 100, 0],
        [ 4, 150, 0],
        [ 5, 210, 0],
    ])
    ilr = IsotonicLogisticRegression(spearman_r_pvalue_thres=2.0, task=task)
    ilr.set_random_state(42+0)
    ilr.fit(Xtrain, ytrain)
    X2 = ilr.transform(Xtest)
    X3 = ilr.inverse_transform(X2)
    np.testing.assert_allclose(X3, Xtest, rtol=1e-6, atol=1e-6)

if __name__ == '__main__':
    test_inverse_transform()
    test_fit_and_predict_proba()
    test_fit_and_predict_with_dups()
    test_fit_and_predict_with_dups(task='regression')
    test_fit_and_predict_with_convex(task='regression')
    
