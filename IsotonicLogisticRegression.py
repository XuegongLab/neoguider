#!/usr/bin/env python

import collections,copy,logging,math,pprint,random,sys,warnings
import numpy as np
import scipy

from scipy.interpolate import interp1d
from scipy.stats import spearmanr

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression, LogisticRegression
#from sklearn.neighbors import KernelDensity, KNeighborsRegressor
#from sklearn.preprocessing import QuantileTransformer

import statsmodels
from statsmodels.stats.multitest import fdrcorrection_twostage

from sklearn.utils import resample
from sklearn.utils.validation import check_is_fitted
from lib import _mannwhitneyu

LOGLEVEL_DEBUG1 = logging.DEBUG + 1

LikelihoodRatioResult = collections.namedtuple('LikelihoodRatioResult', ['statistic', 'pvalue', 'dof'])
MannwhitneyuEffectSizeResult = collections.namedtuple('MannwhitneyuEffectSizeResult', ['statistic', 'pvalue', 'mu', 'sd'])
SpearmanrResult2 = collections.namedtuple('SpearmanrResult2', ['statistic', 'pvalue'])

_EFFECT_SIZES = [(0.01*(i)) for i in range(0, 20+1)]

def _abbrevshow(alist, anum=5):
    if len(alist) <= anum*2: return [alist]
    else: return [alist[0:anum], alist[(len(alist)-anum):len(alist)]]

def _is_any_in(alist, bset):
    ret = 0
    for a in alist:
        if a in bset: ret += 1
    return ret

def _transform_and_partition(x, y):
    if len(x) == 0:
        return []
    
    # Step 1: Transform into a list of (feature_value, (x0count, x1count))
    feature_counts = collections.defaultdict(lambda: [0, 0])
    for feature, label in zip(x, y):
        if label == 0:
            feature_counts[feature][0] += 1
        else:
            feature_counts[feature][1] += 1

    transformed_list = [(feature, (counts[0], counts[1])) for feature, counts in sorted(feature_counts.items())]

    # Step 2: Partition the list into sublists where consecutive tuples with x0count=0 or x1count=0 are merged
    if not transformed_list:
        return []

    partitioned = []
    current_sublist = [transformed_list[0]]

    for i in range(1, len(transformed_list)):
        current_feature, (current_x0, current_x1) = transformed_list[i]
        prev_feature, (prev_x0, prev_x1) = transformed_list[i - 1]

        # Check if current and previous tuples can be merged based on x0count or x1count being zero
        if (current_x0 == 0 and prev_x0 == 0) or (current_x1 == 0 and prev_x1 == 0):
            current_sublist.append(transformed_list[i])
        else:
            partitioned.append(current_sublist)
            current_sublist = [transformed_list[i]]

    partitioned.append(current_sublist)

    return partitioned

def _center_group(contig):
    ftvalwsum = 0
    x0cntsum = 0
    x1cntsum = 0
    for ftval, (x0cnt, x1cnt) in contig:
        ftvalwsum += ftval * (x0cnt + x1cnt)
        x0cntsum += x0cnt
        x1cntsum += x1cnt
    return ftvalwsum / (x0cntsum + x1cntsum), (x0cntsum, x1cntsum)
    
# This piece of code confirms that rank_biserial_correlation, pointbiserialr, and pearsonr are equivalent to each other if used as follows:
# implication: pearsonr=2*(AUC_ROC-0.5) and _fisher_transform can be applied to any of these correlation coefficients
'''
import numpy as np
import scipy
def rank_biserial_correlation(y, x, *args, **kwargs):
	negs = [a2 for a1, a2 in zip(y, x) if a1 == 0]
	poss = [a2 for a1, a2 in zip(y, x) if a1 == 1]
	ret = scipy.stats.mannwhitneyu(negs, poss, *args, **kwargs)
	return 1.0 - 2.0 * ret.statistic / len(negs) * len(poss)
def test_ass_btw_spearman_and_ranksum(n):
	x = np.array(range(n))
	y = np.random.binomial(n, x / float(n + 1))
	pointbiserialr = scipy.stats.pointbiserialr(y, x)
	pearsonr = scipy.stats.pearsonr(y, x)
	spearmanr = scipy.stats.spearmanr(y, x)
	rankbiserialr = rank_biserial_correlation(y, x)
	print(F'pointbiserialr={pointbiserialr}')
	print(F'rankbiserialr={pointbiserialr}')
	print(F'pearsonr={pearsonr}')
	print(F'spermrnr={spearmanr}')
test_ass_btw_spearman_and_ranksum(100)
'''

# https://stackoverflow.com/questions/38248595/likelihood-ratio-test-in-python
def _likeratio2(x1, x2, axis=0, effect_size=-1):
    x1 = np.array(x1)
    x2 = np.array(x2)    
    if axis == 0:
        x1 = x1.transpose()
        x2 = x2.transpose()
    assert x1.shape[0] == x2.shape[0], F'The shapes {x1.shape} and {x2.shape} are not equal in the number of rows (transformed with axis={axis}), so cannot perform _likeratio!'
    ret = []
    for colidx, (hypothesis1_vals, hypothesis2_vals) in enumerate(zip(x1, x2)):
        valid_values = (set(hypothesis1_vals) | set(hypothesis2_vals))
        if len(valid_values) >= 10:
            #logging.warning(F'The column at {colidx} has more than ten possible values but is considered as a categorical variable. '
            #        'This is likely an error. '
            #        'Returning np.nan for the results of the statistical test. ')
            ret.append((np.nan, np.nan, np.nan))
            continue
        elif len(valid_values) == 1:
            logging.warning(F'The column at {colidx} has only one possible value of {valid_values}. '
                    'You should have filtered this column out. '
                    'Returning np.nan for the results of the statistical test. ')
            ret.append((0.0, 1.0, 0.0))
            continue
        hypothesis1_category2count = collections.Counter(hypothesis1_vals)
        hypothesis2_category2count = collections.Counter(hypothesis2_vals)
        hypothesis1_counts = [hypothesis1_category2count[c] for c in valid_values]
        hypothesis2_counts = [hypothesis2_category2count[c] for c in valid_values]
        loglike_diff = 0
        k = np.sum(hypothesis2_counts) + sys.float_info.epsilon
        n = np.sum(hypothesis1_counts) + np.sum(hypothesis2_counts) + sys.float_info.epsilon * 2
        p = k/float(n)
        for v1, v2 in zip(hypothesis1_counts, hypothesis2_counts):
            sub_k = v1 + sys.float_info.epsilon
            sub_n = v1 + v2 + sys.float_info.epsilon
            sub_p = sub_k / sub_n
            sub_loglike = scipy.stats.binom.logpmf(k=sub_k, n=sub_n, p=sub_p)
            loglike = scipy.stats.binom.logpmf(k=sub_k, n=sub_n, p=p)
            if effect_size < 0:
                loglike_diff += sub_loglike - loglike
            else:
                null_hyp_frac1 = (1 + effect_size + sys.float_info.epsilon) / (2.0 + sys.float_info.epsilon * 2)
                null_hyp_frac2 = 1 - null_hyp_frac1
                null_hyp_min_frac, null_hyp_max_frac = min((null_hyp_frac1, null_hyp_frac2)), max((null_hyp_frac1, null_hyp_frac2))
                if null_hyp_min_frac < sub_p and sub_p < null_hyp_max_frac:
                    null_loglike1 = scipy.stats.binom.logpmf(k=sub_k, n=sub_n, p=null_hyp_min_frac)
                    null_loglike2 = scipy.stats.binom.logpmf(k=sub_k, n=sub_n, p=null_hyp_max_frac)
                    assert sub_loglike - max((null_loglike1, null_loglike2)) >= 0, F'{sub_loglike} - max({null_loglike1}, {null_loglike2}) failed for loglike computation v1={v1} and v2={v2}!'
                    loglike_diff += sub_loglike - max((null_loglike1, null_loglike2))
                # else; do nothing
        statistic = 2*loglike_diff
        dof = len(valid_values) - 1 # degree of freedom is the number of extra params in the mode
        pvalue = scipy.stats.chi2.sf(statistic, dof)
        ret.append((statistic, pvalue, dof))
    statistic, pvalue, dof = zip(*ret)
    return LikelihoodRatioResult(statistic, pvalue, dof)

def _fisher_transform(rho, n, rho_thres=0, n_tails=1): # one-tailed if the null hypothesis assumes that the efect size is some value away from zero
    # rho_sgn = (0 if rho == 0 else (1 if rho > 0 else -1))
    if n > 3:
        if rho not in [-1.0, 1.0]:
            F = 0.5 * math.log((1.0 + rho) / (1.0 - rho))
            F_thres = 0.5 * math.log((1.0 + rho_thres) / (1.0 - rho_thres))
            F_se = 1 / math.sqrt(n - 3)
            # assert 0.95-1e-9 < 1-scipy.stats.norm.sf(scipy.stats.norm.ppf(0.95)) < 0.95+1e-9
            # assert 1.96-1e-9 < scipy.stats.norm.ppf(1-scipy.stats.norm.sf(1.96)) < 1.96+1e-9
            std_observed = (F_thres - abs(F)) / F_se
            pvalue_observed = scipy.stats.norm.sf(std_observed) * n_tails # one-tailed towards zero
        else:
            pvalue_observed = 0.0
    else:
        pvalue_observed = 1.0
    assert 0 <= pvalue_observed and pvalue_observed <= 1, F'The pvalue {pvalue_observed} is not between zero and one!'
    return pvalue_observed

def _approx_H0_assume_some_effect_size_pval(statistics, n, effect_sizes, mus, sds, stat_test=''):
    for stat in statistics: assert np.isnan(stat) or (-1 <= stat and stat <= 1), f'The statistic {stat} is not between -1 and +1!'
    ret = {}
    for effect_size in effect_sizes:
        ret[effect_size] = [] # [np.nan] * len(statistics)
        for i, stat in enumerate(statistics):
            if np.isnan(stat): 
                assert stat_test == 'spearmanr'
                pvalue = 0.5
            elif stat_test == 'spearmanr':
                pvalue = _fisher_transform(stat, n, effect_size)
            elif stat_test == 'mannwhitneyu':
                # mus and sds are not tie-corrected
                if not np.allclose(abs(-mus[i]), abs(stat), atol=1e-7):
                    raise ValueError(F'{mus[i]}=={stat} failed (ignoring sign)!')
                if sds[i] == 0:
                    pvalue = 1 if abs(stat) > effect_size else 0
                else:
                    pvalue = scipy.stats.norm.sf((effect_size - abs(stat)) / sds[i]) + scipy.stats.norm.sf((effect_size + abs(stat)) / sds[i])
                    assert 0 <= pvalue and pvalue <= 1.0, F'The p-value {pvalue} is invalid!'
            else:
                raise ValueError(f'The stat_test {stat_test} is invalid!')
            assert 0 <= pvalue and pvalue <= 1.0, F'The p-value {pvalue} is invalid for test {stat_test}!'
            ret[effect_size].append(pvalue)
    return ret

def _moving_average(x, w):
    return np.convolve(x, np.ones(w), "valid") / w

def _rank_biserial_correlation(u1, x1size, x2size):
    return 1.0 - 2.0 * u1 / (x1size * x2size) # rank-biserial correlation

def mannwhitneyu2(a1, a2, *args, **kwargs):
    # from https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html#Notes
    # ret = scipy.stats.mannwhitneyu(a1, a2, *args, **kwargs)
    ret = _mannwhitneyu.mannwhitneyu(a1, a2, *args, **kwargs)
    axis = kwargs.get('axis', 0)
    U1 = ret.statistic
    mult = 2.0 / (a1.shape[axis] * a2.shape[axis])
    abs1_stat = _rank_biserial_correlation(U1, a1.shape[axis], a2.shape[axis])
    assert ((0 <= ret.pvalue) & (ret.pvalue <= 1)).all(), F'{ret.pvalue} is not between zero and one!'
    # Udiff is not tie-corrected
    return MannwhitneyuEffectSizeResult(statistic=abs1_stat, pvalue=ret.pvalue, mu=mult*ret.Udiff, sd=mult*ret.Udiff_sd)

def spearmanr2(X, y, *args, **kwargs):
    ret = scipy.stats.spearmanr(X, y, *args, **kwargs)
    pvalue = np.nan_to_num(ret.pvalue, nan=0.5)
    assert ((0 <= pvalue) & (pvalue <= 1)).all(), F'{pvalue} is not between zero and one!'
    if isinstance(ret.statistic, float) or isinstance(ret.statistic, int):
        return SpearmanrResult2(
                statistic=np.array([ret.statistic]),
                pvalue=np.array([pvalue]))
    else:
        return SpearmanrResult2(
                statistic=np.array([ret.statistic[i,-1] for i in range(X.shape[1])]), 
                pvalue=np.array([pvalue[i,-1] for i in range(X.shape[1])]))

# This Monte-Carlo simulation is too computationally intensive, please avoid using it if possible. 
# This is still kept because the asymptotic formula may not work for very small sample (less than 8 positives or less than 8 negatives)
# But in this case of having very small sample, the user should probably collect more samples anyway. 
# group1 contains the negative examples
def _bootstrap_H0_assume_some_effect_size_pval(test_meth, group1, group2, effect_sizes, max_n_susbamples, n_iterations):
    assert len(group1) == len(group2) or (group1.shape[-1] == group2.shape[-1]), F'{group1.shape} == {group2.shape} in the first or last dimension failed! '
    is_mannwhit = (len(group1.shape) == len(group2.shape))
    ret = {effect_size: np.zeros(group1.shape[1]) for effect_size in effect_sizes}
    for i in range(n_iterations):
        if is_mannwhit:
            sample1 = resample(group1, replace=False, n_samples=min((max_n_susbamples, len(group1))), random_state=i*2)
            sample2 = resample(group2, replace=False, n_samples=min((max_n_susbamples, len(group2))), random_state=i*2+1)
        elif len(group1.shape) == len(group2.shape) + 1:
            sample_indices = resample(np.arange(len(group1)), replace=True, n_samples=min((max_n_susbamples, len(group1))), random_state=i)
            sample1 = group1[sample_indices]
            sample2 = group2[sample_indices]
        else:
            raise ValueError(F'The arrays of shapes {group1.shape} and {group2.shape} are incompatible with each other! ')        
        test_result = test_meth(sample1, sample2, axis=0)
        if is_mannwhit: logging.log(LOGLEVEL_DEBUG1, f'{test_result}={test_meth}({sample1}, {sample2})')
        for effect_size in effect_sizes:
            H0_indicators = np.where(np.abs(test_result.statistic) > effect_size, 1, 0)
            ret[effect_size] += H0_indicators
            if is_mannwhit: logging.log(LOGLEVEL_DEBUG1, f'{test_result}.effect_size={effect_size}={H0_indicators}')
    ret = {k : (v / float(n_iterations)) for k, v in sorted(ret.items())}
    if is_mannwhit: logging.log(LOGLEVEL_DEBUG1, f'ret={ret}')
    return ret

# Three one-dimension regressors

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

class ScalingRegressor1D(BaseEstimator, ClassifierMixin, RegressorMixin):
    def __init__(self, shift_scaling_factor=0.0, scaling_factor=1.0):
        self.shift_scaling_factor = shift_scaling_factor
        self.scaling_factor = scaling_factor
    def fit(self, X, y=None):
        self.mean_ = np.nanmean(np.array(X).flatten())
        return self
    def transform(self, X):
        return copy.deepcopy((X - self.mean_ * self.shift_scaling_factor) * self.scaling_factor)
    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)
    def predict(self, X):
        return self.transform(X)

class SciPyPiecewiseLinearRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, fill_value="extrapolate"):
        self.fill_value = fill_value  # Handles extrapolation
        self.interpolator = None

    def fit(self, X, y):
        X = np.array(X).flatten()
        y = np.array(y).flatten()
        
        assert len(np.unique(X)) == len(X), F'The array {X} should contain unique values!'
        # Sort X and y (required for interp1d)
        sorted_idx = np.argsort(X)
        X_sorted = X[sorted_idx]
        y_sorted = y[sorted_idx]
        
        # Create linear interpolator
        self.interpolator = interp1d(
            X_sorted, 
            y_sorted, 
            kind='linear', 
            fill_value=self.fill_value, 
            bounds_error=False
        )
        return self
    
    def transform(self, X): return self.predict(X)
    
    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    def predict(self, X):
        if self.interpolator is None:
            raise ValueError("Model not fitted yet!")
        return self.interpolator(np.array(X).flatten())
 
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
        y1 = _moving_average(y, mov_avg_width)
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
        logging.log(LOGLEVEL_DEBUG1, F'pivots={self.pivotlo},{self.pivothi}')
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
# Maybe TODO: this can be used as an activation function for neural networks. 
# However, I did not find any relevant work about the use of non-parametric curve as activation function for neural network. 
# Maybe the biggest problem is to design the back-propagation algorithm for such neural networks?

class IsotonicLogisticRegression(BaseEstimator, ClassifierMixin, RegressorMixin):

    def __init__(self,
            categorical_cols='auto',
            nontransformed_cols=[],
            increasing_cols=[],
            decreasing_cols=[],
            nonstrict_mono_cols=[],
            convex_cols=[],
            freeform_cols=[],

            task='classification',
            final_predictor=None, # (ElasticNetCV() if taks=='regression' else LogisticRegression()),
            final_pred_init_params={},

            random_state=-1,
            # adaKDE_* are only used when random_state<0
            adaKDE_min_width=2,
            adaKDE_width_adjust_factor=1.0, # 0.9 for Silverman's rule (but 0.5 in practice)
            adaKDE_exponent_inverse=3,
            adaKDE_freeform_min_width=36, #1e99,
            
            postCIR_mov_avg_window_size=0,
            
            fit_add_measure_error=None,
            transform_add_measure_error=None,
            ft_fit_add_measure_error=None,
            ft_transform_add_measure_error=None,
            
            fit_data_clear=False,
            
            set_feature_importances=True,
            effect_sizes=_EFFECT_SIZES,
            feat_effect_size_thres=0.15,
            feat_pvalue_method='auto', # mann-whitney-U and spearman for classification and regression, respectively
            feat_pvalue_thres=0.05,
            feat_pvalue_correction='bh', # the default one from https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.false_discovery_control.html            
            feat_pvalue_drop=True,
            feat_pvalue_warn=True,
            
            increasing = 'auto', # the default one from https://scikit-learn.org/stable/modules/generated/sklearn.isotonic.IsotonicRegression.html
            nan_policy = 'raise', # similar to https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html
            **kwargs):
        """ 
        Class for performing adaptive kernel density estimation (adaKDE), isotonic regression (IR), and centered IR.
        
        Parameters
        ----------

        categorical_cols: list of str or the special value ``auto``,
            columns to denote categorical features. The special str value ``auto`` means auto infer from the data (features with less than five distinct values are assumed to be categorical).

        nontransformed_cols: list of str
            columns to remain untransformed

        increasing_cols: list of str
            columns that are monotonically increasing with respect to the label

        decreasing_cols: list of str
            columns that are monotonically decreasing with respect to the label

        nonstrict_mono_cols: list of str
            columns that do not require the increase/decrease to be strictly monotonic

        convex_cols: list of str
            columns that are subject to convex regression (modeling by convex functions) instead of isotonic regression (monotonic functions)
        
        freeform_cols: list of str
            columns that are only transformed by adaKDE (instead of being transformed by isotonic (aka monotonic) or convex regression as the next step). 
        
        task: ``classification`` (default) or ``regression``

        final_predictor: scikit-learn predictor
            The final scikit-learn predictor (which has the predict method) to be used afer feature transformations, 
            defaults to LogisticRegression and LinearRegression with default params for classification and regression, respectively. 
        
        random_state: integer or RandomState instance
            The state for generating random numbers (just like the random_state from sklearn), -1 means disable_random=True
        
        adaKDE_min_width: float
            Inclusive minimum number of examples in each adaptive KDE of a response value (i.e., log odds. Only used when disable_random=True).
            If this value is negative, then auto infer the minimum number. 
        
        adaKDE_width_adjust_factor: float
            The adjustment of kernel size (similar to bw_adjust from https://seaborn.pydata.org/generated/seaborn.kdeplot.html)
            Factor that multiplicatively scales the value chosen with adaKDE_exponent_inverse. 
            Increasing this value will make the curve smoother. 
            Some rules of thumb:
              0.9 for Siverman's rule for normal-kernel smoothing normal-like PDF
        
        adaKDE_exponent_inverse: integer or float
            The multiplicative inverse of the exponent of the number of datapoints. 
            This value is used to shrink the kernel bandwidth as the number of datapoints increases. 
            This value is only used when disable_random=True.
            If this value is -1, then use adaKDE_min_width as the bandwidth for all features.
            The computation of the bandwidth implicitly assumes that the log odds vs contig (a set of feature values having the same label) ordinal follows a linear-like curve
            (i.e., the curve is globally linear but can be locally non-linear).
            Some rules of thumb:
              2 for any non-differentiable density (i.e., uniform distribution PDF) 
              3 for any once-differentiable (e.g., piecewise-linear) density derived from the minimax theory (e.g., https://doi.org/10.1007/978-0-387-79052-7_1, page 15)
              5 for Silverman's and Scott's rules for normal-kernel smoothing normal PDF.
        
        adaKDE_freeform_min_width:
            The minimum bandwith to disable isotonic and convex regression (because the bandwidth covered enough samples to estimate density without any constraint).
            This has the same effect as setting the relevant freeform_cols. 

        postCIR_mov_avg_window_size: integer
            If set to greater zero, then perform moving average with this value on the breakpoints resulting from CIR.
            This parameter is not applicable to categorical features.
 
        fit_add_measure_error: True or False
            If set to true, then introduce noise to the fit                      method to prevent overfitting. Empirical evidence supports its use for plain decision trees. 
        transform_add_measure_error: True or False
            If set to true, then introduce noise to the transform                method to prevent overfitting. This option is advanced. 
        ft_fit_add_measure_error: True or False
            If set to true, then introduce noise to the fit part of       fit_transform to prevent overfitting. This option is advanced. 
        ft_transform_add_measure_error: True or False
            If set to true, then introduce noise to the transform part of fit_transform to prevent overfitting. This option is advanced. 
        
        fit_data_clear: True or False
            Let the fit method perform clear_intermediate_internal_data at its end

        set_feature_importances: True or False
            If set to True, then will set feature importances during the fit. 
            Setting this to false may prevent a runtime error (due to some not-yet discovered bug) at the cost of not getting feature importance.

        effect_sizes: list of float
            List of effect sizes used for computing feature importances
                 
        feat_effect_size_thres: float
            The null hypothesis assumes that the effect size is greater than this threshold. 
            The features that deviates from this hypothesis are rejected with the feat_pvalue_thres. 
            The rejected features are filtered out if feat_pvalue_drop is set. 
            If set to zero, then the null hypothesis instead assumes that the effect size is zero, and the rejected features will be used instead. 
        
        feat_pvalue_thres: float
            the p-value for the null hypothesis that the label as a function of a feature is neither increasing nor decreasing
        
        feat_pvalue_correction: one of the valid strings from https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.false_discovery_control.html        
        
        feat_pvalue_drop: True or False
            Zero out the feature if the null hypothesis fails to hold at the p-value threshold of feat_pvalue_thres for the feature. 
            If the p-value threshold is <0 and >1, then always zero out and and keep unchanged the feature, respectively.
            It is highly recommended to use sklearn.feature_selection.VarianceThreshold to remove the features that are zeroed-out. 
            Typically, when all columns are in increasing_cols, decreasing_cols, or convex_cols, then feat_pvalue_drop should be set to False
            because these columns provide prior info to the relationship between the label and the features represented by these columns.

        feat_pvalue_warn: True or False
            When set to True, gives warning with warnings.warn if the null hypothesis fails to hold at the given p-value threshold of feat_pvalue_thres
       
        nan_policy: how to treat nan values in the input sample-times-feature matrix
                
        **kwargs: dict
            Maintain compatiblity with clone.
        """
        
        super().__init__()

        self.nontransformed_cols = nontransformed_cols
        self.categorical_cols = categorical_cols
        self.increasing_cols = increasing_cols
        self.decreasing_cols = decreasing_cols
        self.nonstrict_mono_cols = nonstrict_mono_cols
        self.convex_cols = convex_cols
        self.freeform_cols = freeform_cols
        
        self.task = task
        self.final_predictor = final_predictor
        self.final_pred_init_params = final_pred_init_params
        # Probability can be calibrated with:
        # n_splits=5, random_state=1, cccv_n_jobs=-1,
        # sklearn.calibration.CalibratedClassifierCV(estimator=None, *, method='sigmoid', cv=KFold(n_splits=5, shuffle=True, random_state=1), n_jobs=-1, ensemble=True)
        # sklearn.model_selection.KFold(n_splits=5, *, shuffle=False, random_state=None)
        # self.cccv = CalibratedClassifierCV(estimator=self._internal_predictor, method='isotonic', cv=KFold(n_splits=n_splits, shuffle=True, random_state=random_state), n_jobs=cccv_n_jobs, ensemble=True)
        # We tested calibration and confirmed that LogisticRegression is already well-calibrated, which is as expected from theory.
        self.random_state = random_state
        self.adaKDE_min_width = adaKDE_min_width
        self.adaKDE_width_adjust_factor = adaKDE_width_adjust_factor
        self.adaKDE_exponent_inverse = adaKDE_exponent_inverse
        self.adaKDE_freeform_min_width = adaKDE_freeform_min_width

        self.postCIR_mov_avg_window_size = postCIR_mov_avg_window_size
                
        self.fit_add_measure_error = fit_add_measure_error
        self.transform_add_measure_error = transform_add_measure_error
        self.ft_fit_add_measure_error = ft_fit_add_measure_error
        self.ft_transform_add_measure_error = ft_transform_add_measure_error
        
        self.fit_data_clear = fit_data_clear
        
        self.set_feature_importances = set_feature_importances
        self.effect_sizes = effect_sizes
        self.feat_effect_size_thres = feat_effect_size_thres
        self.feat_pvalue_method = feat_pvalue_method
        self.feat_pvalue_thres = feat_pvalue_thres
        self.feat_pvalue_warn = feat_pvalue_warn
        self.feat_pvalue_drop = feat_pvalue_drop
        self.feat_pvalue_correction = feat_pvalue_correction
        
        self.increasing = increasing
        self.nan_policy = nan_policy
        self.kwargs = kwargs

    def __sklearn_is_fitted__(self):
        """
        Check fitted status and return a Boolean value.
        """
        return hasattr(self, "_is_fitted") and self._is_fitted
    
    def set_random_state(self, random_state):
        self.random_state = random_state
    def custom_get_params(self):
        """ Recursively get the params of this model """
        return {'log_OR': self.logORX, 'LogisticRegression.params': self._internal_predictor.get_params(), 'IsotonicRegression.params' : [ir.get_params() for ir in self.mat_x2y_regs_0_]}
        return ret
    def clear_intermediate_internal_data(self, steps=[0,1,2]):
        if 0 in steps:
            self.X0_ = None
            self.X1_ = None
            self.raw_log_odds_ = None
        if 1 in steps:
            self.mat_x_values_1_ = None
            self.mat_x2y_regs_1_ = None
            self.mat_y_values_1_ = None
        if 2 in steps:
            self.mat_x_values_2_ = None
            self.mat_x2y_regs_2_ = None
            self.mat_y_values_2_ = None
    def get_info(self):
        """ Recursively get the fitted params of this model """
        int_pred = self._internal_predictor
        if self.task == 'regression':
            int_pred_info = [int_pred.coef_, int_pred.intercept_, int_pred.n_features_in_]
        else:
            int_pred_info = [int_pred.classes_, int_pred.coef_, int_pred.intercept_, int_pred.n_features_in_, int_pred.n_iter_]
        isor_info = []
        for i in range(len(self.mat_x2y_regs_0_)):
            ir = self.mat_x2y_regs_0_[i]
            isor_info.append([ir.X_min_, ir.X_max_, ir.X_thresholds_, ir.y_thresholds_, ir.f_, ir.increasing_])
        return [int_pred_info, isor_info]
   
    ''' 
    def _split(self, X, is_already_splitted=False):
        X1 = np.array(X)
        if not is_already_splitted:
            nontransformed_cols = set(self.nontransformed_cols)
            ex_colidxs = []            
            for colidx in range(X1.shape[1]):
                if colidx in self.nontransformed_cols:
                    ex_colidxs.append(i)
            if hasattr(X, 'columns'):
                for colidx, colname in enumerate(X.columns):
                    if colname in self.nontransformed_cols:
                        ex_colidxs.append(colidx)
            self.ex_colidxs = sorted(list(set(ex_colidxs)))
        ex_colidxs = self.ex_colidxs
        in_colidxs = [colidx for colidx in range(X1.shape[1]) if (not colidx in ex_colidxs)]
        if hasattr(X, 'iloc'):
            return X.iloc[:,in_colidxs], X.iloc[:,ex_colidxs], in_colidxs, ex_colidxs
        else:
            return X1[:,in_colidxs], X1[:,ex_colidxs], in_colidxs, ex_colidxs
    '''
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
    
    def _prep_input(self, X):
        arr = copy.deepcopy(X)
        if self.nan_policy == 'mean':
            col_means = np.nanmean(arr, axis=0)
            nan_indices = np.isnan(arr)
            arr[nan_indices] = np.take(col_means, np.where(nan_indices)[1])
        return arr
    def _assert_input(self, X, y, is_num_asserted=True, is_binary_clf_asserted=True):
        for rowit in range(X.shape[0]):
            for colit in range(X.shape[1]):
                if is_num_asserted:
                    if self.nan_policy in ['assert', 'raise']: assert not math.isnan(X[rowit][colit]), F'Nan value encountered in row {rowit} col {colit} ({X[rowit]})'
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
    
    def _get_feature_importances(self, analysis, stat_test):
        def adjust_for_categorical(v1, v2): return np.where(self.are_variables_categorical_, v2, v1)
        
        if analysis == 'categorical': return self.are_variables_categorical_
        
        if stat_test == 'auto':
            if self.task == 'regression': stat_test = 'spearmanr'
            elif self.task == 'classification': stat_test = 'mannwhitneyu'
            else: raise RuntimeError(F'The self.task {self.task} is invalid (only "classification" and "regression" are valid)!')

        stat_test_rev_pvalue = None
        if   stat_test == 'spearmanr': 
            stat_test_retval = self.spearmanr_retval_
            stat_test_rev_pvalue = self.spearmanr_H0_assume_feat_is_useful_pval_
        elif stat_test == 'mannwhitneyu': 
            stat_test_retval = self.mannwhitneyu_retval_
            stat_test_rev_pvalue = self.mannwhitneyu_H0_assume_feat_is_useful_pval_
        elif stat_test == 'odds_spearmanr':
            OddsSpearmanrResult = collections.namedtuple('OddsSpearmanrResult', ['statistic', 'pvalue'])
            stat_test_retval = OddsSpearmanrResult(
                    statistic=np.array([x for x in self.feat_odds_spearman_rs_]),
                    pvalue=np.array([(0.5 if np.isnan(x) else x) for x in self.feat_odds_spearman_pvalues_]))
        else: raise TypeError(F'The statistical test {stat_test} is invalid (only "auto", "spearmanr" and "mannwhitneyu" are valid)!')

        adjusted_pvalues = adjust_for_categorical(stat_test_retval.pvalue, self.likeratio_retval_.pvalue)
        if stat_test_rev_pvalue:
            adjusted_rev_pvalue = {k : adjust_for_categorical(noncat_pvalue, self.likeratio_H0_assume_feat_is_useful_pval_[k].pvalue) for (k, noncat_pvalue) in stat_test_rev_pvalue.items()}

        assert ((0 <= stat_test_retval.pvalue) & (1>= stat_test_retval.pvalue)).all(), F'The pvalue {stat_test_retval.pvalue} is invalid for analysis={analysis} test={stat_test}'

        if   analysis == 'f2l2f':
            return self.feature_importances_
        elif analysis == 'f2l':
            return self.feature_importances_to_label_
        elif analysis == 'f2f':
            return self.feature_importances_to_features_
        elif analysis == 'statistic':
            return stat_test_retval.statistic
        elif analysis == 'pvalue':
            rejected, pvalue_corrected, m0, alpha_stages = fdrcorrection_twostage(adjusted_pvalues, self.feat_pvalue_thres)
            return pvalue_corrected # return scipy.stats.false_discovery_control(stat_test_retval.pvalue, method=self.feat_pvalue_correction)
        elif analysis == 'h0_assume_correlation_pvalue':
            assert stat_test == 'spearmanr' or stat_test == 'mannwhitneyu', F'Only the tests spearmanr and mannwhitneyu are valid for the test option h0_assume_correlation_pvalue!'
            return {k : fdrcorrection_twostage(v, self.feat_pvalue_thres)[1] for k, v in sorted(adjusted_rev_pvalue.items())}
            #return {k : scipy.stats.false_discovery_control(v, method=self.feat_pvalue_correction) for k, v in sorted(stat_test_rev_pvalue.items())}
            #return {k : scipy.stats.false_discovery_control(v, method=self.feat_pvalue_correction) for k, v in sorted(stat_test_rev_pvalue.items())}
        elif analysis == 'trend':
            rejected, pvalue_corrected, m0, alpha_stages = fdrcorrection_twostage(adjusted_pvalues, self.feat_pvalue_thres)
            return np.where(pvalue_corrected < self.feat_pvalue_thres, np.sign(stat_test_retval.statistic), 0)
            #return np.where(scipy.stats.false_discovery_control(stat_test_retval.pvalue, method=self.feat_pvalue_correction) < self.feat_pvalue_thres, np.sign(stat_test_retval.statistic), 0)
        else:
            raise TypeError(F'The importance type "{analysis}" is invalid, it must be either "f2l", "f2f", "f2l2f", "statistic", "pvalue", or "trend"!')

    def get_feature_importances(self, analysis='f2l2f', stat_test='auto'):
        """ Method to compute feature importance scores
            
            @param analysis: one of the following 1-D array in which each number corresponds to a feature
                "statistic": returning the observed statistic of each single feature by itself with the label when ignoring all other features;
                    the statistic can be either spearman-correlation-coefficient or wilxocon-rank-sum
                "pvalue": returning the P-value of each single feature by itself when ignoring all other features, with the null hypothesis of having statistic=0;
                "trend" : returning one of (-1,0,+1) for each single feature by itself, where the label is (decreasing,constant,increasing) as a function of the feature, when ignoring all other features;
                "f2l"   : feature-to-label, returning the average log odds contributed by each single feature by itself when ignoring all other features;
                "f2f"   : feature-to-features, returning the percent contribution of each feature in the ensemble of all features;
                "f2l2f" : feature-to-label multiplied by feature-to-feature, returning the average log odds contributed by each feature in the ensemble of all features.
                `h0_assume_correlation_pvalue`: dict of float to ndarray
                    mapping from effect size threshold to the p-value corresponding to the null hypothesis that the effect size is above this threshold. 
                    The effect sizes are Spearman correlation coefficient, Biserial correlation coefficient, and log-likelihood difference 
                    for continuous-continuous, continuous-binary, and categorial-continuous feature-label pairs, respectively.
                `categorical`: boolean list, in which each value indicates whether the input feature is categorical
            @param stat_test: one of "auto", "spearmanr", "mannwhitneyu", or "odds_spearmanr" for computing satistic, pvalue or trend:
                "auto"     : "mannwhitneyu" for classication and "spearmanr" for regression
                "mannwhitneyu" : scipy.stats.mannwhitneyu test
                "spearmanr": scipy.stats.spearmanr test
                "odds_spearmanr": scipy.stats.spearmanr test performed on the adaKDE-estimated odds (instead of the raw input values)                
            @return: an array in which each element denotes the importance of the corresponding feature
        """
        check_is_fitted(self)
        return self._get_feature_importances(analysis, stat_test)

    def _set_feature_importances(self, imp_types):
        if 'f2l' in imp_types:
            mu = np.log(self.get_odds_offset()) if self.task == 'classification' else self.get_average_y()
            self.feature_importances_to_label_ = np.array([np.mean([abs(x - mu) for x in self.mat_y_values_0_[colidx]]) for colidx, colname in enumerate(self.feature_names_in_)])
        if 'f2f' in imp_types:
            assert len(self._internal_predictor.coef_.shape) <= 2, F'The shape {self._internal_predictor.coef_.shape} of the coef_ {self._internal_predictor.coef_} of {self._internal_predictor} has more than two dimensions!'
            assert len(self._internal_predictor.coef_.shape) == 1 or self._internal_predictor.coef_.shape[0] == 1, F'The shape {self._internal_predictor.coef_.shape} of the coef_ {self._internal_predictor.coef_} of {self._internal_predictor} is invalid!'
            self.feature_importances_to_features_ = np.array([
                self._internal_predictor.coef_[colidx] if 1==len(self._internal_predictor.coef_.shape) else self._internal_predictor.coef_[0][colidx] 
                for colidx, colname in enumerate(self.feature_names_in_)])
        if 'f2l2f' in imp_types:
            self.feature_importances_ = self.feature_importances_to_label_ * self.feature_importances_to_features_
        return self

    def fit(self, X, y, *args,
            add_measure_error=None, data_clear=None, data_clear_steps=[0,1,2],
            # max_n_susbamples=100, n_iterations=100*25*20, # not used because Monte-Carlos simulation is not performed. 
            **kwargs):
        """
        Parameters
        ----------
        
        kwargs: dict
            Used only to maintain compatibility with other sklearn APIs. 
        """
        
        X_orig, y_orig = X, y
        #inX, exX, inIdxs, exIdxs = self._split(X_orig)
        X = np.array(X_orig)
        y = np.array(y_orig)

        effect_sizes = copy.deepcopy(self.effect_sizes)
        if self.feat_effect_size_thres not in effect_sizes: effect_sizes.append(self.feat_effect_size_thres)
        effect_sizes = sorted(effect_sizes)
        #if feat_pvalue_thres == None: feat_pvalue_thres = self.feat_pvalue_thres
        #if feat_pvalue_warn == None: feat_pvalue_warn = self.feat_pvalue_warn
        #if feat_pvalue_drop == None: feat_pvalue_drop = self.feat_pvalue_drop
        setof_nontransformed_cols = set(self.nontransformed_cols)
        logging.debug(F'setof_nontransformed_cols={setof_nontransformed_cols} from {self.nontransformed_cols}')
        setof_categorical_cols = set(self.categorical_cols)
        setof_increasing_cols = set(self.increasing_cols)
        setof_decreasing_cols = set(self.decreasing_cols)
        setof_nonstrict_mono_cols = set(self.nonstrict_mono_cols)
        setof_convex_cols = set(self.convex_cols)
        setof_freeform_cols = set(self.freeform_cols)

        if self.final_predictor:
            self._internal_predictor = self.final_predictor
        else:
            if self.task == 'regression':
                self._internal_predictor = LinearRegression(**self.final_pred_init_params)
                # self._internal_predictor = ElasticNetCV()
            else:
                # L-BFGS, the default logistic-regression (LR) solver used by the current sklearn version, generates different coefficients from the same training data on different machines
                # Therefore, we use liblinear, the default LR solver in the previous sklearn versions, to solve LR
                self._internal_predictor = LogisticRegression(random_state=0, solver='liblinear', **self.final_pred_init_params)
        self.irrelevant_feature_indexes_ = []
        
        #def triangular_kernel(val, mid, lo, hi): return max((0, ((val-lo) / (mid-lo) if (val < mid) else (hi-val) / (hi-mid))))
        #def heaviside_rectangular_kernel(val, mid, lo, hi): return (1 if (lo < val and val < hi) else (0.5 if (val == lo or val == hi) else 0))
        def powermean(arr, p=1): return 1.0/len(arr) * (sum(ele**p for ele in arr))**p
        add_measure_error = self.get_default(add_measure_error, self.fit_add_measure_error, False)
        data_clear = self.get_default(data_clear, self.fit_data_clear, False)
                
        inColumns = list(X_orig.columns) if hasattr(X_orig, 'columns') else ([''] * (X.shape[1]))
        self.feature_names_in_ = [(inColumn if inColumn != '' else inIdx) for inIdx, inColumn in enumerate(inColumns)] 

        if self.categorical_cols == 'auto':
            self.are_variables_categorical_ = [(len(set(list(X[:,i]))) <= 5) for i in range(len(inColumns))]
        else:
            self.are_variables_categorical_ = [(True if _is_any_in([colidx, colname], setof_categorical_cols) else False) for colidx, colname in enumerate(inColumns)]

        self.increasings_ = ['auto' for i in inColumns]
        for (inIdx, inColname) in enumerate(inColumns):
            if _is_any_in([inIdx, inColname], setof_increasing_cols):
                self.increasings_[inIdx] = True
            if _is_any_in([inIdx, inColname], setof_decreasing_cols):
                assert self.increasings_[inIdx] != True, F'The column {inColname} at index {inIdx} cannot be both increasing and decreasing!'
                self.increasings_[inIdx] = False
        self.increasings_ = np.array(self.increasings_)
        
        self.feat_odds_spearman_rs_ = [None for _ in range(X.shape[1])]
        self.feat_odds_spearman_pvalues_ = [None for _ in range(X.shape[1])]
        
        assert X.shape[0] > 0, F'The input {X} does not have any rows'
        assert X.shape[1] > 0, F'The input {X} does not have any columns'
        
        raw_log_odds = self.raw_log_odds_ = [None for _ in range(X.shape[1])]
        inv0 = self.mat_y2x_regs_0_ = [SciPyPiecewiseLinearRegressor() for i in range(X.shape[1])]
        
        ixs0 = self.mat_x_values_0_ = [[] for _ in range(X.shape[1])]
        irs0 = self.mat_x2y_regs_0_ = [[] for _ in range(X.shape[1])]
        ivs0 = self.mat_y_values_0_ = [[] for _ in range(X.shape[1])]
        ixs1 = self.mat_x_values_1_ = [[] for _ in range(X.shape[1])]
        irs1 = self.mat_x2y_regs_1_ = [IsotonicRegression(increasing=self.increasings_[i], out_of_bounds='clip') for i in range(X.shape[1])]
        ivs1 = self.mat_y_values_1_ = [[] for _ in range(X.shape[1])]
        ixs2 = self.mat_x_values_2_ = [[] for _ in range(X.shape[1])]
        irs2 = self.mat_x2y_regs_2_ = [SciPyPiecewiseLinearRegressor() for i in range(X.shape[1])]
        ivs2 = self.mat_y_values_2_ = [[] for _ in range(X.shape[1])]
        ixs3 = self.mat_x_values_3_ = [[] for _ in range(X.shape[1])]
        irs3 = self.mat_x2y_regs_3_ = [SciPyPiecewiseLinearRegressor() for i in range(X.shape[1])]
        ivs3 = self.mat_y_values_3_ = [[] for _ in range(X.shape[1])]

        self.feat_pvalue_method_ = self.feat_pvalue_method

        self.mannwhitneyu_H0_assume_feat_is_useful_pval_ = {}
        self.spearmanr_H0_assume_feat_is_useful_pval_ = {}

        self.average_y_ = np.mean(y)
        self.winsizes_X_ = [[] for _ in range(X.shape[1])]
        self.winsizes_ = [[] for _ in range(X.shape[1])]
        self.kernel_width_n_minority_samples_ = [-1 for _ in range(X.shape[1])]

        X = self._prep_input(X)        
        if self.task == 'regression':
            self._assert_input(X, y, is_binary_clf_asserted=False)
            self.X0_ = X[y<=np.mean(y),:]
            self.X1_ = X[y> np.mean(y),:]            
            self.prevalence_odds_ = np.nan
            if self.feat_pvalue_method == 'auto': self.feat_pvalue_method_ = 'spearmanr'
        elif self.task == 'classification':
            self._assert_input(X, y, is_binary_clf_asserted=True)
            self.X0_ = X[y==0,:]
            self.X1_ = X[y==1,:]
            self.prevalence_odds_ = len(self.X1_) / float(len(self.X0_))
            if self.feat_pvalue_method == 'auto': self.feat_pvalue_method_ = 'mannwhitneyu'
        else:
            raise TypeError(F'The task name {self.task} is invalid (only `classification` and `regression` are valid)!')        
        self.likeratio_retval_ = _likeratio2(self.X0_, self.X1_, axis=0)
        self.mannwhitneyu_retval_ = mannwhitneyu2(self.X0_, self.X1_, axis=0)
        self.spearmanr_retval_ = spearmanr2(X, y, axis=0)
        self.mannwhitneyu_H0_assume_feat_is_useful_pval_ = _approx_H0_assume_some_effect_size_pval(self.mannwhitneyu_retval_.statistic, len(X), effect_sizes, 
            self.mannwhitneyu_retval_.mu, self.mannwhitneyu_retval_.sd, 'mannwhitneyu')
        self.spearmanr_H0_assume_feat_is_useful_pval_ = _approx_H0_assume_some_effect_size_pval(self.spearmanr_retval_.statistic, len(X), effect_sizes, 
            [], [], 'spearmanr')
        self.likeratio_H0_assume_feat_is_useful_pval_ = {} 
        for effect_size in effect_sizes:
            self.likeratio_H0_assume_feat_is_useful_pval_[effect_size] = _likeratio2(self.X0_, self.X1_, axis=0, effect_size=effect_size)
        
        if self.feat_pvalue_method_ == 'mannwhitneyu':
            self.feat_pvalue_method_retval_ = self.mannwhitneyu_retval_
        elif self.feat_pvalue_method_ == 'spearmanr':
            self.feat_pvalue_method_retval_ = self.spearmanr_retval_
        else:
            raise TypeError(F'The p-value computation method {self.feat_pvalue_method_} (derived from {self.feat_pvalue_method}) is invalid'
                    F' (only `auto`, `mannwhitneyu`, and `spearmanr` are valid)!')

        for colidx, colname in enumerate(inColumns): 
            x = X[:,colidx]
            if self.task == 'regression':                
                xylist = sorted(zip(x,y))
                x1 = np.array([x for (x,y) in xylist])
                y1 = np.array([y for (x,y) in xylist])
                center_log_odds = 0
                self.raw_log_odds_[colidx] = y1
            else:
                xcenters = []
                xodds = []
                x = X[:,colidx]
                if self.random_state < 0:
                    # disable_random=True
                    contig_list = _transform_and_partition(x, y)
                    featval_x0cnt_x1cnt_list = [_center_group(contig) for contig in contig_list]
                    
                    if self.adaKDE_exponent_inverse == -1:
                        kernel_width_n_minority_samples = self.adaKDE_min_width
                    else:
                        # First adaKDE to estimate an exponential-like density function (DF)
                        log_odds_1 = []
                        for i, (featval, (x0cnt_orig, x1cnt_orig)) in enumerate(featval_x0cnt_x1cnt_list):
                            x0cnt = x0cnt_orig
                            x1cnt = x1cnt_orig
                            prev1_x0cnt, next1_x0cnt, prev1_x1cnt, next1_x1cnt = (0, 0, 0, 0)
                            delta = 1
                            while ((x0cnt - 0.5 * (prev1_x0cnt + next1_x0cnt) <= self.adaKDE_min_width-0.001)
                                or (x1cnt - 0.5 * (prev1_x1cnt + next1_x1cnt) <= self.adaKDE_min_width-0.001)):
                                prev1_featval, (prev1_x0cnt, prev1_x1cnt) = featval_x0cnt_x1cnt_list[abs(i-delta)]
                                next1_featval, (next1_x0cnt, next1_x1cnt) = featval_x0cnt_x1cnt_list[min((i+delta,2*len(featval_x0cnt_x1cnt_list)-i-2-delta))]
                                x0cnt += prev1_x0cnt + next1_x0cnt
                                x1cnt += prev1_x1cnt + next1_x1cnt
                                delta += 1
                            x0cnt -= 0.5 * (prev1_x0cnt + next1_x0cnt)
                            x1cnt -= 0.5 * (prev1_x1cnt + next1_x1cnt)
                            log_odds_1.append(np.log(x1cnt / x0cnt))
                        linreg = LinearRegression()
                        linreg.fit(np.array([[idx] for idx in range(len(log_odds_1))]), np.array(log_odds_1))

                        # Example: 0 1 00 1 0000 1 00000000 1 000000000000
                        # 9 contigs, 5 effective samples, mean=1/ln(2), 
                        kernel_width_n_contigs = self.adaKDE_width_adjust_factor * (len(log_odds_1))**(-1.0/self.adaKDE_exponent_inverse) * min([
                                1.0 / (sys.float_info.epsilon + abs(linreg.coef_[0])), len(log_odds_1)])
                        kernel_width_n_minority_samples = kernel_width_n_contigs # *2 for bandwidth-to-windowsize-conversion and then *0.5 for windowsize-to-nMinoritySample-conversion
                    kernel_width_n_minority_samples = max((self.adaKDE_min_width, kernel_width_n_minority_samples))
                    self.kernel_width_n_minority_samples_[colidx] = kernel_width_n_minority_samples
                    
                    # The following commented-out code is not theoretically valid but may give some insight for us in the future
                    '''
                    for i, (featval, (x0cnt_orig, x1cnt_orig)) in enumerate(featval_x0cnt_x1cnt_list):
                        # bias-variance trade-off (this kernel size should be optimal)
                        x0cnt = x0cnt_orig
                        x1cnt = x1cnt_orig
                        prev1_x0cnt, next1_x0cnt, prev1_x1cnt, next1_x1cnt = (0, 0, 0, 0)
                        delta = 1
                        x0cnt_sublist = [x0cnt]
                        x1cnt_sublist = [x1cnt]
                        while (x0cnt - 0.5 * (prev1_x0cnt + next1_x0cnt) < 16) or (x1cnt - 0.5 * (prev1_x1cnt + next1_x1cnt) < 16):
                            #if i-delta <  0:
                            #    (prev1_x0cnt, prev1_x1cnt) = (0, 0)
                            #else: _, (prev1_x0cnt, prev1_x1cnt) = featval_x0cnt_x1cnt_list[i-delta]
                            #if i+delta >= len(featval_x0cnt_x1cnt_list)
                            #    (next1_x0cnt, next1_x1cnt) = (0, 0)
                            #else: _, (next1_x0cnt, next1_x1cnt) = featval_x0cnt_x1cnt_list[i+delta]
                            prev1_featval, (prev1_x0cnt, prev1_x1cnt) = featval_x0cnt_x1cnt_list[abs(i-delta)]
                            next1_featval, (next1_x0cnt, next1_x1cnt) = featval_x0cnt_x1cnt_list[min((i+delta,2*len(featval_x0cnt_x1cnt_list)-i-2-delta))]

                            x0cnt += prev1_x0cnt + next1_x0cnt
                            x1cnt += prev1_x1cnt + next1_x1cnt
                            x0cnt2 = (x0cnt - 0.5 * (prev1_x0cnt + next1_x0cnt))
                            x1cnt2 = (x1cnt - 0.5 * (prev1_x1cnt + next1_x1cnt))
                            x0cnt_sublist.append(x0cnt2)
                            x1cnt_sublist.append(x1cnt2)
                            delta += 1
                        oddsratios = []
                        for i2, (x0cnt, x1cnt) in enumerate(zip(x0cnt_sublist, x1cnt_sublist)):
                            j2 = 2*i2+1
                            if j2 >= len(x1cnt_sublist):
                                oddsratios.append(-j2)
                            else:
                                inner_x0cnt = x0cnt_sublist[i2]
                                inner_x1cnt = x1cnt_sublist[i2]
                                outer_x0cnt = x0cnt_sublist[j2] - x0cnt_sublist[i2]
                                outer_x1cnt = x1cnt_sublist[j2] - x1cnt_sublist[i2]
                                oddsratio = (inner_x0cnt * outer_x1cnt + 1e-20) / (outer_x0cnt * inner_x1cnt + 1e-10)
                                #oddsratio = (x0cnt_sublist[i2] * x1cnt_sublist[j2] + 1e-20) / (x0cnt_sublist[j2] * x1cnt_sublist[i2] + 1e-10)
                                oddsratios.append(min((oddsratio, 1 / oddsratio)))
                        self.winsizes_X_[colidx].append(featval)
                        self.winsizes_[colidx].append(np.argmax(oddsratios))
                    '''
                    
                    # Second akaDKE to more accurate estimate log odds
                    for i, (featval, (x0cnt_orig, x1cnt_orig)) in enumerate(featval_x0cnt_x1cnt_list):
                        x0cnt = x0cnt_orig
                        x1cnt = x1cnt_orig
                        prev1_x0cnt, next1_x0cnt, prev1_x1cnt, next1_x1cnt = (0, 0, 0, 0)
                        delta = 1
                        adaKDE_width = math.floor(kernel_width_n_minority_samples)
                        '''
                        if self.adaKDE_min_width > 0:
                            adaKDE_width = math.floor(kernel_width_n_minority_samples)
                        else:
                            kcnt, ksum = 0, 0
                            kernel_window_sizes = []
                            for j in range(i-8, i+8+1):
                                if j >= 0 and j < len(featval_x0cnt_x1cnt_list):
                                    ksum += self.winsizes_[colidx][j]
                                    kcnt += 1
                                    kernel_window_sizes.append(self.winsizes_[colidx][j])
                            #adaKDE_width = (ksum / kcnt) * 0.999
                            adaKDE_width = np.median(np.array(kernel_window_sizes))
                            colname = self.feature_names_in_[colidx]
                            logging.log(LOGLEVEL_DEBUG1, F'Feature={colname}\tfeature_index={colidx}\tfeature_value={featval:g}\tadaKDE_width={adaKDE_width:g}')
                        '''
                        while ((x0cnt - 0.5 * (prev1_x0cnt + next1_x0cnt) <= adaKDE_width-0.001) 
                            or (x1cnt - 0.5 * (prev1_x1cnt + next1_x1cnt) <= adaKDE_width-0.001)):
                            #if i-delta <  0:
                            #    (prev1_x0cnt, prev1_x1cnt) = (0, 0)
                            #else: _, (prev1_x0cnt, prev1_x1cnt) = featval_x0cnt_x1cnt_list[i-delta]
                            #if i+delta >= len(featval_x0cnt_x1cnt_list):
                            #    (next1_x0cnt, next1_x1cnt) = (0, 0)
                            #else: _, (next1_x0cnt, next1_x1cnt) = featval_x0cnt_x1cnt_list[i+delta]
                            prev1_featval, (prev1_x0cnt, prev1_x1cnt) = featval_x0cnt_x1cnt_list[abs(i-delta)]
                            next1_featval, (next1_x0cnt, next1_x1cnt) = featval_x0cnt_x1cnt_list[min((i+delta,2*len(featval_x0cnt_x1cnt_list)-i-2-delta))]
                            
                            x0cnt += prev1_x0cnt + next1_x0cnt
                            x1cnt += prev1_x1cnt + next1_x1cnt
                            delta += 1
                        x0cnt -= 0.5 * (prev1_x0cnt + next1_x0cnt)
                        x1cnt -= 0.5 * (prev1_x1cnt + next1_x1cnt)
                        #assert x0cnt > 1.9, f'{x0cnt} > 1.9 failed for {featval_x0cnt_x1cnt_list} at {i}-th index (elemnt={featval, (x0cnt, x1cnt)})'
                        #assert x1cnt > 1.9, f'{x1cnt} > 1.9 failed for {featval_x0cnt_x1cnt_list} at {i}-th index (elemnt={featval, (x0cnt, x1cnt)})'
                        xcenter = featval
                        odds = x1cnt / x0cnt
                        xcenters.append(xcenter)
                        xodds.append(odds)
                else:
                    xord = self.ensure_total_order(x)
                    xylist = sorted(zip(xord,y)) #xylist = (self.total_order(x, y) if (len(set(x)) > 1) else sorted(zip(x,y)))
                    xylistlist = self.partition(xylist)                
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
                        odds = (powermean((len(curr_xylist), powermean((pre2_len, nex2_len))))) / (powermean((prev_len, next_len)))
                        xcenters.append(xcenter)
                        xodds.append((odds) if (ylabel == 1) else (1/odds))
                        prev_ylabel = ylabel
                raw_log_odds = np.log(xodds)
                center_log_odds = np.log(self.prevalence_odds_)                
                self.raw_log_odds_[colidx] = raw_log_odds
                x1 = np.array(xcenters)
                y1 = relative_log_odds = raw_log_odds - center_log_odds
                
            spearman_r, pvalue_observed = spearmanr(x1, y1)
            self.feat_odds_spearman_rs_[colidx] = spearman_r
            self.feat_odds_spearman_pvalues_[colidx] = pvalue_observed
            
            test_statistic = self._get_feature_importances('statistic', self.feat_pvalue_method_)[colidx]
            pvalue_observed = self._get_feature_importances('pvalue', self.feat_pvalue_method_)[colidx]
            is_inc_or_dec = self._get_feature_importances('trend', self.feat_pvalue_method_)[colidx]
            
            if _is_any_in([colidx, colname], setof_convex_cols):
                # if not colidx in self.convex_cols: self.convex_cols.append(colidx)
                self.mat_x2y_regs_1_[colidx] = ConvexRegression()
            if _is_any_in([colidx, colname], setof_nonstrict_mono_cols):
                is_centered = False
            else:
                is_centered = True

            x1a = x1
            self.mat_x_values_1_[colidx] = x1
            y1a = self.mat_x2y_regs_1_[colidx].fit_transform(x1, y1)
            self.mat_y_values_1_[colidx] = 1*center_log_odds + y1a
            
            if _is_any_in([colidx, colname], setof_nontransformed_cols):
                scaling_factor = (1 if (test_statistic >= 0) else -1)
                self.mat_x_values_0_[colidx] = x1
                self.mat_x2y_regs_0_[colidx] = ScalingRegressor1D(scaling_factor=scaling_factor).fit(x1) # ColumnTransformer([], remainder='passthrough')
                self.mat_y_values_0_[colidx] = self.mat_x2y_regs_0_[colidx].transform(x1)
                self.mat_y2x_regs_0_[colidx] = ScalingRegressor1D(scaling_factor=scaling_factor).fit(x1)
            elif len(set(x1a)) == 1 or len(set(y1a)) == 1:
                self.mat_x_values_0_[colidx] = [0] # x-values
                self.mat_x2y_regs_0_[colidx] = AlwaysConstantRegressor(0)  # regressors
                self.mat_y_values_0_[colidx] = [0] # y-value
                self.mat_y2x_regs_0_[colidx] = AlwaysConstantRegressor(0) # inverse regressors            
            elif (not is_centered):
                self.mat_x_values_0_[colidx] = self.mat_x_values_1_[colidx] # x-values
                self.mat_x2y_regs_0_[colidx] = self.mat_x2y_regs_1_[colidx] # regressors
                self.mat_y_values_0_[colidx] = self.mat_y_values_1_[colidx] # y-value
                self.mat_y2x_regs_0_[colidx] = copy.deepcopy(self.mat_x2y_regs_0_[colidx]).fit(y1, x1) # inverse regressors
            else:
                x2, y2 = self._center(x1, self.mat_x2y_regs_1_[colidx].predict(x1))
                
                self.mat_x_values_2_[colidx] = x2
                y2a = self.mat_x2y_regs_2_[colidx].fit_transform(x2, y2)
                self.mat_y_values_2_[colidx] = 1*center_log_odds + y2a

                if (self.postCIR_mov_avg_window_size <= 0) or self.are_variables_categorical_[colidx]:
                    self.mat_x_values_0_[colidx] = self.mat_x_values_2_[colidx]
                    self.mat_x2y_regs_0_[colidx] = self.mat_x2y_regs_2_[colidx]
                    self.mat_y_values_0_[colidx] = self.mat_y_values_2_[colidx]
                    self.mat_y2x_regs_0_[colidx].fit_transform(y2, x2)
                else:
                    x2list, y2list = list(x2), list(y2)
                    eps1 = 3 * sys.float_info.epsilon * abs(x1a[0])
                    eps2 = 3 * sys.float_info.epsilon * abs(x1a[-1])
                    x2list = [x1a[0]-eps1] + x2list + [x1a[-1]+eps2]
                    y2list = [y1a[0]]      + y2list + [y1a[-1]]
                    x3 = _moving_average(np.array(x2list), self.postCIR_mov_avg_window_size)
                    y3 = _moving_average(np.array(y2list), self.postCIR_mov_avg_window_size)
                    
                    self.mat_x_values_3_[colidx] = x3
                    y3a = self.mat_x2y_regs_3_[colidx].fit_transform(x3, y3)
                    self.mat_y_values_3_[colidx] = 1*center_log_odds + y3a
                    
                    self.mat_x_values_0_[colidx] = self.mat_x_values_2_[colidx]
                    self.mat_x2y_regs_0_[colidx] = self.mat_x2y_regs_3_[colidx]
                    self.mat_y_values_0_[colidx] = self.mat_y_values_2_[colidx]
                    self.mat_y2x_regs_0_[colidx].fit_transform(y3, x3)
            is_freeform_1 = _is_any_in([colidx, colname], setof_freeform_cols)
            is_freeform_2 = (self.kernel_width_n_minority_samples_[colidx] >= self.adaKDE_freeform_min_width)
            if is_freeform_1 or is_freeform_2:
                self.mat_x_values_0_[colidx] = x1 # x-values
                self.mat_x2y_regs_0_[colidx] = SciPyPiecewiseLinearRegressor().fit(x1, y1) # regressors
                self.mat_y_values_0_[colidx] = y1 # y-value
                #self.mat_y2x_regs_0_[colidx] = SciPyPiecewiseLinearRegressor().fit_transform(y1, x1) # inverse may not exist

        if self.set_feature_importances:
            self._set_feature_importances(['f2l'])
            
            self.irrelevant_feature_indexes_ = []
            effect_size_to_pvals = self._get_feature_importances('h0_assume_correlation_pvalue', self.feat_pvalue_method_)
            effect_size_eq0_pvals = self._get_feature_importances('pvalue', self.feat_pvalue_method_)
            for colidx, colname in enumerate(inColumns):
                pvalue = effect_size_to_pvals[self.feat_effect_size_thres][colidx]
                
                assuming_some_effect_fails = (effect_size_to_pvals[self.feat_effect_size_thres][colidx] < self.feat_pvalue_thres)
                assuming_zero_effect_fails = (effect_size_eq0_pvals[colidx] > self.feat_pvalue_thres)
                assumption_fails = (assuming_zero_effect_fails if self.feat_effect_size_thres == 0 else assuming_some_effect_fails)
                
                if assumption_fails:
                    self.irrelevant_feature_indexes_.append(colidx)
                    #assert effect_size_to_pvals[self.feat_effect_size_thres][colidx]
                    if self.feat_pvalue_warn:
                        pval = effect_size_to_pvals[self.feat_effect_size_thres][colidx]
                        if self.feat_pvalue_drop:
                            warnings.warn(F'The feature {colname} at column index {colidx} seems to be irrelevant and is dropped (not kept) at '
                                    + F'pvalue={pval} pvalue_thres={self.feat_pvalue_thres} ES={self.feat_effect_size_thres}. ')
                        else:
                            warnings.warn(F'The feature {colname} at column index {colidx} seems to be irrelevant but is still kept (not dropped) at '
                                    + F'pvalue={pval} pvalue_thres={self.feat_pvalue_thres} ES={self.feat_effect_size_thres}. ')
                    if self.feat_pvalue_drop and not _is_any_in([colidx, colname], setof_nontransformed_cols):
                        self.mat_x2y_regs_0_[colidx] = AlwaysConstantRegressor(0)
        
        log_ratios = self._transform(X, add_measure_error=add_measure_error, is_inverse=False)
        #self._internal_predictor.fit(np.hstack([log_ratios, exX]), y, **kwargs)
        self._internal_predictor.fit(log_ratios, y)
        if data_clear: self.clear_intermediate_internal_data(data_clear_steps)
        self.n_features_in_ = X.shape[1]
        if self.set_feature_importances: self._set_feature_importances(['f2f', 'f2l2f'])
        
        self._is_fitted = True
        return self
    
    def get_average_y(self):
        return self.average_y_
    def get_odds_offset(self):
        return self.prevalence_odds_
    def get_density_estimated_X(self):
        return self.mat_x_values_1_
    def get_density_estimated_log_odds(self):
        return self.raw_log_odds_
    def get_isotonic_X(self):
        return self.mat_x_values_1_
    def get_isotonic_log_odds(self):
        return self.mat_y_values_1_
    def get_centered_isotonic_X(self):
        return self.mat_x_values_2_
    def get_centered_isotonic_log_odds(self):
        return self.mat_y_values_2_
    def get_centered_2_X(self):
        return self.mat_x_values_3_
    def get_centered_2_log_odds(self):
        return self.mat_y_values_3_
    def get_final_pre_transformed(self):
        return self.mat_x_values_0_
    def get_final_post_transformed(self):
        return self.mat_y_values_0_

    def get_kernel_width_covered_n_positives(self):
        return copy.deepcopy(self.kernel_width_n_minority_samples_)
    def get_adaDKE_X(self):
        check_is_fitted(self)
        return self.winsizes_X_
    def get_adaDKE_width(self):
        check_is_fitted(self)
        return self.winsizes_
 
    def _transform(self, X, add_measure_error, is_inverse, column_idx=None):
        if column_idx != None:
            if column_idx in self.ex_colidxs:
                return X
            elif is_inverse:
                return self.mat_y2x_regs_0_[column_idx].transform(X)
            else:
                return self.mat_x2y_regs_0_[column_idx].transform(X)
                #return (self.mat_y2x_regs_0_[column_idx].transform(X) if is_inverse else (
                #self.convex_regressions_0_[column_idx].transform(X)
                #if (column_idx in self.convex_cols)
                #else self.mat_x2y_regs_0_[column_idx].predict(X)))
        if add_measure_error:
            XT = np.array([self.ensure_total_order(X[:,colidx]) for colidx in range(X.shape[1])])
        else:
            XT = np.array([(X[:,colidx]) for colidx in range(X.shape[1])])
        assert len(XT) == len(self.mat_x2y_regs_0_), F'{len(XT)} == {len(self.mat_x2y_regs_0_)} failed!'
        
        return np.array([(
            self.mat_y2x_regs_0_[colidx].transform(xT) if is_inverse else (
            self.mat_x2y_regs_0_[colidx].transform(xT)
            )) for colidx,xT in enumerate(XT)]).transpose()
        '''
        return np.array([(
            self.mat_y2x_regs_0_[colidx].transform(xT) if is_inverse else (
            self.convex_regressions_0_[colidx].transform(xT)
            if (colidx in self.convex_cols or (hasattr(X, 'columns') and X.columns[colidx] in self.convex_cols))
            else self.mat_x2y_regs_0_[colidx].predict(xT)
            )) for colidx,xT in enumerate(XT)]).transpose()
        '''
    def transform(self, X1, add_measure_error=None, is_inverse=False, column_idx=None):
        """ scikit-learn transform 
            add_measure_error: set to True to add measurement error to prevent overfitting
            is_inverse: set to True to perform inverse transform. Please use the inverse_transform method instead if possible.
        """
        check_is_fitted(self)
        add_measure_error = self.get_default(add_measure_error, self.transform_add_measure_error, False)
        X = np.array(X1)
        X = self._prep_input(X)
        if column_idx != None: return self._transform(X, add_measure_error=add_measure_error, is_inverse=is_inverse, column_idx=column_idx)
        return self._transform(X, add_measure_error=add_measure_error, is_inverse=is_inverse)
        #inX, exX, inIdxs, exIdxs = self._split(X, True)
        #test_orX = self._transform(inX, add_measure_error=add_measure_error, is_inverse=is_inverse)
        #X2 = np.zeros((X.shape[0], X.shape[1]))
        #X2[:,inIdxs] = test_orX
        #X2[:,exIdxs] = exX
        #return X2
    
    def inverse_transform(self, X1, column_idx=None):
        """
        scikit-learn inverse_transform
        caveat: inverse_transform(transform(X)) != X and transform(inverse_transform(X)) != X for a column x of X 
                if at least one scalar value of x is not within the range in which the transform function of x is monotonically increasing
        """
        return self.transform(X1, column_idx=column_idx, is_inverse=True)
    
    def fit_transform(self, X1, y1, *args, fit_add_measure_error=None, transform_add_measure_error=None, **kwargs):
        """ scikit-learn fit_transform 
            fit_add_measure_error: set to True to add measurement error to prevent overfitting (it may work for plain decision trees)
            transform_add_measure_error: set to True to add measurement error to prevent overfitting
        """
        # NOTE: Setting add_measure_error=True may improve the performance of some ML methods 
        #   such as DecisionTreeClassifier (DT) presumably because DT without regularization tends to overfit.
        fit_add_measure_error = self.get_default(fit_add_measure_error, self.ft_fit_add_measure_error, False)
        transform_add_measure_error = self.get_default(transform_add_measure_error, self.ft_transform_add_measure_error, True)
        self.fit(X1, y1, *args, add_measure_error=fit_add_measure_error, **kwargs)
        return self.transform(X1, add_measure_error=transform_add_measure_error)
    
    def _extract_features(self, X):
        #inX, exX, inIdxs, exIdxs = self._split(X, True)
        #test_orX = self._transform(inX, add_measure_error=False, is_inverse=False)
        #return np.hstack([test_orX, exX])
        return self._transform(X, add_measure_error=False, is_inverse=False)
 
    def predict(self, X1):
        """ scikit-learn predict using logistic regression built on top of isotonic scaler """
        check_is_fitted(self)
        X = np.array(X1)
        X = self._prep_input(X)
        allfeatures = self._extract_features(X)
        return self._internal_predictor.predict(allfeatures)

    def predict_proba(self, X1):
        """ scikit-learn predict_proba using logistic regression built on top of isotonic scaler """
        check_is_fitted(self)
        X = np.array(X1)
        X = self._prep_input(X)
        allfeatures = self._extract_features(X)
        if self.task == 'regression':
            ret = self._internal_predictor.predict(allfeatures)
            return np.array([(x,x) for x in ret])
        return self._internal_predictor.predict_proba(allfeatures)

def test_fit_and_predict_proba(ilr=None):
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
    
    if not ilr: ilr = IsotonicLogisticRegression(feat_pvalue_thres=2.0, nontransformed_cols=['col3'])
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

def test_fit_and_predict_with_dups(ilr=None, task='classification'):
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

    if not ilr: ilr = IsotonicLogisticRegression(feat_pvalue_thres=2.0, nontransformed_cols=['col3'], task=task)
    ilr.set_random_state(42+0)
    logging.info(F'self_nontransformed_cols={ilr.nontransformed_cols}')
    X2 = ilr.fit_transform(X, y)
    X3 = ilr.transform(X)
    y1 = ilr.predict(X)
    ilr.set_random_state(42+1)
    x2 = ilr.ensure_total_order(X.iloc[:,0])
    ordered_xs = list(zip(X.iloc[:,0],x2))
    print(F'train_ordered_X={ordered_xs}')
    print(F'train_transformed_X=\n{X2}')
    print(F'test_transformed_X=\n{X3}')

def test_fit_and_predict_with_convex(ilr=None, task='classification'):
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
    
    if not ilr : ilr = IsotonicLogisticRegression(feat_pvalue_thres=2.0, nontransformed_cols=['col3'],convex_cols=['col2'], task='regression')
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

def test_inverse_transform(ilr=None, task='classification'):
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
    if not ilr: ilr = IsotonicLogisticRegression(feat_pvalue_thres=2.0, task=task)
    ilr.set_random_state(42+0)
    ilr.fit(Xtrain, ytrain)
    X2 = ilr.transform(Xtest)
    X3 = ilr.inverse_transform(X2)
    np.testing.assert_allclose(X3, Xtest, rtol=1e-6, atol=1e-6)

def test_with_simulation(tasks=['classification', 'regression'], n_samples_s=[2**8], seeds=list(range(10))):
    odds_spearmanr_PVALUE_THRES = 1e-9
    import itertools, random, sys
    import numpy as np
    import scipy as sp
    import pandas as pd
    logging.basicConfig(format='test_with_simulation %(asctime)s - %(message)s', level=logging.INFO)
    def create_random_typed_mat(x):
        r = random.random() 
        if r < 1.0/3: return list(x)
        elif r < 2.0/3: return np.array(x)
        else: 
            if 1 == len(x.shape): return pd.Series(x, ['vector_name_{}'.format(i+1) for i in range(x.shape[0])])
            elif 2 == len(x.shape): return pd.DataFrame(x, columns=['matrix_column_{}'.format(i+1) for i in range(x.shape[1])])
    
    ilr_default = IsotonicLogisticRegression() # check that the default constructor without any param works
    for task, n_samples, seed_val in itertools.product(tasks, n_samples_s, seeds): #['classification', 'regression']:
        info1 = F'task={task} n_samples={n_samples} seed_val={seed_val}'

        random.seed(seed_val)
        np.random.seed(seed_val)
        drop_feat = seed_val%10//5
        increasings = (True, False, True, (True if seed_val%4//2 else False), (True if seed_val%4%2 else False)) if (seed_val%5) else []
        increasing_cols, decreasing_cols = [], []
        for i, inc in enumerate(increasings):
            if inc==True: increasing_cols.append(i)
            if inc==False: decreasing_cols.append(i)
        # NOTE: some common cause of test-run failure:
        # feat_pvalue_thres being too high or feat_effect_size_thres being too high
        # C being too high (e.g., C=1)
        if task == 'classification':
            lr_predictor = LogisticRegression(C=0.1)
            ilr = IsotonicLogisticRegression(task=task, feat_pvalue_thres=1e-3, feat_effect_size_thres=0.10, feat_pvalue_drop=drop_feat, final_predictor=lr_predictor, increasing_cols=increasing_cols, decreasing_cols=decreasing_cols)
        else:
            ilr = IsotonicLogisticRegression(task=task, feat_pvalue_thres=1e-3, feat_effect_size_thres=0.10, feat_pvalue_drop=drop_feat, increasing_cols=increasing_cols, decreasing_cols=decreasing_cols)
        Xtrain = np.array([[i, -i, i + 0.2*float(scipy.stats.norm.rvs(size=1)), 0, 0.2*float(scipy.stats.norm.rvs(size=1))] for i in range(n_samples)])
        ytrain_odds = np.exp((np.array(range(n_samples)) * 2 - (n_samples-1)) * 5 / n_samples) # 2**2 / (1 + np.array(range(n_samples)))

        if task == 'classification':
            ytrain_probs = ytrain_odds / (1 + ytrain_odds)
            ytrain = scipy.stats.bernoulli.rvs(ytrain_probs)
        else:
            ytrain = ytrain_odds

        
        for_trans_vals = ilr.fit_transform(create_random_typed_mat(Xtrain), create_random_typed_mat(ytrain))
        inv_trans_vals = ilr.inverse_transform(create_random_typed_mat(for_trans_vals))
        inv_trans_vals2 = np.array(inv_trans_vals)
        logging.log(LOGLEVEL_DEBUG1, F'inv_trans_vals2={inv_trans_vals2}\nXtrain={Xtrain}')
        ind = int(n_samples - n_samples/6)
        assert np.allclose(inv_trans_vals2[ind:-ind,:], Xtrain[ind:-ind,:]), F'{inv_trans_vals[ind:-ind,:]} == {Xtrain[ind:-ind,:]} failed!'            
        
        #logging.info(F'Are_variable_categorical={ilr.are_variables_categorical_}')
        if task == 'classification':
            Xpred_odds = np.exp(for_trans_vals) * ilr.get_odds_offset()
            ypred_probas = ilr.predict_proba(create_random_typed_mat(Xtrain))
            #print(F'Predicted probabilities {ypred_probas} from {Xtrain}')
        else:
            Xpred_odds = for_trans_vals
        ypred_labels = ilr.predict(create_random_typed_mat(Xtrain))
        
        logging.log(LOGLEVEL_DEBUG1, F'Xtrain=\n{Xtrain}')
        logging.log(LOGLEVEL_DEBUG1, F'ytrain=\n{ytrain}')
        logging.log(LOGLEVEL_DEBUG1, F'mannwhitneyu_retval={ilr.mannwhitneyu_retval_}')
        logging.log(LOGLEVEL_DEBUG1, F'spearmanr_retval={ilr.spearmanr_retval_}')
        logging.log(LOGLEVEL_DEBUG1, F'Xpred_odds=\n{Xpred_odds}')
        logging.info(F'>>> Kernel_sizes={ilr.get_kernel_width_covered_n_positives()}')

        rt, frac2, rt2, frac3, rt3, frac4, rt4 = 3.0, 0.95, 2.5, 0.85, 2.0, 0.7, 1.5  # relative tolerance
        edgedist = 0 # this seems to be irrelevant
        def to_display(a1, a2):
            ret = ([[i, round(e1,5), round(e2,5)] for i, (e1, e2) in enumerate(zip(a1, a2))])
            return ret if len(a1) < 500 else ret[0:200] + ['...', '...', '...'] + ret[-200:]
        def check_close(a1, a2, rtol1, frac2, rtol2, frac3, rtol3, frac4, rtol4, is_reverse=False):
            assert len(a1) == len(a2), F'Internal error: {a1} and {a2} are not equal in length!'
            a1 = np.log(a1)
            a2 = np.log(a2)
            if is_reverse: msg = 'succeeded (but should fail)'
            else: msg = 'failed (but should succeed)'
            assert is_reverse != (np.allclose(a1, a2, atol=rtol1)), (
                    F'for seed={seed_val} and n_samples={n_samples}, allclose with ln(tolerance)={rtol1} {msg}!\n{to_display(a1, a2)}')
            assert is_reverse != (sum(np.isclose(a1, a2, atol=rtol2)) >= len(a1) * frac2), (
                    F'for seed={seed_val} and n_samples={n_samples}, isclose with ln(tolerance)={rtol2} at {frac2*100}% {msg}!\n{to_display(a1, a2)}')
            assert is_reverse != (sum(np.isclose(a1, a2, atol=rtol3)) >= len(a1) * frac3), (
                    F'for seed={seed_val} and n_samples={n_samples}, isclose with ln(tolerance)={rtol3} at {frac3*100}% {msg}!\n{to_display(a1, a2)}')
            assert is_reverse != (sum(np.isclose(a1, a2, atol=rtol4)) >= len(a1) * frac4), (
                    F'for seed={seed_val} and n_samples={n_samples}, isclose with ln(tolerance)={rtol3} at {frac3*100}% {msg}!\n{to_display(a1, a2)}')
        def compute_n_monotonic_violations(xarr):
            n_gt, n_lt, n_eq = 0, 0, 0
            for i in range(0, len(xarr)-1, 1):
                for j in range(i+1, len(xarr), 1):
                    if   xarr[i] > xarr[j]: n_gt += 1
                    elif xarr[i] < xarr[j]: n_lt += 1
                    else: n_eq += 1
            return n_gt, n_lt
        check_close(Xpred_odds [edgedist:-1-edgedist:1,0], ytrain_odds [edgedist:-1-edgedist:1], rt, frac2, rt2, frac3, rt3, frac4, rt4)
        check_close(Xpred_odds [edgedist:-1-edgedist:1,1], ytrain_odds [edgedist:-1-edgedist:1], rt, frac2, rt2, frac3, rt3, frac4, rt4)
        check_close(Xpred_odds [edgedist:-1-edgedist:1,2], ytrain_odds [edgedist:-1-edgedist:1], rt*1.5, frac2, rt2*1.5, frac3, rt3*1.5, frac4, rt4*1.5) 
        if task == 'classification':
            #print(f'ypred_probas={ypred_probas} ytrain_probs={ytrain_probs}')
            check_close(ypred_probas[edgedist:-1-edgedist:1,1], ytrain_probs[edgedist:-1-edgedist:1], rt, frac2, rt2, frac3, rt3, frac4, rt4)
            rmat = [list(x) + [y] for x, y in zip(list(Xtrain), list(ypred_labels))]
            rmat='\n'.join(str(_) for _ in rmat)
            n_gt, n_lt = compute_n_monotonic_violations(ypred_labels)
            assert min(n_gt, n_lt) / (n_gt + n_lt) < 1e-3, f'{list(ypred_labels)} violated multiple monotonicity constraints with {info1} with data=\n{rmat}\nwith coefs={ilr._internal_predictor.coef_}\nn_gt, n_lt = {(n_gt, n_lt)}'
        else:
            check_close(ypred_labels[edgedist:-1-edgedist:1  ], ytrain_odds[edgedist:-1-edgedist:1], rt, frac2, rt2, frac3, rt3, frac4, rt4)
        check_close(Xpred_odds [edgedist:-1-edgedist:1,3], ytrain_odds [edgedist:-1-edgedist:1], rt, frac2, rt2, frac3, rt3, frac4, rt4, is_reverse=True)
        
        def aeq(a, b, tol=1.0): return abs(a-b) < sys.float_info.min or (a / b <= 1+tol and a/b >= 1/(1+tol))
        def beq(a, b): return aeq(a, b, 100.0)
        
        # For (mannwhitneyu, spearmanr, odds_spearmanr)
        def check_statistic(fi, approx_ranges=None, stat_method='NA'):
            if (np.isnan(fi[3]) and stat_method in ['spearmanr', 'odds_spearmanr']): fi[3] = 0
            assert aeq(fi[0] ,    -fi[1]), F'{fi[0]} ~    {-fi[1]}  failed for stat_test={stat_method} {info1}!'
            assert 1.1*fi[0] >     fi[2],  F'{fi[0]} >     {fi[2]}  failed for stat_test={stat_method} {info1}!'
            assert     fi[2] > abs(fi[3]), F'{fi[2]} > abs({fi[3]}) failed for stat_test={stat_method} {info1}!'
            assert     fi[2] > abs(fi[4]), F'{fi[2]} > abs({fi[4]}) failed for stat_test={stat_method} {info1}!'
            if approx_ranges:
                for i, (lo, hi) in enumerate(approx_ranges):
                    assert lo <= fi[i] and fi[i] <= hi, F'{lo} <= {fi[i]} <= {hi} failed for {fi} at {i} for stat_test={stat_method} {info1}'
        def check_pvalue(fi, approx_ranges=None, stat_method='NA'): 
            if (np.isnan(fi[3]) and stat_method in ['spearmanr', 'odds_spearmanr']): fi[3] = 0.5
            mul = odds_spearmanr_PVALUE_THRES if stat_method in ['odds_spearmanr'] else 0.1
            assert beq(fi[0] ,     fi[1]), F'{fi[0]} ~     {fi[1]}  failed for stat_test={stat_method} {info1}!'
            assert mul*fi[0] <=    fi[2],  F'{fi[0]} <     {fi[2]}  failed for stat_test={stat_method} {info1}!'
            assert     fi[2] <    (fi[3]), F'{fi[2]} < abs({fi[3]}) failed for stat_test={stat_method} {info1}!'
            assert     fi[2] <    (fi[4]), F'{fi[2]} < abs({fi[4]}) failed for stat_test={stat_method} {info1}!'
            if approx_ranges:
                for i, (lo, hi) in enumerate(approx_ranges):
                    assert lo <= fi[i] and fi[i] <= hi, F'{lo} <= {fi[i]} <= {hi} failed for {fi} at {i} for stat_test={stat_method} {info1}'
        def check_trend(fi, approx_ranges=None, stat_method='NA', pvalues=None):
            assert fi[0] ==  1, F'{fi[0]} ==  1 failed for stat_test={stat_method} {info1} with pvalues={pvalues}!'
            assert fi[1] == -1, F'{fi[1]} == -1 failed for stat_test={stat_method} {info1} with pvalues={pvalues}!'
            assert fi[2] ==  1, F'{fi[2]} ==  1 failed for stat_test={stat_method} {info1} with pvalues={pvalues}!'
            assert fi[3] ==  0, F'{fi[3]} ==  0 failed for stat_test={stat_method} {info1} with pvalues={pvalues}!'
            if stat_method in ['odds_spearmanr']:
                pass
            else:
                assert fi[4] ==  0, F'{fi[4]} ==  0 failed for stat_test={stat_method} {info1} with pvalues={pvalues}!'
            if approx_ranges:
                for i, (lo, hi) in enumerate(approx_ranges):
                    assert lo <= fi[i] and fi[i] <= hi, F'{lo} <= {fi[i]} <= {hi} failed for {fi} at {i} for stat_test={stat_method} {info1} with pvalues={pvalues}'
        fi1 = ilr.get_feature_importances('f2l2f')
        fi2 = ilr.get_feature_importances('f2l')
        fi3 = ilr.get_feature_importances('f2f')
        fi1min = min((fi1[0], fi1[1], fi1[2])) + 1e-8
        fi1sum = sum((fi1[0], fi1[1], fi1[2]))
        fi1max = max((fi1[3], fi1[4])) - 1e-8
        fi2min = min((fi2[0], fi2[1], fi2[2])) + 1e-8
        fi2sum = sum((fi2[0], fi2[1], fi2[2]))
        fi2max = max((fi2[3], fi2[4])) - 1e-8
        fi3min = min((fi3[0], fi3[1], fi3[2])) + 1e-8
        fi3sum = sum((fi3[0], fi3[1], fi3[2]))
        fi3max = max((fi3[3], fi3[4])) - 1e-8
        assert fi1min > fi1max, F'min(({fi1[0]}, {fi1[1]}, {fi1[2]})) > max({fi1[3]}, {fi1[4]}) failed!'
        assert fi1sum > 5/4.0,  F'sum(({fi1[0]}, {fi1[1]}, {fi1[2]})) > 5/4.0 failed!'
        assert fi2min > fi2max, F'min(({fi2[0]}, {fi2[1]}, {fi2[2]})) > max({fi2[3]}, {fi2[4]}) failed!'
        assert fi2sum > 5/4.0,  F'sum(({fi2[0]}, {fi2[1]}, {fi2[2]})) > 5/4.0 failed!'
        assert fi3min > fi3[3], F'min(({fi3[0]}, {fi3[1]}, {fi3[2]})) > {fi3[3]} failed!'

        if drop_feat:
            assert fi3min > fi3max, F'min(({fi3[0]}, {fi3[1]}, {fi3[2]})) > max({fi3[3]}, {fi3[4]}) failed!'
        assert fi3sum > 0.80,   F'sum(({fi3[0]}, {fi3[1]}, {fi3[2]})) > 0.80 failed!'
        assert fi3sum < 1.20,   F'sum(({fi3[0]}, {fi3[1]}, {fi3[2]})) < 1.20 failed!'

        for stat_method in ['mannwhitneyu', 'spearmanr', 'odds_spearmanr']:
            fi4 = ilr.get_feature_importances('statistic', stat_method)
            fi5 = ilr.get_feature_importances('pvalue', stat_method)
            fi6 = ilr.get_feature_importances('trend', stat_method)
            if (stat_method == 'mannwhitneyu' and task == 'classification') or (stat_method == 'spearmanr' and task == 'regression'):
                assert np.allclose(ilr.get_feature_importances('statistic'), fi4, equal_nan=True), (
                        F'''{ilr.get_feature_importances('statistic')} ~ {fi4} failed!''')
                assert np.allclose(ilr.get_feature_importances('pvalue'), fi5, equal_nan=True), (
                        F'''{ilr.get_feature_importances('pvalue')} ~ {fi5} failed!''')
                assert np.allclose(ilr.get_feature_importances('trend'), fi6, equal_nan=True), (
                        F'''{ilr.get_feature_importances('trend')} ~ {fi6} failed!''')
            if stat_method == 'mannwhitneyu':
                lo, mi, hi = 0.5, 0.6, 1.0 # 9*n_samples, n_samples**1.5, n_samples**2
            elif stat_method == 'spearmanr' and task == 'classification':
                lo, mi, hi = 0.5, 0.6, 1.0
            elif stat_method == 'spearmanr' and task == 'regression':
                lo, mi, hi = 0.2, 0.9, 1.0
            elif stat_method == 'odds_spearmanr' and task == 'classification':
                lo, mi, hi = 0.9, 0.6, 1.0
            elif stat_method == 'odds_spearmanr' and task == 'regression':
                lo, mi, hi = 0.2, 0.9, 1.0
            else: raise ValueError(F'The statistical test {stat_method} is invalid for task {task}!')
            H0pv = (odds_spearmanr_PVALUE_THRES if stat_method in 'odds_spearmanr' else 1e-2)
            statistic_ranges = [(mi, hi), (-hi, -mi), (mi, hi), (-lo, lo), (-lo, lo)]
            check_statistic(fi4, statistic_ranges, stat_method=stat_method)
            if stat_method != 'odds_spearmanr':
                check_pvalue(fi5, [(0, 1e-9), (0, 1e-9), (0, 1e-9), (H0pv, 1), (H0pv, 1)], stat_method=stat_method)
            check_trend(fi6, stat_method=stat_method, pvalues=fi5)
    logging.info(F'Successfully tested (ran without any Error) the combinations of tasks={tasks}, n_samples_s={n_samples_s}, seeds={seeds}')

def test_like_ratio_test():
    hypothesis1_category2count = np.array([[10,20,30], [ 5,10,15]]).transpose()
    hypothesis2_category2count = np.array([[40,50,60], [20,25,30]]).transpose()
    l1 = _likeratio2(hypothesis1_category2count, hypothesis2_category2count)
    print(F'LogLike1={l1}')

if __name__ == '__main__':
    test_like_ratio_test()
    ilr1 = IsotonicLogisticRegression(feat_pvalue_drop=False, task='classification', nontransformed_cols=['col3'])
    ilr2 = IsotonicLogisticRegression(feat_pvalue_drop=False, task='regression', nontransformed_cols=['col3'])
    #test_inverse_transform(ilr1)
    #test_fit_and_predict_proba(ilr1)
    test_fit_and_predict_with_dups(ilr1)
    test_fit_and_predict_with_dups(ilr2)
    test_fit_and_predict_with_convex(ilr2)
    test_with_simulation() #(tasks=['classification', 'regression'])

