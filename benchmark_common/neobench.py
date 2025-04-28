#!/usr/bin/env python

import argparse, collections, copy, datetime, json, itertools, logging, os, pickle, pprint, random, sys
from collections import Counter, defaultdict, namedtuple


scriptpath = (os.path.realpath(__file__))
scriptdir = (os.path.dirname(os.path.realpath(__file__)))
# First parser setup and parse
parser1 = argparse.ArgumentParser()
parser1.add_argument('-n', '--n_hyper_iter', type=int, default=50)
parser1.add_argument('--rand', type=int, default=0)
parser1.add_argument('-I', '--isolib', default=scriptdir+'/../IsotonicLogisticRegression#IsotonicLogisticRegression',
        help='The NeoGuider feature transformation library file')

args1, remaining_argv = parser1.parse_known_args()


import numpy as np
import pandas as pd
from joblib import Parallel, delayed # multiprocessing can hang if the virtual memory allocated is too big
from scipy import stats

import matplotlib
import matplotlib.gridspec as gridspec

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt
matplotlib.use('Agg')  # Use a non-GUI backend
import seaborn as sns

import imblearn
#from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler

# From https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
#  and https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html
import sklearn

from sklearn import metrics
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer

from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis)
from sklearn.ensemble import (AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier,)

from sklearn.exceptions import NotFittedError

from sklearn.feature_selection import VarianceThreshold

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict, cross_val_score, GroupKFold

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

#from sklearn.pipeline import make_pipeline, Pipeline

from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
    # minmax_scale,       # same as MinMaxScaler
    # FunctionTransformer # using user-implemented custom function
)

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.utils import resample
from sklearn.utils.validation import check_is_fitted

from xgboost import XGBClassifier

# https://www.kaggle.com/code/sivasaiyadav8143/10-hyperparameter-optimization-frameworks/notebook#9.-Scikit-Optimize
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer

#logger = logging.getLogger(__name__)
def config_logging(function_name=''): logging.basicConfig(level=logging.INFO, format=F'%(asctime)s %(pathname)s:%(lineno)d %(levelname)s {function_name} - %(message)s')
config_logging('MAIN')

isopath = args1.isolib.split('#')[0]
isolibname = args1.isolib.split('#')[1]
ISO_DIR = os.path.dirname(isopath)
ISO_NAME = os.path.basename(isopath)
ISO_MODULE, ISO_EXT = os.path.splitext(ISO_NAME)
sys.path.append(ISO_DIR)
logging.debug(F'isopath={isopath} isolibname={isolibname} ISO_DIR={ISO_DIR} ISO_NAME={ISO_NAME} ISO_MODULE={ISO_MODULE} ISO_EXT={ISO_EXT}')
IsotonicLogisticRegression = __import__(ISO_MODULE, globals(), locals(), [isolibname], 0)
IsotonicLogisticRegression = IsotonicLogisticRegression.__dict__[isolibname]
NG_default = 'NG'

HYPERPARAM_EPS = 1e-5

# All categorical and numerical hyperparams from https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html
grid_param_AB = {
    'n_estimators' : Integer(100//10, 100*10, prior='log-uniform'),
    'learning_rate' : Real(1/10.0, 1*10.0, prior='log-uniform'),
}

# All categorical and numerical hyperparams from https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
grid_param_RF = {
    'n_estimators'            : Integer(100//10, 100*10, prior='log-uniform'), # 100
    'criterion'               : Categorical(['gini', 'entropy', 'log_loss']), # gini
    #'max_depth_flag'          : Categorical([None, 1]),  # None
    'max_depth'               : Integer(6-5, 6+5),  # None
    'min_samples_split'       : Integer(2, 2+2), # 2
    'min_samples_leaf'        : Integer(1, 1+2), # 1
    'min_weight_fraction_leaf': Real(HYPERPARAM_EPS, 0.5-HYPERPARAM_EPS, prior='log-uniform'), # 0
    'max_features'            : Categorical(['sqrt', 'log2', None]), # None
    'max_leaf_nodes'          : Categorical([2,3,None]), # None
    # 'min_impurity_decrease' # Added in version 0.19.
    'bootstrap'               : Categorical([False, True]), # True
    'class_weight'            : Categorical(['balanced', 'balanced_subsample', None]) # None
}

grid_param_DT = {
    'criterion': Categorical(['gini', 'entropy', 'log_loss']),
    'splitter': Categorical(['best', 'random']), 
    'max_depth': Integer(6-5, 6+5),
    'min_samples_split'       : Integer(2, 2+2), # 2
    'min_samples_leaf'        : Integer(1, 1+2), # 1
    'min_weight_fraction_leaf': Real(HYPERPARAM_EPS, 0.5-HYPERPARAM_EPS, prior='log-uniform'),
    'max_features'            : Categorical(['sqrt', 'log2', None]),
    #random_state=None,
    'max_leaf_nodes'          : Categorical([2,3,None]), # None
    #min_impurity_decrease=0.0, 
    #class_weight=None, 
    #ccp_alpha=0.0,
}

# All categorical and numerical hyperparams from https://xgboost.readthedocs.io/en/latest/parameter.html
grid_param_XGB = {
    'eta' : Real(0.3/10, 0.3*10, prior='log-uniform'),
    'gamma': Integer(0, 2), # 0
    'max_depth': Integer(6-5,6+5), # 6
    'min_child_weight': Real(1/10.0, 1*10.0), # 1
    'max_delta_step': Integer(0,2), # 0
    'subsample': Real(0.5, 1), # 1 where 0.5 is the lower limit
    # Check failed: param.sampling_method == TrainParam::kUniform (1 vs. 0) : Only uniform sampling is supported, gradient-based sampling is only support by GPU Hist.
    # xgboost.core.XGBoostError: Invalid Input: 'subsample', valid values are: {'gradient_based', 'uniform'}
    # 'sampling_method': Categorical(['uniform', 'subsample']), # Categorical(['uniform', 'subsample', 'gradient_based']), # uniform
    'colsample_bytree': Real(0.5, 1), # 1 1 1
    'colsample_bylevel': Real(0.5, 1),
    'colsample_bynode': Real(0.5, 1),
    'lambda': Real(0.1, 10, prior='log-uniform'), # 
    'alpha': Real(0.0, 0.5), # 0
    'tree_method': Categorical(['auto', 'exact', 'approx', 'hist']), # Categorical(['auto', 'exact', 'approx', 'hist']), # auto
    'scale_pos_weight': Real(0.1, 10, prior='log-uniform'), # 1
    # 'updater' is advanced and should not be set in a typical use case
    'refresh_leaf' : Integer(0, 1), # 1 (0 or 1)
    #'process_type' : Categorical(['default', 'update']), # default, because update results in a runtime error
    'grow_policy'  : Categorical(['depthwise', 'lossguide']), # depthwise
    'max_leaves'   : Integer(0, 2), # 0
    'max_bin'      : Integer(256//10, 256*10), # 256
    'num_parallel_tree' : Integer(1, 1+2), # 1
    'objective': Categorical(['binary:logistic', 'binary:logitraw', 'binary:hinge']), # reg:squarederror is changed into all binary objectives
    'eval_metric': Categorical(['logloss', 'auc']), # logloss # auprc is not recognized, resulting in runtime error
}

grid_param_MLP = {
    #'hidden_layer_sizes': Categorical([(10,), (100,), (1000,)]),
    #'hidden_layer_sizes': (Real(1.0, 10.0), pow2intmap),
    'hidden_layer_sizes': Integer(10, 1000, prior='log-uniform'),
    'activation' : Categorical(['identity', 'logistic', 'tanh', 'relu']),
    'solver': Categorical(['lbfgs', 'sgd', 'adam']),
    'alpha' : Real(0.00001, 0.001, prior='log-uniform'),
    'batch_size': Integer(20, 2000, prior='log-uniform'),
    'learning_rate': Categorical(['constant', 'invscaling', 'adaptive']),
    'learning_rate_init' : Real(0.0001, 0.01, prior='log-uniform'),
    'power_t' : Real(0.25, 0.75),
    'max_iter': Integer(20,2000,prior='log-uniform'),
    'shuffle' : Categorical([True, False]),
    'tol' : Real(0.00001, 0.001, prior='log-uniform'),
    #verbose=False, warm_start=False, 
    'momentum' : Real(0.8, 1.0-HYPERPARAM_EPS),
    'nesterovs_momentum' : Categorical([True, False]),
    'early_stopping' : Categorical([True, False]),
    'validation_fraction' : Real(0.05, 0.20, prior='log-uniform'), # If this number is too slow, then we may encounter a runtime error at validation
    'beta_1' : Real(0.8, 1.0-HYPERPARAM_EPS),
    'beta_2' : Real(0.998, 1.0-HYPERPARAM_EPS),
    'epsilon' : Real(1e-09, 1e-7, prior='log-uniform'),
    'n_iter_no_change' : Integer(2,100,prior='log-uniform'),
    'max_fun' : Integer(1500, 150000, prior='log-uniform'),
}

grid_param_KNN = {
    'n_neighbors' : Integer(1, 25, prior='log-uniform'),
    'weights' : Categorical(['uniform', 'distance']), 
    #algorithm='auto', 'ball_tree', 'kd_tree', 'brute'
    #leaf_size=30, 
    'p' : Real(0.2, 20, prior='log-uniform'),
    'metric' : Categorical(['minkowski', 'cosine', 'nan_euclidean']), # ValueError: Haversine distance only valid in 2 dimensions
    #'metric_params': NA
}

# all hyperparams from https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
grid_param_LR = {
    'penalty' : Categorical(['l1', 'l2', 'elasticnet', None]),
    #dual=False # only implementd for liblinear solver
    'tol' : Real(1e-5, 1e-3, prior='log-uniform'),
    'C'   : Real(0.1, 10, prior='log-uniform'),
    'fit_intercept' : Categorical([True, False]),
    'intercept_scaling'   : Real(0.1, 10, prior='log-uniform'),
    'class_weight' : Categorical(['balanced', None]),
    'solver'       : Categorical(['saga']), # only this one works for elasticnet
    'max_iter'     : Integer(10, 1000, prior='log-uniform'),
    'l1_ratio'     : Real(0, 1),
}

# all scikit-learn feature preprocessors from https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html
HPARAM_DEFLT_FT_PREPROC_NAME2TECH = {
    'IdentityTransformer' : ColumnTransformer([], remainder='passthrough'),
    'MaxAbsScaler'        : MaxAbsScaler(),
    'MinMaxScaler'        : MinMaxScaler(),
    'Normalizer'          : Normalizer(),
    'PowerTransformer'    : PowerTransformer(),
    'QuantileTransformer' : QuantileTransformer(random_state=args1.rand),
    'RobustScaler'        : RobustScaler(),
    'StandardScaler'      : StandardScaler(),
    # 'NormalTransformer' : QuantileTransformer(random_state=args1.rand, output_distribution='normal'), # not used with default value
    F'{NG_default}'       : IsotonicLogisticRegression(random_state=args1.rand, excluded_cols=['ln_NumTested']),
    'NG_withoutNumTested' : IsotonicLogisticRegression(random_state=args1.rand, excluded_cols=[]),
}

# Let StandardScaler represent IdentityTransformer, MaxAbsScaler, MinMaxScaler, and RobustScaler since they are all linear maps
# Then, all scikit-learn feature preprocessors, 
#   mentioned at https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html, 
# can be represented by the following items:
HPARAM_TUNED_FT_PREPROC_NAME2TECH = {
    
    # The identity map results in runtime error for hyperparam optimimization for MLP (and possibly for LogReg too)
    # 'IdentityTransformer' : ColumnTransformer([], remainder='passthrough'), 
    'Normalizer'          : copy.deepcopy(HPARAM_DEFLT_FT_PREPROC_NAME2TECH['Normalizer']),
    'PowerTransformer'    : copy.deepcopy(HPARAM_DEFLT_FT_PREPROC_NAME2TECH['PowerTransformer']),
    'QuantileTransformer' : copy.deepcopy(HPARAM_DEFLT_FT_PREPROC_NAME2TECH['QuantileTransformer']),
    'StandardScaler'      : copy.deepcopy(HPARAM_DEFLT_FT_PREPROC_NAME2TECH['StandardScaler']),
    F'{NG_default}'       : copy.deepcopy(HPARAM_DEFLT_FT_PREPROC_NAME2TECH[F'{NG_default}']),
    'NG_withoutNumTested' : copy.deepcopy(HPARAM_DEFLT_FT_PREPROC_NAME2TECH['NG_withoutNumTested']),
}

# All classifiers from https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
# The classifiers with runtime error are commented out (or provided with ideas to work around the error)
HPARAM_DEFLT_CLASSIFIER_NAME2TECH = {
    
    # KNN error: too much computational time O(n_examples^2), intractable # workaround: downsample negative examples
    'hParamDefault_KNN': KNeighborsClassifier(), 
    
    # from sklearn.svm import SVC : is not designed to predict probability and is not designed to handle large sample size
    # SVC error: too much computational time O(n_examples^2) # workaround: downsample negative examples
    'hParamDefault_SVC': SVC(probability=True),
    
    # GP error: O(n_examples^3*n_iterations) RAM requirement, with numpy.core._exceptions._ArrayMemoryError: Unable to allocate 1.25 TiB for ... 
    # workaround: none AFAIK
    # 'hParamDefault_GP':  GaussianProcessClassifier(random_state=args1.rand),

    'hParamDefault_DT':  DecisionTreeClassifier(random_state=args1.rand),
    'hParamDefault_RF':  RandomForestClassifier(random_state=args1.rand),
    'hParamDefault_MLP': MLPClassifier(random_state=args1.rand),
    'hParamDefault_AB':  AdaBoostClassifier(random_state=args1.rand),
    'hParamDefault_GNB': GaussianNB(),
        
    'hParamDefault_QDA': QuadraticDiscriminantAnalysis(),
    
    'hParamDefault_LDA': LinearDiscriminantAnalysis(), # Not listed in plot_classifier_comparison.html
    'hParamDefault_LR' : LogisticRegression(random_state=args1.rand), # Not listed in plot_classifier_comparison.html
    'hParamDefault_XGB': XGBClassifier(random_state=args1.rand), # Not listed in plot_classifier_comparison.html and benchmarked by github.com/XuegongLab/NeoRanking

    # 'ET': ExtraTreesClassifier(random_state=args1.rand)      , # Not listed in plot_classifier_comparison.html and performs worse than RF
    # 'GB': GradientBoostingClassifier(random_state=args1.rand), # Not listed in plot_classifier_comparison.html and similar to XGB but runs much slower
}

# Other authors also performed split in patient-unspecific manner (the same patient's peptides are used for both training and assessing hyperparam-goodness)
# for hypeparam tuning, for example: https://github.com/XuegongLab/NeoRanking/blob/5df3b6c2/Classifier/OptimizationParams.py#L356,
# We think it is fine because our hyperparam-goodness score is the roc-auc for the ensemble of peptides over multiple patients
HPARAM_TUNING_PARAMS = {
    'scoring': 'roc_auc',
    'random_state': args1.rand, 
    'n_iter': args1.n_hyper_iter, 
    'cv': 3,
    'n_jobs': 12,
    'verbose': 9,
}

# We performed hyperparam tuning to the classifiers that were not used with their default hyperparameter values at
# https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
HPARAM_TUNED_CLASSIFIER_NAME2TECH = {
    
    # GP error: same as the error with hParamDefault_GP
    # 'hParamTuned_GP'  BayesSearchCV(GaussianProcessClassifier(random_state=args1.rand), grid_param_GP , **HPARAM_TUNING_PARAMS),
     
    # AB was used with its default hyperparameter values
    'hParamTuned_AB' : BayesSearchCV(AdaBoostClassifier    (random_state=args1.rand), grid_param_AB , **HPARAM_TUNING_PARAMS),
    
    'hParamTuned_DT' : BayesSearchCV(DecisionTreeClassifier(random_state=args1.rand), grid_param_DT , **HPARAM_TUNING_PARAMS),
    
    # KNN consumes too much running time
    'hParamTuned_KNN': BayesSearchCV(KNeighborsClassifier  (                       ), grid_param_KNN, **HPARAM_TUNING_PARAMS),
    'hParamTuned_MLP': BayesSearchCV(MLPClassifier         (random_state=args1.rand), grid_param_MLP, error_score=0, **HPARAM_TUNING_PARAMS),
    'hParamTuned_RF' : BayesSearchCV(RandomForestClassifier(random_state=args1.rand), grid_param_RF , **HPARAM_TUNING_PARAMS),
    
    # SVC consumes too much running time
    'hParamTuned_SVC': BayesSearchCV(DecisionTreeClassifier(random_state=args1.rand), grid_param_DT , **HPARAM_TUNING_PARAMS),
    
    # LR and XGB are not listed in plot_classifier_comparison.html
    'hParamTuned_LR' : BayesSearchCV(LogisticRegression    (random_state=args1.rand), grid_param_LR,  **HPARAM_TUNING_PARAMS),
    'hParamTuned_XGB': BayesSearchCV(XGBClassifier         (random_state=args1.rand), grid_param_XGB, **HPARAM_TUNING_PARAMS),
}

# These are the classifiers at https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
# that did not run into any errors (including excessive runtime, i.e., O(n^2) runtime where n is the size of training set)
# plus LogisticRegression (LR)
HPARAM_DEFLT_CLASSIFIER_LIST = ['hParamDefault_AB', 'hParamDefault_DT', 'hParamDefault_GNB', 'hParamDefault_MLP', 'hParamDefault_QDA', 'hParamDefault_RF',  'hParamDefault_LR']

# These are the classifiers from HPARAM_DEFLT_CLASSIFIER_LIST that were not used with their default hyperparameter values at
# https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
HPARAM_TUNED_CLASSIFIER_LIST = ['hParamTuned_DT', 'hParamTuned_MLP', 'hParamTuned_RF', 'hParamTuned_LR']

FINAL_FT_PREPROC_NAMES = list(HPARAM_DEFLT_FT_PREPROC_NAME2TECH.keys())
FINAL_CLASSIFIER_NAMES = HPARAM_TUNED_CLASSIFIER_LIST + HPARAM_DEFLT_CLASSIFIER_LIST

CLASSIFIERS_REQUIRING_STRONG_BALANCE = set([
    'hParamDefault_KNN', 'hParamTuned_KNN',
    'hParamDefault_GP' , 'hParamTuned_GP' ,
    'hParamDefault_SVC', 'hParamTuned_SVC',
])

CLASSIFIERS_REQUIRING_BALANCE = set(HPARAM_TUNED_CLASSIFIER_NAME2TECH.values())

SOFT_NAME_TO_MANUSCRIPT_NAME = {
    'Score_EL': 'NetMHCpan_ScoreEL',
    'MT_BindAff': 'NetMHCpan_ICfiftyBA',
    'BindStab': 'NetMHCstabpan_BindStab',
    'Quantification': 'KallistoTPM_NeoAbundance',
    'Agretopicity' : 'ICfiftyBA_Agretop',
    'ln_NumTested' : 'NumTested',
}

SOFT_NAME_TO_MANUSCRIPT_NAME_ALWAYS = {
    'ln_NumTested' : 'NumTested',
    'mhcflurry_aff_percentile'         : 'MHCflurry_aff_%',
    'mhcflurry_presentation_percentile': 'MHCflurry_presentation_%',
    #'DeepHLApan_immunogenic_score'     : 'DeepHLApan_immuno_score'
    '%Rank_EL' : 'NetMHCpan_%RankEL',
    '%Rank_BA' : 'NetMHCpan_%RankBA',
    'Score_EL' : 'NetMHCpan_ScoreEL',
    'Score_BA' : 'NetMHCpan_ScoreBA',
}

def add_redundant_names(elements, idx=[0,1,2,3]):
    if isinstance(elements, str):
        assert isinstance(idx, int)
        return elements + '_' + str(idx)
    ret = list(elements)
    for e in elements:
        for i in idx:
            ret.append(e+'_' + str(i))
    return ret

def prep_input(self, X):
    arr = copy.deepcopy(X)
    col_means = np.nanmean(arr, axis=0)
    nan_indices = np.isnan(arr)
    arr[nan_indices] = np.take(col_means, np.where(nan_indices)[1])
    return arr

def make_my_arr(x, colidx, ncols):
    return np.array([([e] * ncols) for e in x]).transpose()

def ax_trans(ft, colidx):
    print(F'transform(X..., column_idx={colidx})')
    return lambda x2:ft.transform(x2, column_idx=colidx)
def ax_inverse_trans(ft, colidx):
    print(F'inverse_transform(X..., column_idx={colidx})')
    return lambda x2:ft.inverse_transform(x2, column_idx=colidx)

def pairplot_showing_pretrans_feat_vals(df1, df2, feature_transformer):
    dfall = pd.concat([df1, df2], axis=0)
    dfall = dfall.apply(pd.to_numeric)
    #print(F'DEBUG:::dfall=\n{dfall}\n')
    allmax = dfall.max().max() #max([max(row) for row in dfall])
    allmin = dfall.min().min() #min([min(row) for row in dfall])
    #assert type(allmax) == float, F'The DataFrame {dfall} is not all numeric because allmax={allmax}!'
    intmax = int(round(allmax))
    intmin = int(round(allmin))
    #print(dfall)
    n_vars = len(dfall.columns)
    transformed_ticks = list(range(intmin, intmax+1))
    transformed_tick_2d = np.array([transformed_ticks for _ in dfall.columns]).transpose()
    if feature_transformer: tick_2d = feature_transformer.inverse_transform(transformed_tick_2d)

    figsize = n_vars * 4.25
    fig = plt.figure(figsize=(figsize, figsize))
    gs = gridspec.GridSpec(n_vars, n_vars)
    for i in range(n_vars):
        for j in range(n_vars):
            ax = fig.add_subplot(gs[i, j])
            ax.set_xlabel(dfall.columns[j])
            ax.set_ylabel(dfall.columns[i])
            ax.set_xlim(intmin-0.25, intmax+0.25)

            if i == j:
                ax.hist([df1.iloc[:, i], df2.iloc[:, i]], bins=10, alpha=0.7, label=['Negative', 'Positive'])
                ax.legend()
            else:
                ax.scatter(df1.iloc[:, j], df1.iloc[:, i], alpha=0.5)
                ax.scatter(df2.iloc[:, j], df2.iloc[:, i], alpha=0.5)

                # Custom axis settings
                ax.set_ylim(intmin-0.25, intmax+0.25)
                ax.set_xticks(transformed_ticks)
                ax.set_yticks(transformed_ticks)
                if feature_transformer:
                    ax2x = ax.twiny()
                    ax2x.set_xlim(intmin-0.25, intmax+0.25)
                    ax2x.set_xticks(transformed_ticks)
                    ax2x.set_xticklabels([F'{v:.2g}' for v in tick_2d[:,j]], ha='left')
                    ax2x.set_xlabel('Raw feature values', color='green')
                    ax2x.tick_params(labelrotation=45, colors='green')
                    ax2y = ax.twinx()
                    ax2y.set_ylim(intmin-0.25, intmax+0.25)
                    ax2y.set_yticks(transformed_ticks)
                    ax2y.set_yticklabels([F'{v:.2g}' for v in tick_2d[:,i]], va='bottom')
                    ax2y.set_ylabel('Raw feature values', color='green')
                    ax2y.tick_params(labelrotation=90-45, colors='green')
            # Hide redundant labels for cleaner output
            #if i != n_vars - 1:
            #    ax.set_xticklabels([])
            #if j != 0:
            #    ax.set_yticklabels([])
    plt.tight_layout()
    #plt.show()
    return fig

# Section on pre-defined features

# from https://github.com/SchubertLab/benchmark_TCRprediction
PMHC_TCR_PRED_60_MODELS = 'predictions_atm-tcr,predictions_attntap_MCPAS,predictions_attntap_VDJDB,predictions_bertrand,predictions_dlptcr_ALPHA,predictions_dlptcr_BETA,predictions_epitcr_WITH_MHC,predictions_epitcr_WO_MHC,predictions_ergo-i_AE_MCPAS,predictions_ergo-i_AE_VDJDB,predictions_ergo-i_LSTM_MCPAS,predictions_ergo-i_LSTM_VDJDB,predictions_ergo-ii_MCPAS,predictions_ergo-ii_VDJDB,predictions_imrex_DOWNSAMPLED,predictions_imrex_FULL,predictions_itcep,predictions_nettcr_t.0.v.1,predictions_nettcr_t.0.v.2,predictions_nettcr_t.0.v.3,predictions_nettcr_t.0.v.4,predictions_nettcr_t.1.v.0,predictions_nettcr_t.1.v.2,predictions_nettcr_t.1.v.3,predictions_nettcr_t.1.v.4,predictions_nettcr_t.2.v.0,predictions_nettcr_t.2.v.1,predictions_nettcr_t.2.v.3,predictions_nettcr_t.2.v.4,predictions_nettcr_t.3.v.0,predictions_nettcr_t.3.v.1,predictions_nettcr_t.3.v.2,predictions_nettcr_t.3.v.4,predictions_nettcr_t.4.v.0,predictions_nettcr_t.4.v.1,predictions_nettcr_t.4.v.2,predictions_nettcr_t.4.v.3,predictions_panpep,predictions_pmtnet,predictions_stapler,predictions_tcellmatch_GRU_CV0,predictions_tcellmatch_GRU_CV1,predictions_tcellmatch_GRU_CV2,predictions_tcellmatch_GRU_SEP_CV0,predictions_tcellmatch_GRU_SEP_CV1,predictions_tcellmatch_GRU_SEP_CV2,predictions_tcellmatch_LINEAR_CV0,predictions_tcellmatch_LINEAR_CV1,predictions_tcellmatch_LINEAR_CV2,predictions_tcellmatch_LSTM_CV0,predictions_tcellmatch_LSTM_CV1,predictions_tcellmatch_LSTM_CV2,predictions_tcellmatch_LSTM_SEP_CV0,predictions_tcellmatch_LSTM_SEP_CV1,predictions_tcellmatch_LSTM_SEP_CV2,predictions_teim,predictions_teinet_LARGE_DS,predictions_teinet_SMALL_DS,predictions_titan,predictions_tulip-tcr'.split(',')

PMHC_TCR_PRED_TOOLS = 'atm-tcr,attntap_VDJDB,bertrand,dlptcr_BETA,epitcr_WITH_MHC,ergo-i_AE_VDJDB,ergo-ii_VDJDB,imrex_FULL,itcep,nettcr_t.1.v.0,panpep,pmtnet,stapler,tcellmatch_LINEAR_CV1,teim,teinet_SMALL_DS,titan,tulip-tcr'.split(',')

#cohort Mut_peptide HLA_allele Patient Partition
IMPROVE_FTS = 'Aro mw pI Inst CysRed RankEL RankBA NetMHCExp Expression SelfSim Prime PropHydroAro HydroCore PropSmall PropAro PropBasic PropAcidic DAI Stability Foreigness CelPrev PrioScore CYT HLAexp MCPmean'.split()
# response prediction_rf

# The following were already quantile-normalized and therefore not used: PRIME_rank,PRIME_BArank,mhcflurry_aff_percentile,mhcflurry_presentation_percentile
FEATS = 'Quantification,Agretopicity,ln_NumTested'.split(',') # By default, the 'default' of the --add cmd-line option will be added

MULLER_NEOPEP_FTS = 'CCF Clonality rnaseq_TPM rnaseq_alt_support CSCAPE_score mutant_other_significant_alleles mutant_rank mutant_rank_PRIME mutant_rank_netMHCpan Sample_Tissue_expression_GTEx GTEx_all_tissues_expression_mean TCGA_Cancer_expression gene_driver_Intogen nb_same_mutation_Intogen mutation_driver_statement_Intogen bestWTMatchScore_I bestWTMatchOverlap_I bestMutationScore_I bestWTMatchType_I  bestWTPeptideCount_I mut_Rank_Stab mut_netchop_score_ct TAP_score mut_is_binding_pos mut_binding_score mut_aa_coeff seq_len DAI_NetMHC DAI_MixMHC DAI_NetStab DAI_MixMHC_mbp'.strip().split()

MULLER_NEOMUT_FTS = 'CCF Clonality Zygosity Sample_Tissue_expression_GTEx TCGA_Cancer_expression rnaseq_TPM rnaseq_alt_support MIN_MUT_RANK_CI_MIXMHC COUNT_MUT_RANK_CI_MIXMHC WT_BEST_RANK_CI_MIXMHC MIN_MUT_RANK_CI_PRIME COUNT_MUT_RANK_CI_PRIME WT_BEST_RANK_CI_PRIME COUNT_MUT_RANK_CI_netMHCpan CSCAPE_score gene_driver_Intogen nb_mutations_in_gene_Intogen nb_same_mutation_Intogen mutation_driver_statement_Intogen GTEx_all_tissues_expression_mean bestWTMatchScore_I bestWTMatchOverlap_I bestMutationScore_I bestWTPeptideCount_I mut_Rank_EL_0 wt_Rank_EL_0 mut_Rank_EL_1 wt_Rank_EL_1 mut_Rank_EL_2 wt_Rank_EL_2 mut_Rank_Stab_0 mut_Rank_Stab_1 mut_Rank_Stab_2 mut_netchop_score mut_TAP_score_0 next_best_BA_mut_ranks DAI_0 DAI_1 DAI_2'.strip().split()

LISTOF_FEATURES = [FEATS, PMHC_TCR_PRED_60_MODELS, PMHC_TCR_PRED_TOOLS, IMPROVE_FTS, MULLER_NEOPEP_FTS, MULLER_NEOMUT_FTS]

LISTOF_LABELS = [['Label', 'response', 'VALIDATED', 'response_type']]

ASCENDING_FEATURES = ('MT_BindAff,Agretopicity,%Rank_EL,%Rank_BA,PRIME_rank,PRIME_BArank,mhcflurry_aff_percentile,mhcflurry_presentation_percentile,ln_NumTested'.split(',')
    + 'DAI NetMHCExp pI PropBasic Inst PropAcidic RankEL PropSmall ln_NumTested RankBA'.split())

DEBUG_SKLEARN_PIPE = 'sklearn-pipe'

# Section on argparse

parser = argparse.ArgumentParser(description='This script analyzes features (the features are typically the output of relevant software packages, such as kallisto, netMHCpan, mhcflurry, PRIME, ERGO, and netTCR). ', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-i', '--input', nargs='+', default=[ scriptdir+'/media-2.csv', scriptdir+'/media-4.csv' ],
        help='String list of length 2, 4, 6, etc. denoting task1 file1 task2 file2 task3 file3 etc. \n'
        'Task can be tr (train), te (test, aka benchmark or score), cv (cross validate), or comma-separated string of any combination of tr, te, and cv (e.g., tr,te and tr,cv)\n'
        'File can be arbitray CSV or TSV file with the feature names given by the --features param. If this param is not provided, then one of the following types of file can be used: '
        '1 - the file used to train neoguider, '
        '2 - the validated-neoantigen file from https://github.com/SRHgroup/IMPROVE_paper/blob/main/data.zip, '
        '3 - the predictions on the viral and mutation datasets downloaded from https://www.biorxiv.org/content/10.1101/2024.11.06.622261v1.supplementary-material, '
        '4 - the predictions* results generated by https://github.com/SchubertLab/benchmark_TCRprediction. ')
parser.add_argument('-o', '--output', required=True, #(scriptdir+'/tmp/default_out'),
        help='The prefix of the output files')
parser.add_argument('-m', '--model',  default=(None),
        help='The prefix of the directory containing model files in pickle format')

parser.add_argument('-1', '--hparam_deflt_ft_preproc_names', nargs='*', default=[x for x in HPARAM_DEFLT_FT_PREPROC_NAME2TECH],
        help='Names of the feature preprocessing techniques to be assessed')
parser.add_argument('-2', '--hparam_deflt_classifier_names', nargs='*', default=[x for x in HPARAM_DEFLT_CLASSIFIER_NAME2TECH],
        help='Names of the machine-learning classifiers to be assessed')
parser.add_argument('-3', '--hparam_tuned_ft_preproc_names', nargs='*', default=[x for x in HPARAM_TUNED_FT_PREPROC_NAME2TECH],
        help='Names of the feature preprocessing techniques to be assessed with hyperparameter tuning')
parser.add_argument('-4', '--hparam_tuned_classifier_names', nargs='*', default=[x for x in HPARAM_TUNED_CLASSIFIER_NAME2TECH],
        help='Names of the machine-learning classifiers with hyperparameter tuning to be assessed')
parser.add_argument('--inc', default=None, help='Assume that label as a function of each feature is increasing (0, 1, "auto", or None denoting false, true, auto, and inferred)')
parser.add_argument('--sep', default=None, help='csv column separator')
parser.add_argument('--seed', default=43, help='seed for random number generation')
parser.add_argument('--tasks', nargs='*', default=['fa1', 'fa2', 'fa3', 'hla1', 'hla2'], help='Feature-analysis and HLA-analysis tasks')
parser.add_argument('--features', nargs='*', default=[], help='Features analyzed, auto infer if not provided')
parser.add_argument('--label', default='', help='The label analyzed, auto infer if not provided')
parser.add_argument('--debug', nargs='*', default=[], help=F'Debug tokens. {DEBUG_SKLEARN_PIPE}: test sklearn pipeline. ')

# Maintain consistency with Muller et al. 2023, Immunity
parser.add_argument('-uf', '--untest_flag', default=0x1, type=int, help='If the 0x1, 0x2, and 0x4 bits are set, then remove the rows with NA label (not tested for immunogenicity by any immuno-assay validation) for training, test, and cross-validation. ')
parser.add_argument('-pf', '--peplen_flag', default=0x0, type=int, help='If the 0x1, 0x2, and 0x4 bits are set, then remove peptides with lengths greater than 11 (with at least 12 amino acid residues) for training, test, and cross-validation. ')
parser.add_argument('--add', nargs='*', default=['default'], help='pMHC-binding features, like default, netmhc, mhcflurry, and/or prime, to be added to the list of features. The default uses netMHCpan ScoreEL, netMHC binding affinity, and netMHCstabpan binding stability. ')
para_n_jobs = 16

args = parser.parse_args(remaining_argv)

if args.output and not args.model: model_dir_prefix = args.output
elif args.model: model_dir_prefix = args.model
else: raise RuntimeError(F'Cannot infer the directory that contains (or will contain) the trained models from the parsed args: {args}!')
modeldir = f'{model_dir_prefix}.dir'

if 'mhcflurry' in args.add: LISTOF_FEATURES[0].extend(['mhcflurry_aff_percentile', 'mhcflurry_presentation_percentile'])
if 'prime' in args.add: LISTOF_FEATURES[0].extend(['PRIME_BArank', 'PRIME_rank', 'PRIME_score'])
if 'netmhc' in args.add: LISTOF_FEATURES[0].extend(['Score_EL', '%Rank_EL', 'Score_BA', '%Rank_BA', 'MT_BindAff', 'BindStab'])
elif 'default' in args.add: LISTOF_FEATURES[0].extend(['Score_EL', 'MT_BindAff', 'BindStab'])
LISTOF_FEATURES[0] = sorted(set(LISTOF_FEATURES[0]))

random.seed(args.seed)
np.random.seed(args.seed)

assert len(args.input) % 2 == 0, F'''The number of input files ({args.input}) is not (but should be) a multiple of two'''
testfile = args.input[1]
with open(testfile) as file:
    firstline = file.readline()
    if not args.sep:
        if   firstline.count('\t') > 3: csvsep = '\t'
        elif firstline.count(',')  > 3: csvsep = ','
        elif firstline.count(' ')  > 3: csvsep = ' '
        else: raise RuntimeError(F'Cannot infer the column separator string from the first line of the file {testfile}!')    
    else:
        csvsep = args.sep
column_names = pd.read_csv(testfile, index_col=0, nrows=0, sep=csvsep).columns.tolist()

if args.inc == None:
    if sum([(x in column_names) for x in (PMHC_TCR_PRED_TOOLS + PMHC_TCR_PRED_60_MODELS)]) >= (1 + len(PMHC_TCR_PRED_TOOLS)) // 2:
        increasing = True
    else:
        increasing = 'auto'
else:
    increasing = args.inc
if increasing in [True, False]: feat_pvalue_drop = False
else: feat_pvalue_drop = True

logging.info(F'feat_pvalue_drop={feat_pvalue_drop}')

nan_policy='raise' #'mean'
kwargs = {
    'increasing': increasing,
    'random_state': 0, 
    'feat_pvalue_drop': feat_pvalue_drop, 
    'nan_policy': nan_policy,
    'excluded_cols': ['ln_NumTested']}

#for tech in [HPARAM_TUNED_FT_PREPROC_NAME2TECH, HPARAM_DEFLT_FT_PREPROC_NAME2TECH]:
#    tech[F'{NG_default}']       = IsotonicLogisticRegression(**kwargs)
#    tech['NG_withoutNumTested'] = IsotonicLogisticRegression(increasing=increasing, random_state=args1.rand, feat_pvalue_drop=feat_pvalue_drop, nan_policy=nan_policy)

try:
    sklearn.set_config(enable_metadata_routing=False)
except TypeError as err:
    pass

def comb(name1, name2, sep='/'): return name1 + sep + name2
def decomb(name, sep='/'): return name.split(sep)

def compute_hla_mat(df, hlacol, labelcol, patientcol):
    epitope_counts = df.groupby([patientcol, hlacol])[labelcol].sum().reset_index()
    epitope_counts.rename(columns={labelcol: 'Number_of_tested_positives', hlacol: 'HLA_allele'}, inplace=True)
    print(epitope_counts)
    matrix = epitope_counts.pivot(index=patientcol, columns='HLA_allele', values='Number_of_tested_positives')
    matrix = matrix.fillna(-1)
    return matrix

def analyze_hla(df, hlacol, labelcol, figout, patientcol='Patient'):
    matrix = compute_hla_mat(df, hlacol, labelcol, patientcol)
    g = sns.clustermap(
        matrix,
        figsize=(matrix.shape[1]*0.4+1, matrix.shape[0]*0.25+1),
        annot=True,
        mask=(matrix==-1),
        linewidths=0.5,
        linecolor='blue',
        #dendrogram_ratio=0.22,
    )
    g.ax_heatmap.set_xticklabels(
        g.ax_heatmap.get_xticklabels(),
        rotation=30,  # Angle in degrees
        ha='right',   # Horizontal alignment
        rotation_mode='anchor',        
    )
    #g.ax_heatmap.set_xticks(np.arange(matrix.shape[1]+1)-0.5, minor=True)
    #g.ax_heatmap.set_yticks(np.arange(matrix.shape[0]+1)-0.5, minor=True)
    #g.ax_heatmap.grid(which='major', color='blue', linestyle='dotted', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(figout)
    plt.close()
    return matrix

def compute_ranked_df(df, labelcol, patientcol='Patient', predcol=F'{NG_default}/hParamDefault_LR', ranking_mult=1):
    df = df.sort_values(predcol, ascending=(ranking_mult==-1))
    ranks = []
    patient2rank = collections.defaultdict(int)
    for patient in df[patientcol]:
        patient2rank[patient] += 1
        ranks.append(patient2rank[patient])
    df['rank'] = ranks
    return df

def analyze_performance_per_hla(df, hlacol, labelcol, figout, patientcol='Patient', predcol=F'{NG_default}/hParamDefault_LR'):
    matrix = compute_hla_mat(df, hlacol, labelcol, patientcol)
    df = compute_ranked_df(df, labelcol)
    top20df = df.loc[df['rank']<=20]
    matrix2 = matrix.copy()
    for patient in matrix.index:
        for hla in matrix.columns:
            over = len(top20df.loc[(top20df[patientcol] == patient) & (top20df[hlacol] == hla) & (top20df[labelcol] == 1)])
            under = float(matrix.loc[patient,hla])
            matrix2.loc[patient,hla] = (over / under if under > 0 else -1)
    matrix2 = matrix2.loc[:, matrix2.max(axis=0)>-1]
    g = sns.clustermap(
        matrix2,
        figsize=(matrix2.shape[1]*0.4+1, matrix2.shape[0]*0.25+1),
        annot=True,
        mask=(matrix2==-1),
        linewidths=0.5,
        linecolor='blue',
        #dendrogram_ratio=0.22,
    )
    g.ax_heatmap.set_xticklabels(
        g.ax_heatmap.get_xticklabels(),
        rotation=30,  # Angle in degrees
        ha='right',   # Horizontal alignment
        rotation_mode='anchor'
    )
    #g.ax_heatmap.set_xticks(np.arange(matrix2.shape[1]+1)-0.5, minor=True)
    #g.ax_heatmap.set_yticks(np.arange(matrix2.shape[0]+1)-0.5, minor=True)
    #g.ax_heatmap.grid(which='major', color='blue', linestyle='dotted', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(figout)
    logging.info(F'In analyze_performance_per_hla: saved clustermap to {figout}')
    plt.close()
    return matrix2

def between(x, lower, upper): return min((max((lower, x)), upper))

def make_imbalearn_selector(classifier_name, n_positives, n_negatives):
    if classifier_name in CLASSIFIERS_REQUIRING_STRONG_BALANCE:
        new_n_pos = min((n_positives, 1e3))
        new_n_neg = between(n_negatives, n_positives, 1e3) # 1e3 is to avoid out-of-mem error
    elif classifier_name in CLASSIFIERS_REQUIRING_BALANCE:
        # In order to limit computation time during Hyperopt training on neo-peptides,
        # the size of NCI-train_neo-pep was limited by randomly sampling 100,000 non-immunogenic neo-peptides from NCI-train_neo-pep,
        # while all immunogenic neo-peptides in NCI-train_neo-pep were retained
        # 1e5 is from https://www.cell.com/immunity/fulltext/S1074-7613(23)00406-5#sectitle0030
        new_n_pos = min((n_positives, 1e5))
        new_n_neg = between(n_negatives, n_positives, 1e5)
    else:
        logging.info(F'Classifier {classifier_name} was ignored by random sampling with n_positives={n_positives} and n_negatives={n_negatives}!')
        return 0, 'passthrough' # IdentityTransformer VarianceThreshold() # RandomUnderSampler(sampling_strategy=0.0001, random_state=args1.rand)
    if new_n_neg > new_n_pos:
        logging.info(F'Classifier {classifier_name} went through random sampling with n_positives={n_positives} and n_negatives={n_negatives}!')
        return 1, RandomUnderSampler(sampling_strategy=(new_n_pos/new_n_neg), random_state=args1.rand)
    else:
        logging.info(F'Classifier {classifier_name} was not through random sampling with n_positives={n_positives} and n_negatives={n_negatives}!')
        return 0, 'passthrough' # VarianceThreshold()

def construct_ml_pipes(ft_preproc_tech_dict, classifier_dict, hparam_tuned_ft_preproc_tech_dict, hparam_tuned_classifier_dict, y):
    ret = []
    n_positives, n_negatives = len([v for v in y if v == 1]), len([v for v in y if v == 0])
    assert n_positives + n_negatives == len(y), F'The vector y containing elements {set(y)} is not binary!'
    
    for     ( hyperparam_tuning_strategy,     FT_PREPROC_NAME2TECH,              CLASSIFIER_NAME2TECH, ft_preproc_names,                                     classifier_names) in [
            ('hyperparam_tuned', HPARAM_TUNED_FT_PREPROC_NAME2TECH, HPARAM_TUNED_CLASSIFIER_NAME2TECH, args.hparam_tuned_ft_preproc_names, args.hparam_tuned_classifier_names), 
            ('hyperparam_deflt', HPARAM_DEFLT_FT_PREPROC_NAME2TECH, HPARAM_DEFLT_CLASSIFIER_NAME2TECH, args.hparam_deflt_ft_preproc_names, args.hparam_deflt_classifier_names)]:
        for     ft_preproc_name, ft_preproc_tech in sorted(FT_PREPROC_NAME2TECH.items()):
            for classifier_name, classifier_tech in sorted(CLASSIFIER_NAME2TECH.items()):
                if (ft_preproc_name, classifier_name) in [('IdentityTransformer', 'hParamTuned_MLP')]: continue # data
                if not (ft_preproc_name in ft_preproc_names
                    and classifier_name in classifier_names): 
                    continue
                ml_pipename = comb(ft_preproc_name, classifier_name)
                was_balancing_performed, imbalearn_selector = make_imbalearn_selector(classifier_name, n_positives, n_negatives)
                ml_pipe = imblearn.pipeline.make_pipeline(imbalearn_selector, copy.deepcopy(ft_preproc_tech), VarianceThreshold(), copy.deepcopy(classifier_tech))
                ret.append((ml_pipename, ml_pipe))
                if DEBUG_SKLEARN_PIPE in args.debug and not was_balancing_performed and hyperparam_tuning_strategy == 'hyperparam_deflt':
                    for first_step_idx, first_step_tech in enumerate(['passthrough', VarianceThreshold()]):
                        ml_pipe = sklearn.pipeline.make_pipeline(first_step_tech, copy.deepcopy(ft_preproc_tech), VarianceThreshold(), copy.deepcopy(classifier_tech))
                        ret.append((add_redundant_names(ml_pipename, first_step_idx + 1), ml_pipe))
    return ret

def assert_prob_arr(prob_pred, ml_pipename):
    if ml_pipename.endswith('hParamTuned_XGB'): return 1
    assert prob_pred.shape[1] == 2, F'The predicted result {prob_pred} does not have two columns denoting two columns for {ml_pipename}!'
    for x, y in prob_pred:
        assert 0 <= x and x <= 1, F'The probability {x} must be between zero and one for {ml_pipename}!'
        assert 0 <= y and y <= 1, F'The probability {y} must be between zero and one for {ml_pipename}!'
        assert 1-1e-9 < (x + y) and (x + y) < 1+1e-9, F'The probabilities {x} and {y} do not sum to one for {ml_pipename}!'

def drop_feat_from_X(ml_pipename, X):
    X = X.copy()
    #for colname in X.columns:
    #    if colname in ASCENDING_FEATURES: 
    #        X.loc[:,colname] = -X[colname]
    #        logging.info(F'Performed negation to the column {colname} (CHECK_FOR_BUG)')
    if (not 'ln_NumTested' in X.columns):
        return X.copy()
    elif ('withoutNumTested'.lower() in ml_pipename.lower()) or (not 'neoguider' in ml_pipename.lower() and not ml_pipename.startswith(NG_default)): # and not 'NG_' in ml_pipename:
        return X.drop(columns=['ln_NumTested'])
    else:
        return X.copy()

def train_ml_pipe(ml_pipename, ml_pipe, X, y, modeldir):    
    config_logging('FIT')
    taskname = F'training {ml_pipename}'
    logging.info(F'Started {taskname} with input_shape={X.shape}')
    random.seed(args.seed)
    np.random.seed(args.seed)

    X = drop_feat_from_X(ml_pipename, X)
    ml_pipename_in_fname = ml_pipename.replace('/', '_')
    prefilename = F'{modeldir}/{ml_pipename_in_fname}_model.pickle'
    if os.path.exists(prefilename):
        with open(prefilename, 'rb') as file:
            ml_pipe = pickle.load(file)
        logging.info(F'Used already-trained {ml_pipename}')
    else:
        try:
            ml_pipe.fit(X, y)
        except Exception as err:
            err_filename = F'{modeldir}/{ml_pipename_in_fname}_model_error.pickle'
            with open(err_filename, 'wb') as file:
                pickle.dump(ml_pipe, file)
            logging.info(F'Saved the ML pipeline {ml_pipename} with its runtime training error at {err_filename}')
            raise err        
        with open(prefilename, 'wb') as file:
            pickle.dump(ml_pipe, file)
        logging.info(F'Performed training of {ml_pipename}')
    prob_pred = ml_pipe.predict_proba(X)

    assert_prob_arr(prob_pred, taskname)
    logging.info(F'Ended {taskname}')
    return (ml_pipename, ml_pipe, prob_pred[:,1])

def predict_with_ml_pipe(ml_pipename, ml_pipe, X, modeldir):
    config_logging('PREDICT')
    taskname = F'predicting test-set labels using {ml_pipename}'
    logging.info(F'Started {taskname} with input_shape={X.shape}')
    random.seed(args.seed)
    np.random.seed(args.seed)
   
    X = drop_feat_from_X(ml_pipename, X)
    ml_pipename_in_fname = ml_pipename.replace('/', '_')
    with open(F'{modeldir}/{ml_pipename_in_fname}_model.pickle', 'rb') as file:
        ml_pipe2 = pickle.load(file)
        check_is_fitted(ml_pipe2)
        X2 = X.head(n=100)
        try:
            check_is_fitted(ml_pipe)
            y21 = ml_pipe.predict_proba(X2)
            y22 = ml_pipe2.predict_proba(X2)
            assert np.allclose(y21, y22), F'The ML pipeline {ml_pipename} was not saved properly because {y21} is not all-approx-equal to {y22}!'
        except NotFittedError as exc:
            logging.info(f"Model {ml_pipename} has not been trained in this run. ")
    prob_pred = ml_pipe2.predict_proba(X)

    assert_prob_arr(prob_pred, taskname)
    logging.info(F'Ended {taskname}')
    return (ml_pipename, ml_pipe, prob_pred[:,1])

def cross_val_predict_with_ml_pipe(ml_pipename, ml_pipe, X, y, partitions, fidx):
    config_logging('CROSS_VAL_PREDICT')
    taskname = F'predicting out-of-fold labels by cross validation using {ml_pipename}'
    logging.info(F'Start {taskname} with input_shape={X.shape}')
    random.seed(args.seed)
    np.random.seed(args.seed)

    X = drop_feat_from_X(ml_pipename, X)
    ml_pipename_in_fname = ml_pipename.replace('/', '_')
    prefilename = F'{modeldir}/{ml_pipename_in_fname}_{fidx}_cross_val_predict_results.pickle'
    if os.path.exists(prefilename):
        with open(prefilename, 'rb') as file:
            prob_pred = pickle.load(file)
    else:
        prob_pred = cross_val_predict(ml_pipe, X, y, groups=partitions, cv=GroupKFold(), method='predict_proba')
        with open(prefilename, 'wb') as file:
            pickle.dump(prob_pred, file)

    assert_prob_arr(prob_pred, taskname)
    logging.info(F'Ended {taskname}')
    return (ml_pipename, ml_pipe, prob_pred[:,1])

def cross_val_score_with_ml_pipe(ml_pipename, ml_pipe, X, y, partitions, fidx):
    config_logging('CROSS_VAL_SCORE')
    taskname = F'scoring by cross validation using {ml_pipename}'
    logging.info(F'Started {taskname} with input_shape={X.shape}')
    random.seed(args.seed)
    np.random.seed(args.seed)

    X = drop_feat_from_X(ml_pipename, X)
    ml_pipename_in_fname = ml_pipename.replace('/', '_')
    prefilename = F'{modeldir}/{ml_pipename_in_fname}_{fidx}_cross_val_score_results.pickle'
    if os.path.exists(prefilename):
        with open(prefilename, 'rb') as file:
            scores = pickle.load(file)
    else:
        scores = cross_val_score(ml_pipe, X, y, groups=partitions, cv=GroupKFold(), scoring='roc_auc', n_jobs=-1)
        with open(prefilename, 'wb') as file:
            pickle.dump(scores, file)
    
    # assert_prob_arr(scores, taskname)
    logging.info(F'Ended {taskname}')
    return (ml_pipename, ml_pipe, scores)

def compute_topN(df, labelcol, patientcol='Patient', predcol=F'{NG_default}/hParamDefault_LR', topN=20, ranking_mult=1):
    df = compute_ranked_df(df, labelcol, patientcol, predcol, ranking_mult=ranking_mult)
    return len([label for label in (df.loc[df['rank']<=topN,:][labelcol]) if label == 1])

def compute_metric(colname, colname2rocauc, metric_name, metric_val, df_in, labelcol, title_in_colname):
    if colname in colname2rocauc:
        #print(F'colname2rocauc[{colname}]={colname2rocauc[colname]}')
        #print(F'rocauclist={colname2rocauc[colname]}=')
        roc_auc = np.mean(colname2rocauc[colname])
        roc_auc_std = np.std(colname2rocauc[colname], ddof=1)
        sample_std = roc_auc_std
        n = len(colname2rocauc[colname])
        alpha = 1.0 - 0.9 # 90% confidence interval                
        dof = n - 1
        t_critical = stats.t.ppf(1 - alpha/2, dof)  # Two-tailed critical t-value
        moe = t_critical * (sample_std / np.sqrt(n)) # margin of error
    else:
        ranking_mult = (-1 if colname in ASCENDING_FEATURES else 1)
        sign2corr = {-1: 'negative', 1: 'positive'}
        logging.info(F'Computing the ({title_in_colname}) of {colname} with {sign2corr[ranking_mult]} feature-label correlation')
        #y_true = np.where(df_in[labelcol], 1 , 0)
        if metric_name == 'top':
            roc_auc = compute_topN(df_in, labelcol, patientcol='Patient', predcol=colname, topN=metric_val, ranking_mult=ranking_mult)
            #roc_auc, _ = compute_topN(y_true, df_in[colname], df_in['Patient'], metric_val)
        else:                    
            roc_auc = roc_auc_score(df_in[labelcol], ranking_mult*df_in[colname])
        #fpr, tpr, thresholds = metrics.roc_curve(train_df['response'], train_df[clfname], pos_label=1)
        #auc_df.loc[ft_preproc_name,classifier_name] = metrics.auc(fpr, tpr)
        roc_auc_std = np.nan
        moe = np.nan
    return (colname, roc_auc, moe, roc_auc_std)

def benchmark_perf_2(
        df_ins,
        out_fname_fmt,
        ft_preproc_names,
        classifier_names,
        features, # included in model
        ex_feats, # excluded from model
        labelcol, 
        colname2rocauc_list=[{}],
        metric_name='roc_auc', 
        metric_thresholds=[0], 
        titles=[''], 
        barh_fmt='%.4g'):
    n_subfigs = max((len(df_ins), len(colname2rocauc_list), len(metric_thresholds), len(titles)))
    assert len(df_ins) in [1, n_subfigs], F'Found {len(df_ins)} df_ins but only 1 and {n_subfigs} are allowed for generating {out_fname_fmt}!'
    assert len(colname2rocauc_list) in [1, n_subfigs], F'Found {len(colname2rocauc_list)} colname2rocauc_list but only 1 and {n_subfigs} are allowed for generating {out_fname_fmt}!'
    assert len(metric_thresholds) in [1, n_subfigs], F'Found {len(metric_thresholds)} colname2rocauc_list but only 1 and {n_subfigs} are allowed for generating {out_fname_fmt}!'
    assert len(titles) == n_subfigs, F'Please provide {n_subfigs} titles (current titles are: {titles}) to generate {out_fname_fmt}!'
    if len(df_ins) < n_subfigs: df_ins = [df_ins[0]] * n_subfigs
    if len(colname2rocauc_list) < n_subfigs: colname2rocauc_list = [colname2rocauc_list[0]] * n_subfigs
    if len(metric_thresholds) < n_subfigs: metric_thresholds = [metric_thresholds[0]] * n_subfigs
    
    fig_1, ax_1 = plt.subplots(figsize=(6*max((1.5,n_subfigs)), 6*3))
    ax_1.set_axis_off()
    gs = gridspec.GridSpec(2, n_subfigs, height_ratios=[1, 25])
    legend_ax = fig_1.add_subplot(gs[0,:])
    legend_ax.set_axis_off()
    axes = [fig_1.add_subplot(gs[1,j]) for j in range(n_subfigs)]
    for ax_idx, (df_in, colname2rocauc, metric_val, title) in enumerate(zip(df_ins, colname2rocauc_list, metric_thresholds, titles)):        
        def replace_non_alphanumeric(text): return ''.join([c if c.isalnum() else '_' for c in text])
        title_in_fname = '_' + replace_non_alphanumeric(title)
        title_in_colname = title_in_fname

        auc_series = pd.Series(np.nan, features)
        auc_series2 = pd.Series(np.nan, ex_feats)

        auc_df = pd.DataFrame(data=np.nan,
                index   = [name for name in ft_preproc_names],
                columns = [name for name in classifier_names])
        auc_std_df = pd.DataFrame(auc_df)
        
        colnames1 = [
            comb(ft_preproc_name, classifier_name) 
            for ft_preproc_name in ft_preproc_names for classifier_name in classifier_names]
        colnames = add_redundant_names(colnames1)
        colnames = sorted(set(features + ex_feats + colnames))
        colnames = [colname for colname in colnames if colname in df_in.columns]
        metric_results = Parallel(n_jobs=24)(delayed(compute_metric)(colname, colname2rocauc, metric_name, metric_val, 
            df_in[['Patient', labelcol, colname]], labelcol, title_in_colname) for colname in colnames)
        rows = []
        for colname, roc_auc, moe, roc_auc_std in metric_results:
            rows.append((colname, roc_auc, moe))
            if   colname in features:
                auc_series[colname] = roc_auc
            elif colname in ex_feats:
                auc_series2[colname] = roc_auc
            else:
                ft_preproc_name, classifier_name = decomb(colname)
                auc_df.loc[ft_preproc_name, classifier_name] = roc_auc
                auc_std_df.loc[ft_preproc_name, classifier_name] = roc_auc_std
        long_df = pd.DataFrame(rows, columns=['Method', title_in_colname, title_in_colname+'_moe']) # AUROC -> title_in_colname
        long_df.to_csv(out_fname_fmt.format('with_both' + title_in_fname), sep='\t', index=True)
        auc_series2.to_csv(out_fname_fmt.format('with_add_features' + title_in_fname), sep='\t', index=True)
        auc_series.to_csv(out_fname_fmt.format('with_raw_features' + title_in_fname), sep='\t', index=True)
        auc_df.to_csv(out_fname_fmt.format('with_featproc_clf_combs' + title_in_fname), sep='\t', index=True, index_label='FeatPreprocessors\\Classifiers')
        auc_std_df.to_csv(out_fname_fmt.format('with_featproc_clf_combs_std' + title_in_fname), sep='\t', index=True, index_label='FeatPreprocessors\\Classifiers')

        if False:
            fig_heat, ax_heat = plt.subplots(figsize=(9, 4))
            heatmap_ret = sns.heatmap(auc_df, annot=True, fmt='.4g', ax=ax_heat)
            fig_heat.tight_layout()
            fig_heat.savefig(out_fname_fmt.format('with_featproc_clf_combs')+'.pdf')
            fig_heat.savefig(out_fname_fmt.format('with_featproc_clf_combs')+'.png', dpi=600)
        #plt.close()
        #fig, ax = plt.subplots(figsize=(8, 8*2.5))
        ax = axes[ax_idx]
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ypos = list(range(len(long_df)))
        #auroc_method_class_list = zip(long_df[title_in_colname], long_df['Method'], long_df['Method'].apply(lambda x: (-1 if 'neoguider' in x.lower() else (1 if x in features else 0))))
        def meth2id(x):
            if x.startswith(NG_default): return (0 if '/hParamTuned_' in x else 1)
            if x in features: return 4
            if x in ex_feats: return 5
            return (2 if '/hParamTuned_' in x else 3)
        methclass2desc = {
            0: ('NeoGuider/hyperparameter-tuned classifier'),
            1: ('NeoGuider/default-hyperparameter classifier'),
            2:     ('Other/hyperparameter-tuned classifier'),
            3:     ('Other/default-hyperparameter classifier'),
            4: ('Single feature (included in model)'),
            5: ('Single feature (not included in model)')
        }
        # Method with higher priority is ranked before the one with lower priority to break a tie
        # We used these rules, based on the Occam's razor, to assign tie-breaking priorities:
        # Rule 1: single-feature model before multiple-feature model
        # Rule 2: model with default hyperparameters before model with tuned hyperparameters
        # Rule 3: more interpretable model before less interpretable model 
        #         (e.g., linear is more interpretable than non-linear, IC50 is more interpretable than %RankBA)
        methclass2priority = [1, 3, 0, 2, 5, 4]
        long_df['MethClass'] = long_df['Method'].apply(meth2id)
        long_df['MethClassPriority'] = [methclass2priority[mc] for mc in long_df['MethClass']]
        long_df = long_df.sort_values(by=[title_in_colname,'MethClassPriority','Method'])
        long_df['ypos'] = list(range(len(long_df)))
        methclass_df_iterable = long_df.groupby('MethClass')
        hbars_list = []
        methclass_list = []
        cmap = matplotlib.colormaps['tab20']
        smallest_fontsize = min((9, 900.0 / (1.0 + len(long_df))))
        for classidx, (methclass, df) in enumerate(sorted(methclass_df_iterable)):
            xerr = (df[title_in_colname+'_moe']).fillna(0)
            hbars = ax.barh(df['ypos'], df[title_in_colname], align='center', label=methclass2desc[methclass], color=cmap.colors[classidx], xerr=xerr)
            ax.bar_label(hbars, fmt=barh_fmt, padding=2, fontsize=smallest_fontsize)
            hbars_list.append(hbars)
            methclass_list.append(methclass)

        methodnames = [SOFT_NAME_TO_MANUSCRIPT_NAME.get(x, x) for x in long_df['Method']]
        for x in SOFT_NAME_TO_MANUSCRIPT_NAME:
            if x not in features: 
                methodnames = long_df['Method']
                break
        methodnames = [SOFT_NAME_TO_MANUSCRIPT_NAME_ALWAYS.get(x, x) for x in methodnames]
        ax.set_yticks(long_df['ypos'], labels=methodnames, fontsize=smallest_fontsize)
        ax.set_ylim(-1, len(long_df))
        xmin, xmax = np.min(long_df[title_in_colname]), np.max(long_df[title_in_colname] + long_df[title_in_colname+'_moe'].fillna(0))
        ax.set_xlim(xmin - (xmax - xmin) * 0.0, xmax + (xmax - xmin) * 0.2)
        ax.set_xlabel(titles[ax_idx], fontsize=14)
        #ax.legend(fontsize=14)
        def get_ncols(n_labels, n_cols):
            n_cols = int(round(min((1.0*n_cols, n_labels))))
            while n_labels % n_cols != 0: n_cols -= 1
            return n_cols
        if ax_idx == 0: legend_ax.legend(hbars_list, [methclass2desc[i] for i in sorted(methclass_list)], title='Feature-preprocessing-technique/classifier combinations',
                ncol=get_ncols(len(long_df['MethClass'].unique()), n_subfigs),
                loc='center', fontsize=12, title_fontsize=18)

    plt.tight_layout()
    logging.info(F'''Saving pdf and png figures to {out_fname_fmt.format('with_both')}''')
    plt.savefig(out_fname_fmt.format('with_both')+'.pdf')
    plt.savefig(out_fname_fmt.format('with_both')+'.png', dpi=600)
    plt.close()

def benchmark_performance(
        df_ins,
        out_fname_fmt,
        features, # included in model
        ex_feats, # excluded from model
        labelcol,
        colname2rocauc_list=[{}],
        metric_name='roc_auc',
        metric_thresholds=[0],
        titles=[''],
        barh_fmt='%.4g'):
    
    benchmark_perf_2(
        df_ins,
        out_fname_fmt + '_stage1', 
        list(set(HPARAM_DEFLT_FT_PREPROC_NAME2TECH.keys()) | set(HPARAM_TUNED_FT_PREPROC_NAME2TECH.keys())),
        list(set(HPARAM_DEFLT_CLASSIFIER_NAME2TECH.keys()) | set(HPARAM_TUNED_CLASSIFIER_NAME2TECH.keys())),
        features,
        ex_feats,
        labelcol,
        colname2rocauc_list, metric_name, metric_thresholds, titles, barh_fmt)
    benchmark_perf_2(
        df_ins,
        out_fname_fmt + '_stage2', 
        FINAL_FT_PREPROC_NAMES,
        FINAL_CLASSIFIER_NAMES,
        features,
        ex_feats,
        labelcol,
        colname2rocauc_list, metric_name, metric_thresholds, titles, barh_fmt)

def x_allin_y(X, Y):
    for x in X:
        if not x in Y: return False
    return True

def match_col(df, colnames):
    ret = ''
    for colname in colnames:
        if colname in df.columns:
            assert ret == '', F'The columns {ret} and {colname} are both found, aborting!'
            ret = colname
    return ret

# We used the logic of Muller et al., 2023, Immunity at https://doi.org/10.1016/j.immuni.2023.09.002
def prepare_df(df, labelcol, na_op, max_peplen): 
    ret = df.copy()
    pepcol = match_col(ret, ['MT_pep', 'MT_pep_y', 'Mut_peptide'])
    if pepcol:
        ret = ret.loc[ret[pepcol].str.len() <= max_peplen]
    added_feats = []
    #if x_allin_y(IMPROVE_FTS[0:10] + ['Patient', 'Partition'], ret.columns):
    patientcol = match_col(ret, ['Patient', 'PatientID', 'patient'])
    if patientcol: # and 'ln_NumTested' not in ret.columns:
        patient2ntested = collections.defaultdict(int)
        for patient, label in zip(ret[patientcol], ret[labelcol]):
            patient2ntested[patient] += (1 if label in [0, 1] else 0)
        newcol = [np.log(max((1, patient2ntested[p]))) for p in ret[patientcol]]
        if 'ln_NumTested' not in ret.columns:
            ret['ln_NumTested'] = newcol
            added_feats.append('ln_NumTested')
            logging.info(F'Added the column ln_NumTested')
        else:
            assert np.allclose(ret['ln_NumTested'], np.array(newcol))
    if not ('Patient' in ret.columns): ret['Patient'] = ret[patientcol]
    if na_op == 'drop':
        ret = ret.loc[df[labelcol] != -1] # -1 means NotAvailable
    elif na_op == 'zero':
        ret = ret.copy()
        ret[labelcol] = [(0 if x == -1 else x) for x in ret[labelcol]]
    else:
        raise ValueError(F'The value of na_op cannot be `{na_op}` because only `drop` and `zero` are valid ones. ')
    print(F'prep_df=\n{ret}\n')
    return ret, added_feats

DHP_FEATS = ['DeepHLApan_binding_score', 'DeepHLApan_immunogenic_score']
def add_more(df, fpath):
    fpath = os.path.abspath(fpath)
    #fdir = os.path.dirname(fpath)
    #fbase = os.path.basename(fpath)
    fbase, fext = os.path.splitext(fpath)
    # Add DeepHLpan results
    dhp_fname = fbase + '_predicted_result.csv'
    ret = df
    if os.path.exists(dhp_fname):
        df2 = pd.read_csv(dhp_fname, header=0)
        assert list(df2.columns) == 'Annotation,HLA,Peptide,binding score,immunogenic score'.split(',')
        assert len(df2) == len(df), F'{len(df2)} == {len(df1)} failed!'
        df3 = df2[['binding score', 'immunogenic score']]
        df3.columns = DHP_FEATS
        ret = pd.concat([df, df3], axis=1)
        logging.info(F'Concatenate with {dhp_fname} to return {ret}')
    else: logging.warning(F'The filepath {dhp_fname} does not exist. ')
    return ret

def get_filenames(filepaths, prefix=''):
    return [(prefix + x.split('/')[-1].split('.')[0]) for x in filepaths]

OTHER_FEATS = ['%Rank_EL', 'Score_BA', '%Rank_BA', 'PRIME_rank', 'PRIME_score', 'PRIME_BArank', 'mhcflurry_aff_percentile', 'mhcflurry_presentation_percentile']
def train_test_cv(train_fnames, test_fnames, cv_fnames):
    output = args.output # csvsep from global
    tasks = args.tasks
    feature_names = args.features
    label_name = args.label
    untest_flag = args.untest_flag 
    peplen_flag = args.peplen_flag

    ex_feats = DHP_FEATS + OTHER_FEATS

    untest_ops_training_examples = ('drop' if (untest_flag & 0x1) else 'zero')
    untest_ops_test_examples = ('drop' if (untest_flag & 0x2) else 'zero')
    untest_ops_cv_examples = ('drop' if (untest_flag & 0x4) else 'zero')
    
    peplen_max_training_examples = (11 if (peplen_flag & 0x1) else 9999)
    peplen_max_test_examples = (11 if (peplen_flag & 0x2) else 9999)
    peplen_max_cv_examples = (11 if (peplen_flag & 0x4) else 9999)
    
    HLA_COLS= ['HLA_type', 'HLA_type_y', 'HLA_allele', 'mutant_best_alleles_netMHCpan'] # HLA_type_x can contain comma
    # setup
    hparam_deflt_ft_preproc_name2tech = {x: HPARAM_DEFLT_FT_PREPROC_NAME2TECH[x] for x in args.hparam_deflt_ft_preproc_names}
    hparam_deflt_classifier_name2tech = {x: HPARAM_DEFLT_CLASSIFIER_NAME2TECH[x] for x in args.hparam_deflt_classifier_names}
    hparam_tuned_ft_preproc_name2tech = {x: HPARAM_TUNED_FT_PREPROC_NAME2TECH[x] for x in args.hparam_tuned_ft_preproc_names}
    hparam_tuned_classifier_name2tech = {x: HPARAM_TUNED_CLASSIFIER_NAME2TECH[x] for x in args.hparam_tuned_classifier_names}
    ALL_FEATURES = []
    for fts in LISTOF_FEATURES:
        ALL_FEATURES.extend(fts)
    features_superset1 = (ALL_FEATURES if len(feature_names) == 0 else feature_names) # (.split(','))
    labels_superset1 = (LISTOF_LABELS[0] if label_name == '' else [label_name])
    labelcol = None
    hlacol = ''
    in_dfs = []
    for i, train_fname in enumerate(train_fnames):
        in_df = pd.read_csv(train_fname, sep=csvsep)
        in_df = add_more(in_df, train_fname)
        if i == 0:
            features = [colname for colname in in_df.columns if colname in features_superset1]
            ft_weights = [len(set(fts) & set(features)) for fts in LISTOF_FEATURES]
            features_2 = LISTOF_FEATURES[np.argmax(ft_weights)]
            features = [colname for colname in in_df.columns if colname in features_2]
            assert len(features) >= len(features_2) / 2, (F'The features {features} and {features_2} share less than 50% names in common! '
                    'Please use the --features cmd-line option to specify the exact feature (column) names to used. ')
            labels = [colname for colname in in_df.columns if colname in labels_superset1]
            hlacols = [colname for colname in in_df.columns if colname in HLA_COLS]
            assert len(labels) == 1, F'Multiple label names ({labels}) are found, please use the --label cmd-line option to specify the exact label (column) name to use. '
            assert len(hlacols) <= 1, F'Found multiple HLA column names: {hlas}'
            labelcol = labels[0]
            if hlacols: hlacol = hlacols[0]
            in_df, added_feats = prepare_df(in_df, labelcol, na_op=untest_ops_training_examples, max_peplen=peplen_max_training_examples)
            if added_feats: features.extend(added_feats)
            features_superset1 = features
            labels_superset1 = labels
        else:
            in_df, _ = prepare_df(in_df, labelcol, na_op=untest_ops_training_examples, max_peplen=peplen_max_training_examples)
        if in_dfs and not (in_dfs[0].columns == in_df.columns).all():
            logging.warning(F'{in_dfs[0].columns} == {in_df.columns} failed for the column names of the inputs {train_fnames[0]} and {train_fname}')
        in_dfs.append(in_df)
    train_df = pd.concat(in_dfs, join="inner")
    if 'hla1' in tasks: analyze_hla(train_df, hlacol, labelcol, f'{output}_train_hla_stats.pdf')

    features = [f for f in features if f in train_df.columns]
    logging.info(F'Selected, from {train_fnames}, the features {features} (n={len(features)})')

    # feature analysis phase 1: feature importance
    train_X = train_df.loc[:, features].copy()
    big_y   = train_df.loc[:, labelcol].copy()
    train_X = train_X.apply(pd.to_numeric)
    big_y   = big_y.apply(pd.to_numeric)

    ft_preproc_tech = HPARAM_DEFLT_FT_PREPROC_NAME2TECH[F'{NG_default}']
    #train_X = QuantileTransformer(random_state=args1.rand).fit_transform(train_X)
    #train_X = pd.DataFrame(train_X, columns=features)
    big_transformed_X = ft_preproc_tech.fit_transform(train_X, big_y)
    ft_preproc_tech_feature_names = ft_preproc_tech.get_feature_names()
    ft_preproc_tech_feature_importances_1 = ft_preproc_tech.get_feature_importances('f2l')
    ft_preproc_tech_feature_importances_2 = ft_preproc_tech.get_feature_importances('f2f')
    ft_preproc_tech_feature_importances_3 = ft_preproc_tech.get_feature_importances('f2l2f')

    ft_preproc_tech_feature_importances_p1 = ft_preproc_tech.get_feature_importances('pvalue', 'mannwhitneyu')
    ft_preproc_tech_feature_importancesH01 = ft_preproc_tech.get_feature_importances('h0_assume_correlation_pvalue', 'mannwhitneyu')

    ft_preproc_tech_feature_importances_p2 = ft_preproc_tech.get_feature_importances('pvalue', 'spearmanr')
    ft_preproc_tech_feature_importancesH02 = ft_preproc_tech.get_feature_importances('h0_assume_correlation_pvalue', 'spearmanr')

    ft_preproc_tech_feature_importances_p3 = ft_preproc_tech.get_feature_importances('pvalue', 'odds_spearmanr')
    ft_preproc_tech_feature_importances_s1 = ft_preproc_tech.get_feature_importances('statistic', 'mannwhitneyu')
    ft_preproc_tech_feature_importances_s2 = ft_preproc_tech.get_feature_importances('statistic', 'spearmanr')
    ft_preproc_tech_feature_importances_s3 = ft_preproc_tech.get_feature_importances('statistic', 'odds_spearmanr')
    ft_preproc_tech_feature_importances_t1 = ft_preproc_tech.get_feature_importances('trend', 'mannwhitneyu')
    ft_preproc_tech_feature_importances_t2 = ft_preproc_tech.get_feature_importances('trend', 'spearmanr')
    ft_preproc_tech_feature_importances_t3 = ft_preproc_tech.get_feature_importances('trend', 'odds_spearmanr')

    feat_importance_df = pd.DataFrame.from_dict({
            'feature_names'            : ft_preproc_tech_feature_names,
            'feat_to_label_importances': ft_preproc_tech_feature_importances_1,
            'feat_to_feat_importances' : ft_preproc_tech_feature_importances_2,
            'feat_to_lab_to_feat_imps' : ft_preproc_tech_feature_importances_3,
            'effectSize=0_H0_mannwhitR_pvalue': ft_preproc_tech_feature_importances_p1,
            'effectSize=0_H0_spearmanR_pvalue': ft_preproc_tech_feature_importances_p2,
            'effectSize=0_H0_odds_spearmanR_pvalue': ft_preproc_tech_feature_importances_p3,
            'statistic_mannwhitR': ft_preproc_tech_feature_importances_s1,
            'statistic_spearmanR': ft_preproc_tech_feature_importances_s2,
            'statistic_odds_spearmanR': ft_preproc_tech_feature_importances_s3,
            'trend_mannwhitR': ft_preproc_tech_feature_importances_t1,
            'trend_spearmanR': ft_preproc_tech_feature_importances_t2,
            'trend_odds_spearmanR': ft_preproc_tech_feature_importances_t3,
    })

    for effect_size, p_values in sorted(ft_preproc_tech_feature_importancesH01.items()):
        feat_importance_df[F'effectSize>={effect_size}_H0_mannwhitR_pvalue'] = p_values
    for effect_size, p_values in sorted(ft_preproc_tech_feature_importancesH02.items()):
        feat_importance_df[F'effectSize>={effect_size}_H0_spearmanR_pvalue'] = p_values

    feat_importance_df.to_csv(f'{output}_feat_imp.tsv', index='feature_names', sep='\t')

    ilr = ft_preproc_tech
    s1x = ilr.get_density_estimated_X()
    s1y = ilr.get_density_estimated_log_odds()
    s2x = ilr.get_isotonic_X()
    s2y = ilr.get_isotonic_log_odds()
    s3x = ilr.get_centered_isotonic_X()
    s3y = ilr.get_centered_isotonic_log_odds()

    # feature analysis phase 2: feature importance plot
    if 'fa2' in tasks:
        with PdfPages(f'{output}_feat.pdf') as pdf:
            for feature_idx, feature_name in enumerate(ft_preproc_tech_feature_names):
                fig, axes = plt.subplots(2, 1, height_ratios=[1, 1], layout='constrained')
                fig.set_figheight(1.500*2.222)
                fig.set_figwidth(2.250*3.333)
                x1 = [x for (x,y) in zip(train_X[feature_name], big_y) if y == 1]
                x0 = [x for (x,y) in zip(train_X[feature_name], big_y) if y == 0]
                axes[0].hist([x1, x0],
                        label=['Tested positive ($A_f$)', 'Tested negative ($B_f$)'],
                        color=[(0.75, 0.00, 0.00), (0.25, 0.25, 0.25)],
                        bins=40,
                        log=True)
                #print(F'Plot {len(s1x[feature_idx])} {len(s1y)}')
                axes[1].plot(s1x[feature_idx], s1y[feature_idx], label='After step 1: adaptive KDE',             alpha = 0.200, marker = '^', linewidth=0.5, markersize=(16*3)**0.5)
                axes[1].plot(s2x[feature_idx], s2y[feature_idx], label='After step 2: isotonic regression (IR)', alpha = 0.300, marker = '<', linewidth=0.5, markersize=(16*2)**0.5)
                axes[1].plot(s3x[feature_idx], s3y[feature_idx], label='After step 3: centered IR (CIR)',        alpha = 0.600, marker = '>', linewidth=0.5, markersize=(16*1)**0.5)
                impp = ft_preproc_tech_feature_importances_p1[feature_idx]
                imp1, imp2, imp3 = ft_preproc_tech_feature_importances_1[feature_idx], ft_preproc_tech_feature_importances_2[feature_idx], ft_preproc_tech_feature_importances_3[feature_idx]
                axes[1].set_xlabel(feature_name + F' percentile\nimportances: p_value={impp:.2G}, to_label={imp1:.4f}, to_features={imp2:.4f}, to_both_combined={imp3:.4f}')
                axes[1].set_ylabel('Estimated log odds')
                axes[1].legend(title='Feature values')
                pdf.savefig()
                plt.close()

    # feature analysis phase 3: feature-vs-feature pair-plot
    if 'fa3' in tasks:
        big_transformed_df = pd.DataFrame(np.append(big_transformed_X, np.array([[v] for v in big_y]), axis=1), columns=list(features)+[labelcol])
        big_transformed_df = big_transformed_df.apply(pd.to_numeric)
        big_trans_df0 = big_transformed_df.loc[big_transformed_df[labelcol]==0,:] #.sample(n=100, random_state=args1.rand)
        big_trans_df1 = big_transformed_df.loc[big_transformed_df[labelcol]==1,:] #.sample(n=100, random_state=args1.rand)

        big_trans1_df0 = big_trans_df0.loc[:,features] #.sample(n=100, random_state=args1.rand)
        big_trans1_df1 = big_trans_df1.loc[:,features] #.sample(n=100, random_state=args1.rand)

        dfsize = min((len(big_trans_df0), len(big_trans_df1)))
        logging.info(F'Min_nrows={dfsize}')
        dfsize = min((dfsize, 100))
        logging.info(F'Start plotting all neoepitope candidates')
        plot_ret = pairplot_showing_pretrans_feat_vals(big_trans1_df0.sample(n=dfsize, random_state=args1.rand), big_trans1_df1.sample(n=dfsize, random_state=args1.rand), ilr)
        logging.info(F'Mid plotting all neoepitope candidates')
        #sns.pairplot(pd.concat([big_trans_df0.sample(n=dfsize, random_state=args1.rand), big_trans_df1.sample(n=dfsize, random_state=args1.rand)]), hue=labelcol)
        plt.savefig(f'{output}_pairwiseLogOdds.pdf')
        plt.close()
        logging.info(F'End plotting all neoepitope candidates')
        if 'mhcflurry_presentation_percentile' in big_transformed_df.columns:
            big_trans2_df0 = big_trans_df0.loc[big_trans_df0['mhcflurry_presentation_percentile']<=5,features] #big_transformed_df.loc[big_transformed_df[labelcol]==0,:].sample(n=100, random_state=args1.rand)
            big_trans2_df1 = big_trans_df1.loc[big_trans_df1['mhcflurry_presentation_percentile']<=5,features]
            dfsize = min((len(big_trans2_df0), len(big_trans2_df1)))
            logging.info(F'Min_nrows={dfsize}')
            dfsize = min((dfsize, 100))
            dfsize_0 = min((len(big_trans2_df0), 1000))
            #plot_ret = sns.pairplot(pd.concat([big_trans2_df0.sample(n=dfsize_0, random_state=args1.rand), big_trans2_df1.sample(n=dfsize, random_state=args1.rand)]), hue=labelcol)
            plot_ret = pairplot_showing_pretrans_feat_vals(big_trans2_df0.sample(n=dfsize_0, random_state=args1.rand), big_trans2_df1.sample(n=dfsize, random_state=args1.rand), ilr)
            plt.savefig(f'{output}_pairwiseLogOdds_mhcflurry_presentation_5perc.pdf')
            plt.close()

    ml_pipes = construct_ml_pipes(hparam_deflt_ft_preproc_name2tech, hparam_deflt_classifier_name2tech, hparam_tuned_ft_preproc_name2tech, hparam_tuned_classifier_name2tech, big_y)
   
    # train phase
    train_X = train_df.loc[:, features].copy()
    train_y = train_df.loc[:, labelcol].copy()
    train_X = train_X.fillna({col : np.mean(train_X[col]) for col in features})
    
    logging.info(F'Start training')
    train_results = Parallel(n_jobs=24)(delayed(train_ml_pipe)(ml_pipename, ml_pipe, train_X, train_y, modeldir) for ml_pipename, ml_pipe in ml_pipes)
    logging.info(F'End training')
    if not os.path.exists(f'{output}_train.csv.gz.done'):
        logging.info(F'Start saving training-set predictions to {output}_train.csv.gz')
        for result in train_results:
            ml_pipename, ml_pipe, ml_pipe_predicted = result
            train_df[ml_pipename] = ml_pipe_predicted
        train_df.to_csv(f'{output}_train.csv.gz', sep=',', index=None, compression={'method': 'gzip', 'compresslevel': 1, 'mtime': 1})
        with open(f'{output}_train.csv.gz.done', 'w') as file: file.write('done')
        logging.info(F'End saving training-set predictions to {output}_train.csv.gz')
    else:
        logging.info(F'Skip saving pre-saved training-set predictions at {output}_train.csv.gz')
    
    test_dfs = []
    for fidx, test_fname in enumerate(test_fnames):
        fidx += 1
        if test_fname in train_fnames: train_or_test = 'train'
        else: train_or_test = 'test'
        df = pd.read_csv(test_fname, sep=csvsep)
        df = add_more(df, test_fname)
        df, added_feats = prepare_df(df, labelcol, na_op=untest_ops_test_examples, max_peplen=peplen_max_test_examples)
        for f in added_feats: assert f in features, F'{f} in {features} failed!'
        dfXy = df.loc[:,features + [labelcol]]
        #assert (train_df.columns == test_df.columns).all(), F'{train_df.columns} == {test_df.columns} failed for the column names of the inputs {train_fnames} and {test_fname}'
        # test phase
        X = dfXy.loc[:, features].copy()
        X = X.fillna({col : np.mean(X[col]) for col in features})
        test_results = Parallel(n_jobs=para_n_jobs)(delayed(predict_with_ml_pipe)(ml_pipename, ml_pipe, X, modeldir) for ml_pipename, ml_pipe, _, in train_results)
        for result in test_results:
            ml_pipename, ml_pipe, ml_pipe_predicted = result
            assert not np.isnan(ml_pipe_predicted).any()
            df[ml_pipename] = ml_pipe_predicted
        df.to_csv(F'{output}_{fidx}_test.csv.gz', sep=',', index=None, compression={'method': 'gzip', 'compresslevel': 1, 'mtime': 1})
        df2 = df.fillna({col : np.mean(df[col]) for col in features})
        test_dfs.append(df2)
        if 'Patient' in df2.columns:
            benchmark_performance([df2], F'{output}_{fidx}_{train_or_test}_topN_{{}}', 
                features, ex_feats, labelcol, [{}],
                metric_name='top', metric_thresholds=[20,50,100], titles=['Top-20 #True', 'Top-50 #True', 'Top-100 #True'])
        if 'hla1' in tasks: analyze_hla(df2, hlacol, labelcol, F'{output}_{fidx}_{train_or_test}_hla_stats.pdf')
        if 'hla2' in tasks:
            logging.info(F'start analyze_performance_per_hla({df}, {hlacol}, {labelcol}, `_{fidx}_{train_or_test}_hla_bench.pdf`)')
            analyze_performance_per_hla(df, hlacol, labelcol, F'{output}_{fidx}_{train_or_test}_hla_bench.pdf')
            logging.info(F'end analyze_performance_per_hla({df}, {hlacol}, {labelcol}, `_{fidx}_{train_or_test}_hla_bench.pdf`)')
    if test_dfs:
        benchmark_performance(test_dfs, F'{output}_0_{train_or_test}_roc_auc_{{}}', 
            features, ex_feats, labelcol, [{}], 
            metric_name='roc_auc', metric_thresholds=[0], titles=get_filenames(test_fnames, 'AUC-ROC with\nfeature_set='))

    cv_pred_dfs = []
    pipename2score_list = []
    for fidx, fname in enumerate(cv_fnames):
        fidx += 1
        in_df = pd.read_csv(fname, sep=csvsep)
        in_df = add_more(in_df, fname)
        features = [colname for colname in in_df.columns if colname in features_superset1]
        labels = [colname for colname in in_df.columns if colname in labels_superset1]
        hlacols = [colname for colname in in_df.columns if colname in HLA_COLS]
        assert len(labels) == 1
        assert len(hlacols) <= 1, F'Found multiple HLA column names: {hlas}'
        labelcol = labels[0]
        if hlacols: hlacol = hlacols[0]
        
        df, added_feats = prepare_df(in_df, labelcol, na_op=untest_ops_cv_examples, max_peplen=peplen_max_cv_examples)
        features.extend(added_feats)
        dfXy = df.loc[:,features + [labelcol]]
        X = dfXy.loc[:, features].copy()
        X = X.fillna({col : np.mean(X[col]) for col in features})
        y = dfXy.loc[:, labelcol].copy()
        partition_name = None
        THE_PARTITION_NAMES = ['Partition', 'Patient', 'MT_pep', 'ET_pep', 'Epitope']
        for partition_name_1 in THE_PARTITION_NAMES:
            if partition_name_1 in df.columns:
                if partition_name != None:
                    logging.error(F'The partition names {partition_name} and {partition_name_1} cannot co-exist in the tabular file {fname}, keep using {partition_name}! ')
                else: partition_name = partition_name_1
        assert partition_name != None, F'The file {fname} does not contain any of the partitions names {THE_PARTITION_NAMES} as its column name! '
        
        results = Parallel(n_jobs=para_n_jobs)(delayed(cross_val_predict_with_ml_pipe)(ml_pipename, ml_pipe, X, y, df[partition_name], fidx) for ml_pipename, ml_pipe in ml_pipes)
        
        assert len(results) == len(ml_pipes), F'{len(results)} == {len(ml_pipes)} failed!'
        for result in results:
            ml_pipename, ml_pipe, ml_pipe_predicted = result
            df[ml_pipename] = ml_pipe_predicted
        df.to_csv(F'{output}_{fidx}_cv_predict.csv.gz', sep=',', index=None, compression={'method': 'gzip', 'compresslevel': 1, 'mtime': 1})
        df2 = df.fillna({col : np.mean(df[col]) for col in features})
        cv_pred_dfs.append(df2)
        prefilename = F'{output}_{fidx}_cv_score.pickle'
        #if os.path.exists(prefilename):
        #    with open(prefilename, 'rb') as file:
        #        results = pickle.load(file)
        #else:
        results = Parallel(n_jobs=para_n_jobs)(delayed(cross_val_score_with_ml_pipe)(ml_pipename, ml_pipe, X, y, df[partition_name], fidx) for ml_pipename, ml_pipe in ml_pipes)
        #with open(prefilename, 'wb') as file:
        #    pickle.dump(results, file)
        assert len(results) == len(ml_pipes), F'{len(results)} == {len(ml_pipes)} failed!'

        pipename2score = {ml_pipename : results[i][2] for i, (ml_pipename, ml_pipe) in enumerate(ml_pipes)}
        pipename2score_list.append(pipename2score)
    if cv_fnames:
        benchmark_performance(cv_pred_dfs, F'{output}_0_cv_score_roc_auc_{{}}',
            features_superset1, ex_feats, labelcol, pipename2score_list, titles=get_filenames(cv_fnames, 'AUC-ROC with\nfeature_set='))
        benchmark_performance(cv_pred_dfs, F'{output}_0_cv_predict_roc_auc_{{}}', 
            features_superset1, ex_feats, labelcol, [{}],                titles=get_filenames(cv_fnames, 'AUC-ROC with\nfeature_set='))
        
def main():
    config_logging('main')
    output = args.output    
    os.makedirs(modeldir, exist_ok=True)
    os.system(F'cp {scriptpath} {modeldir}')
    with open(modeldir + '/logged_cmd.sh', 'w') as file:
        file.write(F'#DATETIME={datetime.datetime.now().isoformat()}\n')
        file.write(F'#SCRIPT_DIR={scriptdir}\n')
        file.write(F'#CURRENT_WORKING_DIR={os.getcwd()}\n')
        for a in sys.argv: file.write(a + ' \\\n')

    tr_filenames = [filename for i, filename in enumerate(args.input) if ((i % 2 == 1) and 'tr' in args.input[i-1].split(','))]
    te_filenames = [filename for i, filename in enumerate(args.input) if ((i % 2 == 1) and 'te' in args.input[i-1].split(','))]
    cv_filenames = [filename for i, filename in enumerate(args.input) if ((i % 2 == 1) and 'cv' in args.input[i-1].split(','))]
    logging.info(F'tr_files={tr_filenames} te_files={te_filenames} cv_files={cv_filenames}')
    with open(args.output + '.info', 'w') as infofile:
        infofile.write('\t'.join(sys.argv) + '\n')
        infofile.write(str(args))
        infofile.write(F'Train: {tr_filenames}')
        infofile.write(F'Benchmark (test): {tr_filenames}')
        infofile.write(F'CrossValidate: {cv_filenames}')
    train_test_cv(tr_filenames, te_filenames, cv_filenames)

if __name__ == '__main__': main()

