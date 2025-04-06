#!/usr/bin/env python

import argparse, collections, copy, json, itertools, logging, os, pickle, pprint, random, sys

from joblib import Parallel, delayed # multiprocessing can hang if the virtual memory allocated is too big

import numpy as np
import pandas as pd

from collections import defaultdict, namedtuple

from scipy import stats

import matplotlib
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt
matplotlib.use('Agg')  # Use a non-GUI backend

import seaborn as sns

import sklearn
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import ColumnTransformer
# Modified from https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis)
from sklearn.ensemble import (AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier,)
from sklearn.feature_selection import VarianceThreshold
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, cross_val_score, GroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

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

from xgboost import XGBClassifier

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
                ax.hist([df1.iloc[:, i], df2.iloc[:, i]], bins=10, alpha=0.7)
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

# https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html
THE_FEAT_PREPROC_TECHS = {
    'Identity'            : ColumnTransformer([], remainder='passthrough'),
    'MaxAbsScaler'        : MaxAbsScaler(),
    'MinMaxScaler'        : MinMaxScaler(),
    'Normalizer'          : Normalizer(),
    'PowerTransformer'    : PowerTransformer(),
    'QuantileTransformer' : QuantileTransformer(random_state=0),
    'RobustScaler'        : RobustScaler(),
    'StandardScaler'      : StandardScaler(),
     # 'StandardTransformer' : QuantileTransformer(random_state=0, output_distribution='normal'),
    'NG'                  : 'Not-specified-yet', # IsotonicLogisticRegression(random_state=0),
    'NG_withoutNumTested' : 'Not-specified-yet', # IsotonicLogisticRegression(random_state=0),

    #'NeoGuider(P<0.0001)' : 'Not-specified-yet', # IsotonicLogisticRegression(random_state=0),
    #'NeoGuider(P<0.0002)' : 'Not-specified-yet', # IsotonicLogisticRegression(random_state=0),
    #'NeoGuider(P<0.0005)' : 'Not-specified-yet', # IsotonicLogisticRegression(random_state=0),
    #'NeoGuider(P<0.001)'  : 'Not-specified-yet', # IsotonicLogisticRegression(random_state=0),
    #'NeoGuider(P<0.01)'   : 'Not-specified-yet', # IsotonicLogisticRegression(random_state=0),
    #'NeoGuider(P<0.10)'   : 'Not-specified-yet', # IsotonicLogisticRegression(random_state=0),
    #'NeoGuider(P<=1.0)'   : 'Not-specified-yet', # IsotonicLogisticRegression(random_state=0),
}

# https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
THE_CLASSIFIERS = {
    'KNN': KNeighborsClassifier(),
    # 'KN500': KNeighborsClassifier(n_neighbors=500), # Too many ties in probabilities (density of positive examples is too low)
    # 'SVC': SVC(probability=True), # sklearn.svm import SVC is not designed to predict probability and is not designed to handle large sample size (O(n*n) runtime, intractable)
    # 'GP': GaussianProcessClassifier(random_state=0), # Error: numpy.core._exceptions._ArrayMemoryError: Unable to allocate 1.25 TiB for ...
    'DT': DecisionTreeClassifier(random_state=0),
    'RF': RandomForestClassifier(random_state=0),
    'MLP': MLPClassifier(random_state=0),
    'AB': AdaBoostClassifier(random_state=0),
    'GNB': GaussianNB(),
    #'LDA': LinearDiscriminantAnalysis(),
    'QDA': QuadraticDiscriminantAnalysis(),
    'LR' : LogisticRegression(random_state=0),
    # Tree-based algorithms other than the ones used above:
    # 'ET': ExtraTreesClassifier(random_state=0), # worse than RF
    # 'eGB': GradientBoostingClassifier(random_state=0), # similar to XGB
    'XGB': XGBClassifier(random_state=0),
}

# from https://github.com/SchubertLab/benchmark_TCRprediction
SOFTS60 = 'predictions_atm-tcr,predictions_attntap_MCPAS,predictions_attntap_VDJDB,predictions_bertrand,predictions_dlptcr_ALPHA,predictions_dlptcr_BETA,predictions_epitcr_WITH_MHC,predictions_epitcr_WO_MHC,predictions_ergo-i_AE_MCPAS,predictions_ergo-i_AE_VDJDB,predictions_ergo-i_LSTM_MCPAS,predictions_ergo-i_LSTM_VDJDB,predictions_ergo-ii_MCPAS,predictions_ergo-ii_VDJDB,predictions_imrex_DOWNSAMPLED,predictions_imrex_FULL,predictions_itcep,predictions_nettcr_t.0.v.1,predictions_nettcr_t.0.v.2,predictions_nettcr_t.0.v.3,predictions_nettcr_t.0.v.4,predictions_nettcr_t.1.v.0,predictions_nettcr_t.1.v.2,predictions_nettcr_t.1.v.3,predictions_nettcr_t.1.v.4,predictions_nettcr_t.2.v.0,predictions_nettcr_t.2.v.1,predictions_nettcr_t.2.v.3,predictions_nettcr_t.2.v.4,predictions_nettcr_t.3.v.0,predictions_nettcr_t.3.v.1,predictions_nettcr_t.3.v.2,predictions_nettcr_t.3.v.4,predictions_nettcr_t.4.v.0,predictions_nettcr_t.4.v.1,predictions_nettcr_t.4.v.2,predictions_nettcr_t.4.v.3,predictions_panpep,predictions_pmtnet,predictions_stapler,predictions_tcellmatch_GRU_CV0,predictions_tcellmatch_GRU_CV1,predictions_tcellmatch_GRU_CV2,predictions_tcellmatch_GRU_SEP_CV0,predictions_tcellmatch_GRU_SEP_CV1,predictions_tcellmatch_GRU_SEP_CV2,predictions_tcellmatch_LINEAR_CV0,predictions_tcellmatch_LINEAR_CV1,predictions_tcellmatch_LINEAR_CV2,predictions_tcellmatch_LSTM_CV0,predictions_tcellmatch_LSTM_CV1,predictions_tcellmatch_LSTM_CV2,predictions_tcellmatch_LSTM_SEP_CV0,predictions_tcellmatch_LSTM_SEP_CV1,predictions_tcellmatch_LSTM_SEP_CV2,predictions_teim,predictions_teinet_LARGE_DS,predictions_teinet_SMALL_DS,predictions_titan,predictions_tulip-tcr'.split(',')

SOFTS = 'atm-tcr,attntap_VDJDB,bertrand,dlptcr_BETA,epitcr_WITH_MHC,ergo-i_AE_VDJDB,ergo-ii_VDJDB,imrex_FULL,itcep,nettcr_t.1.v.0,panpep,pmtnet,stapler,tcellmatch_LINEAR_CV1,teim,teinet_SMALL_DS,titan,tulip-tcr'.split(',')

#cohort Mut_peptide HLA_allele Patient Partition
IMPROVE_FTS = 'Aro mw pI Inst CysRed RankEL RankBA NetMHCExp Expression SelfSim Prime PropHydroAro HydroCore PropSmall PropAro PropBasic PropAcidic DAI Stability Foreigness CelPrev PrioScore CYT HLAexp MCPmean'.split()
# response prediction_rf

# The following were already quantile-normalized and therefore not used: PRIME_rank,PRIME_BArank,mhcflurry_aff_percentile,mhcflurry_presentation_percentile
FEATS = 'MT_BindAff,BindStab,Quantification,Agretopicity,Score_EL,ln_NumTested'.split(',')

LISTOF_FEATURES = [SOFTS60+SOFTS+IMPROVE_FTS+FEATS]
LISTOF_LABELS = [['Label', 'response', 'VALIDATED']]
ASCENDING_FEATURES = ('MT_BindAff,Agretopicity,%Rank_EL,PRIME_rank,PRIME_BArank,mhcflurry_aff_percentile,mhcflurry_presentation_percentile,ln_NumTested'.split(',')
    + 'Expression Foreigness DAI NetMHCExp pI PropBasic Inst Stability PropAcidic RankEL PropSmall ln_NumTested RankBA'.split())

scriptdir = (os.path.dirname(os.path.realpath(__file__)))
parser = argparse.ArgumentParser(description='This script analyzes features (the features are typically the output of relevant software packages, such as kallisto, netMHCpan, mhcflurry, PRIME, ERGO, and netTCR). ', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

#TASKS = ['traintest', 'crossval', 'pairwise_traintest']
#parser.add_argument('--task', nargs='+', default=['traintest', 'crossval'], help='Task, can be any combination of {TASKS}')
#parser.add_argument('--train', nargs='+', default=[],
#        help='Training csv files from https://www.biorxiv.org/content/10.1101/2024.11.06.622261v1.supplementary-material')
#parser.add_argument('--test', default='',
#        help='Test csv files from https://www.biorxiv.org/content/10.1101/2024.11.06.622261v1.supplementary-material')

parser.add_argument('-i', '--input', nargs='+', default=[ scriptdir+'/media-2.csv', scriptdir+'/media-4.csv' ],
        help='String list of length 2, 4, 6, etc. denoting task1 file1 task2 file2 task3 file3 etc. \n'
        'Task can be tr (train), te (test, aka benchmark or score), cv (cross validate), or comma-separated string of any combination of tr, te, and cv (e.g., tr,te and tr,cv)\n'
        'File can be arbitray CSV or TSV file with the feature names given by the --features param. If this param is not provided, then one of the following types of file can be used: '
        '1 - the file used to train neoguider, '
        '2 - the validated-neoantigen file from https://github.com/SRHgroup/IMPROVE_paper/blob/main/data.zip, '
        '3 - the predictions on the viral and mutation datasets downloaded from https://www.biorxiv.org/content/10.1101/2024.11.06.622261v1.supplementary-material, '
        '4 - the predictions* results generated by https://github.com/SchubertLab/benchmark_TCRprediction. ')
parser.add_argument('-o', '--output', default=(scriptdir+'/tmp/default_out'),
        help='The prefix of the output files')
parser.add_argument('-I', '--isolib', default=scriptdir+'/../IsotonicLogisticRegression#IsotonicLogisticRegression',
        help='The NeoGuider feature transformation library file')
parser.add_argument('-1', '--ft_preproc_techs', nargs='+', default=[x for x in THE_FEAT_PREPROC_TECHS],
        help='Names of the feature preprocessing techniques to be assessed')
parser.add_argument('-2', '--classifiers', nargs='+', default=[x for x in THE_CLASSIFIERS],
        help='Names of the machine-learning classifiers to be assessed')
parser.add_argument('--inc', default=None, help='Assume that label as a function of each feature is increasing (0, 1, "auto", or None denoting false, true, auto, and inferred)')
parser.add_argument('--sep', default=None, help='csv column separator')
parser.add_argument('--seed', default=43, help='seed for random number generation')
parser.add_argument('--tasks', nargs='+', default=['fa1', 'fa2', 'fa3', 'hla1', 'hla2'], help='Feature-analysis and HLA-analysis tasks')
parser.add_argument('--features', nargs='+', default=[], help='Features analyzed, auto infer if not provided')
parser.add_argument('--label', default='', help='The label analyzed, auto infer if not provided')
parser.add_argument('-uf', '--untest_flag', default=0x0, type=int, help='If the 0x1, 0x2, and 0x4 bits are set, then treat NA label values as zeros for training, test, and cross-validation. ')
parser.add_argument('-pf', '--peplen_flag', default=0x0, type=int, help='If the 0x1, 0x2, and 0x4 bits are set, then remove peptides with lengths greater than 11 (with at least 12 amino acid residues) for training, test, and cross-validation. ')

args = parser.parse_args()

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
    if sum([(x in column_names) for x in SOFTS]) >= (1 + len(SOFTS)) // 2:
        increasing = True
    else:
        increasing = 'auto'
else:
    increasing = args.inc
if increasing in [True, False]: feat_pvalue_drop_irrelevant_feature = False
else: feat_pvalue_drop_irrelevant_feature = True
increasing = 'auto'
feat_pvalue_drop_irrelevant_feature = True

isopath = args.isolib.split('#')[0]
isolibname = args.isolib.split('#')[1]
ISO_DIR = os.path.dirname(isopath)
ISO_NAME = os.path.basename(isopath)
ISO_MODULE, ISO_EXT = os.path.splitext(ISO_NAME)
sys.path.append(ISO_DIR)
IsotonicLogisticRegression = __import__(ISO_MODULE, globals(), locals(), [isolibname], 0)
IsotonicLogisticRegression = IsotonicLogisticRegression.__dict__[isolibname]
nan_policy='raise' #'mean'
THE_FEAT_PREPROC_TECHS['NG']       = IsotonicLogisticRegression(increasing=increasing, random_state=0,
        feat_pvalue_drop_irrelevant_feature=feat_pvalue_drop_irrelevant_feature, nan_policy=nan_policy, excluded_cols=['ln_NumTested'])
THE_FEAT_PREPROC_TECHS['NG_withoutNumTested'] = IsotonicLogisticRegression(increasing=increasing, random_state=0,
        feat_pvalue_drop_irrelevant_feature=feat_pvalue_drop_irrelevant_feature, nan_policy=nan_policy)

#THE_FEAT_PREPROC_TECHS['NeoGuider(P<0.0001)'] = IsotonicLogisticRegression(increasing='auto', random_state=0, feat_pvalue_thres=0.0001, nan_policy=nan_policy, excluded_cols=['ln_NumTested'])
#THE_FEAT_PREPROC_TECHS['NeoGuider(P<0.001)']  = IsotonicLogisticRegression(increasing='auto', random_state=0, feat_pvalue_thres=0.001,  nan_policy=nan_policy, excluded_cols=['ln_NumTested'])
#THE_FEAT_PREPROC_TECHS['NeoGuider(P<0.01)']   = IsotonicLogisticRegression(increasing='auto', random_state=0, feat_pvalue_thres=0.01,   nan_policy=nan_policy, excluded_cols=['ln_NumTested'])
#THE_FEAT_PREPROC_TECHS['NeoGuider(P<0.10)']   = IsotonicLogisticRegression(increasing='auto', random_state=0, feat_pvalue_thres=0.10,   nan_policy=nan_policy, excluded_cols=['ln_NumTested'])
#THE_FEAT_PREPROC_TECHS['NeoGuider(P<=1.0)']   = IsotonicLogisticRegression(increasing='auto', random_state=0, feat_pvalue_thres=1.01,   nan_policy=nan_policy, excluded_cols=['ln_NumTested'])

try:
    sklearn.set_config(enable_metadata_routing=False)
except TypeError as err:
    pass

logger = logging.getLogger(__name__)
def config_logging(function_name=''): logging.basicConfig(level=logging.INFO, format=F'%(asctime)s %(pathname)s:%(lineno)d %(levelname)s {function_name} - %(message)s')
config_logging()

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

def compute_ranked_df(df, labelcol, patientcol='Patient', predcol='NG/LR'):
    df = df.sort_values(predcol, ascending=False)
    ranks = []
    patient2rank = collections.defaultdict(int)
    for patient in df[patientcol]:
        patient2rank[patient] += 1
        ranks.append(patient2rank[patient])
    df['rank'] = ranks
    return df

def analyze_performance_per_hla(df, hlacol, labelcol, figout, patientcol='Patient', predcol='NG/LR'):
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

def construct_ml_pipes(ft_preproc_tech_dict, classifier_dict):
    ret = []
    for ft_preproc_name, ft_preproc_tech, in sorted(ft_preproc_tech_dict.items()):
        for classifier_name, classifier in sorted(classifier_dict.items()):
            ml_pipename = comb(ft_preproc_name, classifier_name)
            ml_pipe = make_pipeline(copy.deepcopy(ft_preproc_tech), VarianceThreshold(), copy.deepcopy(classifier))
            ret.append((ml_pipename, ml_pipe))
    return ret

def assert_prob_arr(prob_pred):
    assert prob_pred.shape[1] == 2, F'The predicted result {prob_pred} does not have two columns denoting two columns!'
    for x, y in prob_pred:
        assert 0 <= x and x <= 1, F'The probability {x} must be between zero and one!'
        assert 0 <= y and y <= 1, F'The probability {y} must be between zero and one!'
        assert 1-1e-9 < (x + y) and (x + y) < 1+1e-9, F'The probabilities {x} and {y} do not sum to one!'

def drop_feat_from_X(ml_pipename, X):
    X = X.copy()
    for colname in X.columns:
        if colname in ASCENDING_FEATURES: 
            X.loc[:,colname] = -X[colname]
            logging.info(F'Performed negation to the column {colname} (CHECK_FOR_BUG)')
    if (not 'ln_NumTested' in X.columns):
        return X.copy()
    elif ('withoutNumTested'.lower() in ml_pipename.lower()) or (not 'neoguider' in ml_pipename.lower() and not ml_pipename.startswith('NG')): # and not 'NG_' in ml_pipename:
        return X.drop(columns=['ln_NumTested'])
    else:
        return X.copy()

def train_ml_pipe(ml_pipename, ml_pipe, X, y):
    config_logging('TRAIN')
    logging.info(F'Start training {ml_pipename}')
    X = drop_feat_from_X(ml_pipename, X)
    ml_pipe.fit(X, y)
    prob_pred = ml_pipe.predict_proba(X)
    assert_prob_arr(prob_pred)
    logging.info(F'End training {ml_pipename}')
    return (ml_pipename, ml_pipe, prob_pred[:,1])

def test_ml_pipe(ml_pipename, ml_pipe, X):
    config_logging('TEST')
    X = drop_feat_from_X(ml_pipename, X)
    prob_pred = ml_pipe.predict_proba(X)
    assert_prob_arr(prob_pred)
    return (ml_pipename, ml_pipe, prob_pred[:,1])

def cv_ml_pipe(ml_pipename, ml_pipe, X, y, partitions):
    config_logging('CROSS_VAL')
    random.seed(args.seed)
    np.random.seed(args.seed)

    cv = GroupKFold() #(random_state=0)
    X = drop_feat_from_X(ml_pipename, X)
    prob_pred = cross_val_predict(ml_pipe, X, y, groups=partitions, cv=cv, method='predict_proba')
    assert_prob_arr(prob_pred)
    return (ml_pipename, ml_pipe, prob_pred[:,1])

def compute_topN(df, labelcol, patientcol='Patient', predcol='NG/LR', topN=20):
    df = compute_ranked_df(df, labelcol, patientcol, predcol)
    return len([label for label in (df.loc[df['rank']<=topN,:][labelcol]) if label == 1])
'''
def compute_topN(y_true, y_pred, y_patient, topN):
    dfall = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred, 'y_patient': y_patient})
    dfall = dfall.sort_values('y_pred', ascending=False)
    df_iterable = dfall.groupby('y_patient')
    ret = {}
    for patient, df in df_iterable:
        ret[patient] = df.iloc[0:topN,:]['y_true'].sum()
    return sum(ret.values()), ret
'''

from matplotlib.gridspec import GridSpec
def build_auc_df(df_ins, out_fname_fmt, ft_preproc_techs, classifiers, features, labelcol, colname2rocauc_list=[{}], metric_name='roc_auc', metric_vals=[0], titles=[''], barh_fmt='%.4g'):
    n_subfigs = max((len(df_ins), len(colname2rocauc_list), len(metric_vals), len(titles)))
    assert len(df_ins) in [1, n_subfigs], F'Found {len(df_ins)} df_ins but only 1 and {n_subfigs} are allowed for generating {out_fname_fmt}!'
    assert len(colname2rocauc_list) in [1, n_subfigs], F'Found {len(colname2rocauc_list)} colname2rocauc_list but only 1 and {n_subfigs} are allowed for generating {out_fname_fmt}!'
    assert len(metric_vals) in [1, n_subfigs], F'Found {len(metric_vals)} colname2rocauc_list but only 1 and {n_subfigs} are allowed for generating {out_fname_fmt}!'
    assert len(titles) == n_subfigs, F'Please provide {n_subfigs} titles (current titles are: {titles}) to generate {out_fname_fmt}!'
    if len(df_ins) < n_subfigs: df_ins = [df_ins[0]] * n_subfigs
    if len(colname2rocauc_list) < n_subfigs: colname2rocauc_list = [colname2rocauc_list[0]] * n_subfigs
    if len(metric_vals) < n_subfigs: metric_vals = [metric_vals[0]] * n_subfigs
    #if len(titles) < n_subfigs:

    fig_1, ax_1 = plt.subplots(figsize=(8*n_subfigs, 8*2.5))
    ax_1.set_axis_off()
    gs = GridSpec(2, n_subfigs, height_ratios=[1, 25])
    legend_ax = fig_1.add_subplot(gs[0,:])
    legend_ax.set_axis_off()
    axes = [fig_1.add_subplot(gs[1,j]) for j in range(n_subfigs)]
    for ax_idx, (df_in, colname2rocauc, metric_val, title) in enumerate(zip(df_ins, colname2rocauc_list, metric_vals, titles)):
        auc_series = pd.Series(np.nan, features)
        auc_df = pd.DataFrame(data=np.nan,
                index   = [ft_preproc_name for ft_preproc_name, ft_preproc_tech in ft_preproc_techs.items()],
                columns = [classifier_name for classifier_name, classifier in classifiers.items()])
        auc_std_df = pd.DataFrame(auc_df)
        colnames = features + [comb(ft_preproc_name, classifier_name) for ft_preproc_name, ft_preproc_tech in ft_preproc_techs.items() for classifier_name, classifier in classifiers.items()]

        rows = []
        for colname in colnames:
            if colname not in df_in.columns: continue
            if colname in colname2rocauc:
                roc_auc = np.mean(colname2rocauc[colname])
                roc_auc_std = np.std(colname2rocauc[colname])
            else:
                logging.info(F'Computing the ROC_AUC of {colname}')
                #y_true = np.where(df_in[labelcol], 1 , 0)
                if metric_name == 'top':
                    roc_auc = compute_topN(df_in, labelcol, patientcol='Patient', predcol=colname, topN=metric_val)
                    #roc_auc, _ = compute_topN(y_true, df_in[colname], df_in['Patient'], metric_val)
                else:
                    roc_auc = roc_auc_score(df_in[labelcol], df_in[colname])
                #fpr, tpr, thresholds = metrics.roc_curve(train_df['response'], train_df[clfname], pos_label=1)
                #auc_df.loc[ft_preproc_name,classifier_name] = metrics.auc(fpr, tpr)
                roc_auc_std = np.nan
            rows.append((colname, roc_auc))
            if colname in features:
                auc_series[colname] = roc_auc
            else:
                ft_preproc_name, classifier_name = decomb(colname)
                auc_df.loc[ft_preproc_name, classifier_name] = roc_auc
                auc_std_df.loc[ft_preproc_name, classifier_name] = roc_auc_std
        long_df = pd.DataFrame(rows, columns=['Method', 'AUROC'])
        long_df.to_csv(out_fname_fmt.format('with_both'), sep='\t', index=True)
        auc_series.to_csv(out_fname_fmt.format('with_raw_features'), sep='\t', index=True)
        auc_df.to_csv(out_fname_fmt.format('with_featproc_clf_combs'), sep='\t', index=True, index_label='FeatPreprocessors\\Classifiers')
        auc_std_df.to_csv(out_fname_fmt.format('with_featproc_clf_combs_std'), sep='\t', index=True, index_label='FeatPreprocessors\\Classifiers')

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
        #auroc_method_class_list = zip(long_df['AUROC'], long_df['Method'], long_df['Method'].apply(lambda x: (-1 if 'neoguider' in x.lower() else (1 if x in features else 0))))
        long_df['MethClass'] = long_df['Method'].apply(lambda x: (-1 if (x.startswith('NG') or x.lower().startswith('neoguider')) else (1 if x in features else 0)))
        long_df = long_df.sort_values(by='AUROC')
        long_df['ypos'] = list(range(len(long_df)))
        methclass_df_iterable = long_df.groupby('MethClass')
        methclass2desc = {-1: 'Use NeoGuider (NG)', 0: 'Use other techniques', 1: 'Prioritize with a single feature'}
        hbars_list = []
        for methclass, df in sorted(methclass_df_iterable):
            hbars = ax.barh(df['ypos'], df['AUROC'], align='center', label=methclass2desc[methclass])
            ax.bar_label(hbars, fmt=barh_fmt, padding=2)
            hbars_list.append(hbars)
        ax.set_yticks(long_df['ypos'], labels=long_df['Method'])
        xmin, xmax = np.min(long_df['AUROC']), np.max(long_df['AUROC'])
        ax.set_xlim(xmin - (xmax - xmin) * 0.2, xmax + (xmax - xmin) * 0.2)
        ax.set_title(titles[ax_idx])
        #ax.legend(fontsize=14)
        def get_ncols(n_labels, n_cols):
            n_cols = int(round(min((n_cols, n_labels))))
            while n_labels % n_cols != 0: n_cols -= 1
            return n_cols
        if ax_idx == 0: legend_ax.legend(hbars_list, [v for k,v in sorted(methclass2desc.items())], title='Feature-preprocessing techniques',
                ncol=get_ncols(len(methclass2desc), n_subfigs),
                loc='center', fontsize=14, title_fontsize=16)

    plt.tight_layout()
    plt.savefig(out_fname_fmt.format('with_both')+'.pdf')
    plt.savefig(out_fname_fmt.format('with_both')+'.png', dpi=600)
    plt.close()

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
    patientcol = match_col(ret, ['Patient', 'PatientID'])
    if patientcol: # and 'ln_NumTested' not in ret.columns:
        patient2ntested = collections.defaultdict(int)
        for patient, label in zip(ret[patientcol], ret[labelcol]):
            patient2ntested[patient] += (1 if label in [0, 1] else 0)
        newcol = [np.log(max((1, patient2ntested[p]))) for p in ret[patientcol]]
        if 'ln_NumTested' not in ret.columns:
            ret['ln_NumTested'] = newcol
            added_feats.append('ln_NumTested')
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

def get_filenames(filepaths):
    return [x.split('/')[-1].split('.')[0] for x in filepaths]

def train_test_cv(train_fnames, test_fnames, cv_fnames, output, ft_preproc_techs, classifiers, csvsep, tasks, feature_names, label_name, untest_flag, peplen_flag):
    untest_ops_training_examples = ('zero' if (untest_flag & 0x1) else 'drop')
    untest_ops_test_examples = ('zero' if (untest_flag & 0x2) else 'drop')
    untest_ops_cv_examples = ('zero' if (untest_flag & 0x4) else 'drop')
    
    peplen_max_training_examples = (11 if (peplen_flag & 0x1) else 9999)
    peplen_max_test_examples = (11 if (peplen_flag & 0x2) else 9999)
    peplen_max_cv_examples = (11 if (peplen_flag & 0x4) else 9999)
    
    HLA_COLS= ['HLA_type', 'HLA_type_y', 'HLA_allele'] # HLA_type_x can contain comma
    # setup
    ft_preproc_techs = {x: THE_FEAT_PREPROC_TECHS[x] for x in ft_preproc_techs}
    classifiers = {x: THE_CLASSIFIERS[x] for x in classifiers}
    features_superset1 = (LISTOF_FEATURES[0] if len(feature_names) == 0 else feature_names.split(','))
    labels_superset1 = (LISTOF_LABELS[0] if label_name == '' else label_name)
    labelcol = None
    hlacol = ''
    in_dfs = []
    for i, train_fname in enumerate(train_fnames):
        in_df = pd.read_csv(train_fname, sep=csvsep)
        if i == 0:
            features = [colname for colname in in_df.columns if colname in features_superset1]
            labels = [colname for colname in in_df.columns if colname in labels_superset1]
            hlacols = [colname for colname in in_df.columns if colname in HLA_COLS]
            assert len(labels) == 1
            assert len(hlacols) <= 1, F'Found multiple HLA column names: {hlas}'
            labelcol = labels[0]
            if hlacols: hlacol = hlacols[0]
            in_df, added_feats = prepare_df(in_df, labelcol, na_op=untest_ops_training_examples, max_peplen=peplen_max_training_examples)
            if added_feats: features.extend(added_feats)
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

    ft_preproc_tech = ft_preproc_techs['NG']
    #train_X = QuantileTransformer(random_state=0).fit_transform(train_X)
    #train_X = pd.DataFrame(train_X, columns=features)
    big_transformed_X = ft_preproc_tech.fit_transform(train_X, big_y)
    ft_preproc_tech_feature_names = ft_preproc_tech.get_feature_names()
    ft_preproc_tech_feature_importances_1 = ft_preproc_tech.get_feature_importances('f2l')
    ft_preproc_tech_feature_importances_2 = ft_preproc_tech.get_feature_importances('f2f')
    ft_preproc_tech_feature_importances_3 = ft_preproc_tech.get_feature_importances('f2l2f')

    ft_preproc_tech_feature_importances_p1 = ft_preproc_tech.get_feature_importances('pvalue', 'mannwhitneyu')
    ft_preproc_tech_feature_importances_p2 = ft_preproc_tech.get_feature_importances('pvalue', 'spearmanr')
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
            'pvalue_mannwhitneyu': ft_preproc_tech_feature_importances_p1,
            'pvalue_spearmanr': ft_preproc_tech_feature_importances_p2,
            'pvalue_odds_spearmanr': ft_preproc_tech_feature_importances_p3,
            'statistic_mannwhitneyu': ft_preproc_tech_feature_importances_s1,
            'statistic_spearmanr': ft_preproc_tech_feature_importances_s2,
            'statistic_odds_spearmanr': ft_preproc_tech_feature_importances_s3,
            'trend_mannwhitneyu': ft_preproc_tech_feature_importances_t1,
            'trend_spearmanr': ft_preproc_tech_feature_importances_t2,
            'trend_odds_spearmanr': ft_preproc_tech_feature_importances_t3,
    })
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
        big_trans_df0 = big_transformed_df.loc[big_transformed_df[labelcol]==0,:] #.sample(n=100, random_state=0)
        big_trans_df1 = big_transformed_df.loc[big_transformed_df[labelcol]==1,:] #.sample(n=100, random_state=0)

        big_trans1_df0 = big_trans_df0.loc[:,features] #.sample(n=100, random_state=0)
        big_trans1_df1 = big_trans_df1.loc[:,features] #.sample(n=100, random_state=0)

        dfsize = min((len(big_trans_df0), len(big_trans_df1)))
        logging.info(F'Min_nrows={dfsize}')
        dfsize = min((dfsize, 100))
        logging.info(F'Start plotting all neoepitope candidates')
        plot_ret = pairplot_showing_pretrans_feat_vals(big_trans1_df0.sample(n=dfsize, random_state=0), big_trans1_df1.sample(n=dfsize, random_state=0), ilr)
        logging.info(F'Mid plotting all neoepitope candidates')
        #sns.pairplot(pd.concat([big_trans_df0.sample(n=dfsize, random_state=0), big_trans_df1.sample(n=dfsize, random_state=0)]), hue=labelcol)
        plt.savefig(f'{output}_pairwiseLogOdds.pdf')
        plt.close()
        logging.info(F'End plotting all neoepitope candidates')
        if 'mhcflurry_presentation_percentile' in big_transformed_df.columns:
            big_trans2_df0 = big_trans_df0.loc[big_trans_df0['mhcflurry_presentation_percentile']<=5,features] #big_transformed_df.loc[big_transformed_df[labelcol]==0,:].sample(n=100, random_state=0)
            big_trans2_df1 = big_trans_df1.loc[big_trans_df1['mhcflurry_presentation_percentile']<=5,features]
            dfsize = min((len(big_trans2_df0), len(big_trans2_df1)))
            logging.info(F'Min_nrows={dfsize}')
            dfsize = min((dfsize, 100))
            dfsize_0 = min((len(big_trans2_df0), 1000))
            #plot_ret = sns.pairplot(pd.concat([big_trans2_df0.sample(n=dfsize_0, random_state=0), big_trans2_df1.sample(n=dfsize, random_state=0)]), hue=labelcol)
            plot_ret = pairplot_showing_pretrans_feat_vals(big_trans2_df0.sample(n=dfsize_0, random_state=0), big_trans2_df1.sample(n=dfsize, random_state=0), ilr)
            plt.savefig(f'{output}_pairwiseLogOdds_mhcflurry_presentation_5perc.pdf')
            plt.close()

    ml_pipes = construct_ml_pipes(ft_preproc_techs, classifiers)

    # train phase
    train_X = train_df.loc[:, features].copy()
    train_y = train_df.loc[:, labelcol].copy()
    train_X = train_X.fillna({col : np.mean(train_X[col]) for col in features})

    logging.info(F'Start training')
    train_results = Parallel(n_jobs=24)(delayed(train_ml_pipe)(ml_pipename, ml_pipe, train_X, train_y) for ml_pipename, ml_pipe in ml_pipes)
    logging.info(F'End training')
    for result in train_results:
        ml_pipename, ml_pipe, ml_pipe_predicted = result
        train_df[ml_pipename] = ml_pipe_predicted
    train_df.to_csv(f'{output}_train.csv', sep=',', index=None)

    test_dfs = []
    for fidx, test_fname in enumerate(test_fnames):
        fidx += 1
        if test_fname in train_fnames: train_or_test = 'train'
        else: train_or_test = 'test'
        df = pd.read_csv(test_fname, sep=csvsep)
        df, added_feats = prepare_df(df, labelcol, na_op=untest_ops_test_examples, max_peplen=peplen_max_test_examples)
        dfXy = df.loc[:,features + [labelcol]]
        #assert (train_df.columns == test_df.columns).all(), F'{train_df.columns} == {test_df.columns} failed for the column names of the inputs {train_fnames} and {test_fname}'
        # test phase
        X = dfXy.loc[:, features].copy()
        X = X.fillna({col : np.mean(X[col]) for col in features})
        test_results = Parallel(n_jobs=24)(delayed(test_ml_pipe)(ml_pipename, ml_pipe, X) for ml_pipename, ml_pipe, _, in train_results)
        for result in test_results:
            ml_pipename, ml_pipe, ml_pipe_predicted = result
            assert not np.isnan(ml_pipe_predicted).any()
            df[ml_pipename] = ml_pipe_predicted
        df.to_csv(F'{output}_{fidx}_test.csv', sep=',', index=None)
        df2 = df.fillna({col : np.mean(df[col]) for col in features})
        test_dfs.append(df2)
        if 'Patient' in df2.columns:
            build_auc_df([df2], F'{output}_{fidx}_{train_or_test}_'+'topN_{}.tsv', ft_preproc_techs, classifiers, features, labelcol, [{}], metric_name='top', metric_vals=[20,50,100], titles=['Top-20 #True', 'Top-50 #True', 'Top-100 #True'])
        if 'hla1' in tasks: analyze_hla(df2, hlacol, labelcol, F'{output}_{fidx}_{train_or_test}_hla_stats.pdf')
        if 'hla2' in tasks:
            logging.info(F'start analyze_performance_per_hla({df}, {hlacol}, {labelcol}, `_{fidx}_{train_or_test}_hla_bench.pdf`)')
            analyze_performance_per_hla(df, hlacol, labelcol, F'{output}_{fidx}_{train_or_test}_hla_bench.pdf')
            logging.info(F'end analyze_performance_per_hla({df}, {hlacol}, {labelcol}, `_{fidx}_{train_or_test}_hla_bench.pdf`)')
    if test_dfs:
        build_auc_df(test_dfs, F'{output}_0_{train_or_test}_roc_auc_{{}}.tsv', ft_preproc_techs, classifiers, features, labelcol, [{}], titles=get_filenames(test_fnames))

    cv_pred_dfs = []
    pipename2score_list = []
    for fidx, fname in enumerate(cv_fnames):
        fidx += 1
        in_df = pd.read_csv(fname, sep=csvsep)

        features = [colname for colname in in_df.columns if colname in features_superset1]
        labels = [colname for colname in in_df.columns if colname in labels_superset1]
        hlacols = [colname for colname in in_df.columns if colname in HLA_COLS]
        assert len(labels) == 1
        assert len(hlacols) <= 1, F'Found multiple HLA column names: {hlas}'
        labelcol = labels[0]
        if hlacols: hlacol = hlacols[0]
        
        df, added_feats = prepare_df(in_df, labelcol, na_op=untest_ops_cv_examples, max_peplen=peplen_max_cv_examples)

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

        results = Parallel(n_jobs=24)(delayed(cv_ml_pipe)(ml_pipename, ml_pipe, X, y, df[partition_name]) for ml_pipename, ml_pipe in ml_pipes)
        for result in results:
            ml_pipename, ml_pipe, ml_pipe_predicted = result
            df[ml_pipename] = ml_pipe_predicted
        df.to_csv(F'{output}_{fidx}_'+'cv.csv', sep=',', index=None)
        df2 = df.fillna({col : np.mean(df[col]) for col in features})
        cv_pred_dfs.append(df2)
        results = Parallel(n_jobs=24)(delayed(cross_val_score)(ml_pipe, drop_feat_from_X(ml_pipename, X), y, groups=df[partition_name], cv=GroupKFold(), scoring='roc_auc', n_jobs=-1)
            for ml_pipename, ml_pipe in ml_pipes)
        pipename2score = {ml_pipename : results[i] for i, (ml_pipename, ml_pipe) in enumerate(ml_pipes)}
        pipename2score_list.append(pipename2score)
    if cv_fnames:
        build_auc_df(cv_pred_dfs, F'{output}_0_'+'cv_predict_roc_auc_{}.tsv', ft_preproc_techs, classifiers, features_superset1, labelcol, [{}], titles=get_filenames(cv_fnames))
        build_auc_df(cv_pred_dfs, F'{output}_0_'+'cv_score_roc_auc_{}.tsv', ft_preproc_techs, classifiers, features_superset1, labelcol, pipename2score_list, titles=get_filenames(cv_fnames))

if __name__ == '__main__':
    tr_filenames = [filename for i, filename in enumerate(args.input) if ((i % 2 == 1) and 'tr' in args.input[i-1].split(','))]
    te_filenames = [filename for i, filename in enumerate(args.input) if ((i % 2 == 1) and 'te' in args.input[i-1].split(','))]
    cv_filenames = [filename for i, filename in enumerate(args.input) if ((i % 2 == 1) and 'cv' in args.input[i-1].split(','))]
    print(F'tr_files={tr_filenames} te_files={te_filenames} cv_files={cv_filenames}')
    with open(args.output + '.info', 'w') as infofile:
        infofile.write('\t'.join(sys.argv) + '\n')
        infofile.write(str(args))
        infofile.write(F'Train: {tr_filenames}')
        infofile.write(F'Benchmark (test): {tr_filenames}')
        infofile.write(F'CrossValidate: {cv_filenames}')
    train_test_cv(tr_filenames, te_filenames, cv_filenames, args.output, args.ft_preproc_techs, args.classifiers, csvsep, args.tasks, args.features, args.label, args.untest_flag, args.peplen_flag)

