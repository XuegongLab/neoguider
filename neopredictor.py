import argparse, copy, json, logging, os, pickle, pprint, random, sys

from joblib import Parallel, delayed # multiprocessing can hang if the virtual memory allocated is too big

import numpy as np
import pandas as pd

from collections import defaultdict, namedtuple

from scipy import stats

from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import ColumnTransformer
# Modified from https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis)
from sklearn.ensemble import (AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier,)
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression
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

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from IsotonicLogisticRegression import IsotonicLogisticRegression

random.seed(0)
np.random.seed(0)

BIG_INT = 2**30

# https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html
SCALERS = {    
    'Identity'            : ColumnTransformer([], remainder='passthrough'),
    'MaxAbsScaler'        : MaxAbsScaler(),
    'MinMaxScaler'        : MinMaxScaler(),
    'Normalizer'          : Normalizer(),
    'PowerTransformer'    : PowerTransformer(),
    'QuantileTransformer' : QuantileTransformer(random_state=0),    
    'RobustScaler'        : RobustScaler(),
    'StandardScaler'      : StandardScaler(),
    'StandardTransformer' : QuantileTransformer(random_state=0, output_distribution='normal'),
    '_KDEIsoTransformer00' : IsotonicLogisticRegression(random_state=0, fit_data_clear=True, 
        fit_add_measure_error=False, transform_add_measure_error=False, ft_fit_add_measure_error=False, ft_transform_add_measure_error=False),
    '_KDEIsoTransformer01' : IsotonicLogisticRegression(random_state=0, fit_data_clear=True, 
        fit_add_measure_error=False, transform_add_measure_error=False, ft_fit_add_measure_error=False, ft_transform_add_measure_error=True ),
    '_KDEIsoTransformer10' : IsotonicLogisticRegression(random_state=0, fit_data_clear=True, 
        fit_add_measure_error=False, transform_add_measure_error=True,  ft_fit_add_measure_error=False, ft_transform_add_measure_error=False),
    '_KDEIsoTransformer11' : IsotonicLogisticRegression(random_state=0, fit_data_clear=True, 
        fit_add_measure_error=False, transform_add_measure_error=True,  ft_fit_add_measure_error=False,  ft_transform_add_measure_error=True),
}

# https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
PREDICTORS = {
    'KN': KNeighborsClassifier(),
    # 'KN500': KNeighborsClassifier(n_neighbors=500), # Too many ties in probabilities (density of positive examples is too low)
    # 'SVC': SVC(probability=True), # sklearn.svm import SVC is not designed to predict probability and is not designed to handle large sample size (O(n*n) runtime, intractable)
    # 'GP': GaussianProcessClassifier(random_state=0), # Error: numpy.core._exceptions._ArrayMemoryError: Unable to allocate 1.25 TiB for ...
    'DT': DecisionTreeClassifier(random_state=0),
    'RF': RandomForestClassifier(random_state=0),
    'MLP': MLPClassifier(random_state=0),
    'AB': AdaBoostClassifier(random_state=0),
    'GNB': GaussianNB(),
    'LDA': LinearDiscriminantAnalysis(),
    'QDA': QuadraticDiscriminantAnalysis(),
    
    'LR' : LogisticRegression(random_state=0),
    # Tree-based algorithms other than the ones used above:
    'eET': ExtraTreesClassifier(random_state=0),
    'eGB': GradientBoostingClassifier(random_state=0), # GB is similar to XGB
    'eXGB': XGBClassifier(random_state=0),
}

logger = logging.getLogger(__name__)
logging.basicConfig(format=('neopredictor %(asctime)s - %(message)s'), level=logging.DEBUG)

def u2d(s): return '--' + s.replace('_', '-')
def isna(arg): return arg in [None, '', 'NA', 'Na', 'N/A', 'None', 'none', '.']
def nan_replace(v, w=0): return (w if np.isnan(v) else v)

def col2last(df, colname): return (df.insert(len(df.columns)-1, colname, df.pop(colname)) if colname in df.columns else -1)
def dropcols(df, colnames):
    xs = [x for x in colnames if x in df.columns]
    df.drop(xs, axis = 1)

def aaseq2canonical(aaseq): return aaseq.upper().replace('U', 'X').replace('O', 'X')

# Foreignness : min at 1e-16
#   http://book.bionumbers.org/how-many-chromosome-replications-occur-per-generation/
#   https://en.wikipedia.org/wiki/Cell_division#cite_note-8
# Note: Probability of immunogenicity is not spearmanR-correlated with Foreignness, so Foreignness is not used. 

def safediv(a, b, defaultval=np.nan): return ((a/b) if (b != 0) else defaultval)

def assess_top20_top50_top100_ttif_fr_auprc(df):
    df0 = df.loc[df['VALIDATED'] >= 0,:].copy()
    df1 = df.loc[df['VALIDATED'] == 1,:].copy()
    ranks = sorted(list(df1['Rank']))
    top20  = len([r for r in ranks if (r <= 20)])
    top50  = len([r for r in ranks if (r <= 50)])
    top100 = len([r for r in ranks if (r <= 100)])
    ttif = safediv(top20, len(df0.loc[df0['Rank'] <= 20, :]))
    fr = safediv(top100, len(ranks))
    auroc = (roc_auc_score(df0['VALIDATED'], -df0['Rank']) if (set(df0['VALIDATED']) == set([0, 1])) else np.nan)
    auprc = (average_precision_score(df0['VALIDATED'], -df0['Rank']) if (set(df0['VALIDATED']) == set([0, 1])) else np.nan)
    PerformanceResult = namedtuple("PerformanceResult", "top20 top50 top100 TTIF FR AUPRC TFA_mean AUROC")
    return PerformanceResult(top20, top50, top100, ttif, fr, auprc, np.mean([ttif, fr, auprc]), auroc)

def filtdf(df, peplens):
    if not ('PepTrace' in df.columns): return df
    df = df.loc[~pd.isna(df['PepTrace']),:].copy()
    df['Agretopicity'] = df['Agretopicity'].fillna(1e-9)
    MT_pep_col = ('MT_pep' if 'MT_pep' in df.columns else 'MT_pep_x')
    is_kept = df[MT_pep_col].str.len().isin(peplens)
    df = df.loc[is_kept,:]
    return df

def filter_testdf(df, additonal_vars):
    if not ('PepTrace' in df.columns): return df
    is_SNV_detected_from_RNA = df['Identity'].str.startswith('SNV_R')
    is_FSV_detected_from_RNA = df['Identity'].str.startswith('FSV_R')
    is_INS_detected_from_RNA = df['Identity'].str.startswith('INS_R')
    is_DEL_detected_from_RNA = df['Identity'].str.startswith('DEL_R')
    is_FUS = (df['Identity'].str.startswith('FU_') | df['Identity'].str.startswith('FUS_'))
    is_SP  = (df['Identity'].str.startswith('SP_') | df['Identity'].str.startswith('SPL_'))
    is_discarded = (
              (is_SNV_detected_from_RNA & (not 'rna_snv' in additonal_vars.split(',')))
            | (is_FSV_detected_from_RNA & (not 'rna_fsv' in additonal_vars.split(','))) 
            | (is_INS_detected_from_RNA & (not 'rna_indel' in additonal_vars.split(','))) 
            | (is_DEL_detected_from_RNA & (not 'rna_indel' in additonal_vars.split(',')))
            | (is_FUS & (not 'fusion' in additonal_vars.split(','))) 
            | (is_SP & (not 'splicing' in additonal_vars.split(',')))
    )
    df = df.loc[~is_discarded, :]
    return df

def compute_are_in_cum(df):
    if 'VALIDATED' in df.columns: 
        are_in_cum = np.where(df['VALIDATED'] >= 0, 1, 0)
    else:
        # are_in_cum = np.where(df['%Rank_EL'] < 2.0, 1, 0) # this is supposed to be better but is not common practice yet
        are_in_cum = np.where(df['MT_BindAff'] < 500.0, 1, 0)
    #df['InTested_RankEL_LT0.5_frac'] = ((sum(np.where(df['%Rank_EL'] < 0.5, 1, 0) * are_in_cum) / sum(are_in_cum)) if sum(are_in_cum) else -1)
    #df['InTested_RankEL_LT2.0_frac'] = ((sum(np.where(df['%Rank_EL'] < 2.0, 1, 0) * are_in_cum) / sum(are_in_cum)) if sum(are_in_cum) else -1)
    df['ln_NumTested'] = (np.log(sum(are_in_cum)) if sum(are_in_cum) else 0)
    #for idx, selector in enumerate([[1] * len(df), are_in_cum]):
    #    res = stats.spearmanr(
    #        [x for (x,y) in zip(df['%Rank_EL'],     selector) if y],
    #        [x for (x,y) in zip(df['Agretopicity'], selector) if y])
    #    df[F'F_{idx}_RankEL_agr_SPEARMAN_sgn_inv_P'] = nan_replace(np.sign(res.statistic) / max((1e-100, res.pvalue)), 0)
    #    df[F'F_{idx}_RankEL_agr_SPEARMAN_statistic'] = nan_replace(res.statistic, 0)
    return df, are_in_cum

def applyF(arr, f=sum):
    if len(arr): return f(arr)
    else: return np.nan

def patientwise_predict_test(tuple_arg):
    ilrs, pipeline_names, pipelines, infile, suffix, peplens, baseline, listof_features = tuple_arg
    logger = logging.getLogger('patientwise_predict_test')
    outpref = infile + '.' + suffix
    logger.info(F'InFile={infile} OutFile={outpref}')

def patientwise_predict(tuple_arg):
    ilrs, pipeline_names, pipelines, infile, suffix, peplens, baseline, listof_features, additonal_vars = tuple_arg
    logger = logging.getLogger('Test')
    logging.basicConfig(format=(F'neo-test %(asctime)s - %(message)s'), 
        level=logging.DEBUG)

    outpref = infile + '.' + suffix
    logger.info(F'START: infile={infile}')
    df1 = pd.read_csv(infile, sep='\t')
    df1 = filtdf(df1, peplens)
    df1 = filter_testdf(df1, additonal_vars)
    if len(df1) == 0:
        logger.warning(F'Skipping file={infile} because it is empty after filtering. ')
        return -1
    
    df1, are_in_cum = compute_are_in_cum(df1)
    if ('method' in baseline.split(',')):
        evalres3 = {}
        evaldict = defaultdict(list)
        for i, (pipeline_name, pipeline) in enumerate(zip(pipeline_names, pipelines)):        
            pipedf = df1.copy()
            if len(pipedf) > 50*1000: logger.info(F'pipe={pipeline_name:30} {(i+1):3}/{len(pipelines)} {infile}')
            y = pipeline.predict_proba(pipedf.loc[:, listof_features[0]])
            pipedf['PredictedProbWithOtherMethod'] = [v[1] for v in y]
            pipedf['Rank'] = pipedf['PredictedProbWithOtherMethod'].rank(method='first', ascending=False)
            pipedf['Rank'] = pipedf['Rank'].fillna(BIG_INT)
            pipedf = pipedf.sort_values('Rank')
            pipedf = pipedf.astype({"Rank":int})
            pipedf['ML_pipeline'] = pipeline_name
            other_pred = F'{outpref}.other_method_{(i+1):03d}.baseline'
            #pipedf.to_csv(other_pred, sep='\t', header=1, index=0, na_rep='NA')
            pipedf.iloc[range(min((len(pipedf),1000))),:].to_csv(other_pred + '.top1000.gz', sep='\t', header=1, index=0, na_rep='NA', compression='gzip')

            if 'VALIDATED' in pipedf.columns:
                evalres = assess_top20_top50_top100_ttif_fr_auprc(pipedf)
                evalres2 = evalres._asdict()
                evalres3[pipeline_name] = evalres2
                
                transform_name, ml_model_name = pipeline_name.split('/')
                evaldict['infile'].append(infile)
                evaldict['feature_transform'].append(transform_name)
                evaldict['ML_model_name'].append(ml_model_name)
                for k, v in copy.deepcopy(evalres._asdict()).items():
                    evaldict[k].append(v)
            del pipedf
        with open(outpref + '.methods.performance', 'w') as file:
            json.dump(evalres3, file, indent=2)
        with open(outpref + '.methods.performance.tsv', 'w') as file:
            eval_df = pd.DataFrame.from_dict(evaldict)
            eval_df.to_csv(file, sep='\t', header=1, index=0, na_rep='NA')
    
    df = df1
    
    # These features have been tested for calibrating probabilities, and none of them performs better than ln_NumTested.
    '''
    for idx, selector in enumerate([np.array([1] * len(df)), are_in_cum]):
        res = stats.spearmanr(
            [x for (x,y) in zip(df['%Rank_EL'],       selector) if y], 
            [x for (x,y) in zip(df['Quantification'], selector) if y])
        df[F'{idx}_RankEL_TPM_SPEARMAN_sgnP'] = np.sign(res.statistic) * res.pvalue
        df[F'{idx}_RankEL_TPM_SPEARMAN_stat'] = res.statistic
        foreign_selector = np.where(((df['%Rank_EL'] < 0.5) & (df['Foreignness'] > 1e-50)), 1, 0) * selector
        agretop_selector = np.where(((df['%Rank_EL'] < 0.5) & (df['Agretopicity'] < 1e-1)), 1, 0) * selector
        df[F'{idx}_RankEL_LT0.5/TCRP_GT0E/medTPM'] = np.median(list(filter(lambda x:(x>0), foreign_selector * (df['Quantification'] + 1e-50))))
        df[F'{idx}_RankEL_LT0.5/agr_LT0.1/medTPM'] = np.median(list(filter(lambda x:(x>0), agretop_selector * (df['Quantification'] + 1e-50))))
        df[F'{idx}_RankEL_LT0.5/medTPM']  = np.median(list(filter(lambda x:(x>0), np.where(df['%Rank_EL'] < 0.5, 1, 0) * selector * (df['Quantification'] + 1e-50))))
        df[F'{idx}_RankEL_LT2.0/medTPM']  = np.median(list(filter(lambda x:(x>0), np.where(df['%Rank_EL'] < 2.0, 1, 0) * selector * (df['Quantification'] + 1e-50))))
        df[F'{idx}_RankEL_LT100/medTPM']  = np.median(list(filter(lambda x:(x>0), np.where(df['%Rank_EL'] < 100, 1, 0) * selector * (df['Quantification'] + 1e-50))))
        #df['RankEL_LT0.5_totMedTPM']  = np.median(list(filter(lambda x:(x>0), np.where(df['%Rank_EL'] < 0.5, 1, 0)              * (df['Quantification'] + 1e-50))))
        #df['RankEL_LT2.0_totMedTPM']  = np.median(list(filter(lambda x:(x>0), np.where(df['%Rank_EL'] < 2.0, 1, 0)              * (df['Quantification'] + 1e-50))))
        #df['RankEL_LT100_totMedTPM']  = np.median(list(filter(lambda x:(x>0), np.where(df['%Rank_EL'] < 100, 1, 0)              * (df['Quantification'] + 1e-50))))

        n_foreign_peps = applyF(np.where(((df['%Rank_EL'] < 0.5) & (df['Foreignness'] > 1e-50)), 1, 0) * selector)
        n_agretop_peps = applyF(np.where(((df['%Rank_EL'] < 0.5) & (df['Agretopicity'] < 0.1)), 1, 0) * selector)
        df[F'{idx}_RankEL_LT0.1_N'] = sum(np.where(df['%Rank_EL'] < 0.1, 1, 0) * selector)
        n_all_peps = \
        df[F'{idx}_RankEL_LT0.5_N'] = sum(np.where(df['%Rank_EL'] < 0.5, 1, 0) * selector)
        df[F'{idx}_RankEL_LT2.0_N'] = sum(np.where(df['%Rank_EL'] < 2.0, 1, 0) * selector)
        df[F'{idx}_RankEL_LT6.0_N'] = sum(np.where(df['%Rank_EL'] < 6.0, 1, 0) * selector)
        
        df[F'{idx}_RankEL_LT0.5/agr_LT0.1_N'] = n_agretop_peps
        df[F'{idx}_RankEL_LT0.5/TCRP_GT0E_N'] = n_foreign_peps
        
        df[F'{idx}_RankEL_LT0.5_F'] = sum(np.where(df['%Rank_EL'] < 0.5, 1, 0) * selector) / sum(selector)
        df[F'{idx}_RankEL_LT2.0_F'] = sum(np.where(df['%Rank_EL'] < 2.0, 1, 0) * selector) / len(selector)
        df[F'{idx}_RankEL_LT0.5/agr_LT0.1_F'] = n_agretop_peps / n_all_peps
        df[F'{idx}_RankEL_LT0.5/TCRP_GT0E_F'] = n_foreign_peps / n_all_peps
        #df['RankEL_LT0.5_totN'] = sum(np.where(df['%Rank_EL'] < 0.5, 1, 0))
        #df['RankEL_LT2.0_totN'] = sum(np.where(df['%Rank_EL'] < 2.0, 1, 0))
        # df['Foreignness'] = df['Foreignness'].astype(float)
        # totnum = applyF(np.where(((df['%Rank_EL'] < 0.5) & (df['Foreignness'] > 1e-50)), 1, 0))        
        #df['RankEL_LT0.5_AND_TCRP_GT0E_TOTNUM']  = totnum
        #df['RankEL_LT0.5_AND_TCRP_GT0E_TOTFRAC'] = totnum / df['RankEL_LT0.5_totN']

    sorted_ELs = sorted(df['%Rank_EL'])
    #rank001_EL = (sorted_ELs[  1-0] if len(sorted_ELs) > (  1-0) else np.nan)
    rank010_EL = (sorted_ELs[ 10-1] if len(sorted_ELs) > ( 10-1) else np.nan)
    #rank020_EL = (sorted_ELs[ 20-1] if len(sorted_ELs) > ( 20-1) else np.nan)
    rank100_EL = (sorted_ELs[100-1] if len(sorted_ELs) > (100-1) else np.nan)
    #df['RankEL_1st'] = rank001_EL
    df['0_RankEL_10th'] = rank010_EL
    #df['RankEL_20st'] = rank020_EL
    df['0_RankEL_100th'] = rank100_EL
    '''
    for i, (features, ilr) in enumerate(zip(listof_features, ilrs)):
        if i == 0 or ('feature' in baseline.split(',')):
            y = ilr.predict_proba(df[features])
            if i != 0:
                df[F'PredictedProbWithOtherFeatureSet_{i}'] = [v[1] for v in y]
            else:
                df['PredictedProbability'] = [v[1] for v in y]
                df['Rank'] = df['PredictedProbability'].rank(method='first', ascending=False)
                df['Rank'] = df['Rank'].fillna(BIG_INT)
                df = df.sort_values('Rank')
                df = df.astype({"Rank":int})
                _, are_in_cum = compute_are_in_cum(df)
                if 'VALIDATED' in df.columns: 
                    VALID_N_TESTED = sum(are_in_cum)
                    VALID_CUMSUM = np.cumsum(df['VALIDATED'] * are_in_cum)
                else:
                    VALID_N_TESTED = -1
                    VALID_CUMSUM = -1
                PROBA_CUMSUM = np.cumsum(df['PredictedProbability'] * are_in_cum)
                # df['BindAff_LessThan100_NUM'] = sum(np.where(df['MT_BindAff'] < 100, 1, 0) * are_in_cum) / sum(are_in_cum)
                # df['RankEL_LT0.5_VALFRAC'] = sum(np.where(df['%Rank_EL'] < 0.5, 1, 0) * are_in_cum) / sum(are_in_cum)
                # df['RankEL_LT2.0_VALFRAC'] = sum(np.where(df['%Rank_EL'] < 2.0, 1, 0) * are_in_cum) / sum(are_in_cum)
                df = df.assign(VALID_N_TESTED=VALID_N_TESTED, VALID_CUMSUM=VALID_CUMSUM, PROBA_CUMSUM=PROBA_CUMSUM, ML_pipeline='default_ML_pipe')
    col2last(df, 'SourceAlterationDetail')
    col2last(df, 'PepTrace')
    dropcols(df, ['BindLevel', 'BindAff'])
    logger.info(F'N_rows={len(df)} N_cols={len(df.columns)} for {outpref}')
    df.to_csv(outpref, sep='\t', header=1, index=0, na_rep='NA')
    df.iloc[range(min((len(df),1000))),:].to_csv(outpref + '.top1000', sep='\t', header=1, index=0, na_rep='NA')

    if 'VALIDATED' in df.columns:
        print(df)
        evalres = assess_top20_top50_top100_ttif_fr_auprc(df)
        evalres2 = evalres._asdict()
        evalres2['ML_pipeline'] = 'default_ML_pipe'
        with open(outpref + '.performance', 'w') as file:
            json.dump(evalres2, file, indent=2)
        evaldict = defaultdict(list)
        evaldict['infile'].append(infile)
        evaldict['feature_transform'].append('default_transform')
        evaldict['ML_model_name'].append('default_model')
        for k, v in copy.deepcopy(evalres._asdict()).items():
            evaldict[k].append(v)
        with open(outpref + '.performance.tsv', 'w') as file:
            eval_df = pd.DataFrame.from_dict(evaldict)
            eval_df.to_csv(file, sep='\t', header=1, index=0, na_rep='NA')
    logger.info(F'END:   infile={infile}')
    return 0

from datetime import datetime
def mapfunc(tuple_arg):    
    pipename, pipe, big_train_X, big_train_y = tuple_arg
    logger = logging.getLogger('Training')
    logging.basicConfig(format=(F'neopredictor-train %(asctime)s - %(message)s'), 
        level=logging.DEBUG)
    logger.info(F'START: {pipename}')
    pipe.fit(big_train_X, big_train_y)
    logger.info(F'END:   {pipename}')
    return pipe

def main():
    
    # PRIME_rank,PRIME_BArank,mhcflurry_aff_percentile,mhcflurry_presentation_percentile
    parser = argparse.ArgumentParser(description='ML classifier for neoepitopes', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--train',  help='Train .validation files. ', required=False, nargs='*', default=[])
    parser.add_argument('--test',   help='Test  .validation files. ', required=False, nargs='*', default=[])
    parser.add_argument('--model',  help='Trained model file. ', required=False)
    parser.add_argument('--suffix', help='Suffix of the result files containing prediction. The format of the prediction file is <--test filename>.<--suffix>). ', 
        required = False, default = 'prediction')
    parser.add_argument('--peplens',help='Peptide length for keeping peptides. ', required=False, default='8,9,10,11')
    parser.add_argument('--ncores', help='Number of CPU cores to use for --train and --test. The special numbers -2 and 0 mean using one and all CPUs, respectively. ', required=False, type=int, default=16)
    parser.add_argument('--baseline', help='Comma-separated keywords. Keyword feature: test other feature sets. Keyword method: test other methods. ', required = False, default = 'feature')
    parser.add_argument('--feature-sets', help= 
        'List of strings with each string (i.e., feature set) consisting of comma-separated features. '
        'The first feature set is used by default, and all other feature sets are used as baselines. ', 
        required=False, nargs='+', default=[
        '%Rank_EL,MT_BindAff,Quantification,BindStab,Agretopicity,ln_NumTested',
        '%Rank_EL,MT_BindAff,Quantification,BindStab,Agretopicity'])
    parser.add_argument('--mintrain', help='Minimized train file to be outputted (empty string means not outputted). ', required=False, default='')
    parser.add_argument('-a', '--additonal_vars', help='Additional variants, a string consisting of comma-separated substrings, '
        'where each substring can be fusion, splicing, rna_snv, rna_indel, or rna_fsv ', default='')
    args = parser.parse_args()
    listof_features = [feature_set.split(',') for feature_set in args.feature_sets]
    peplens = [int(x) for x in args.peplens.split(',')]
    
    script_path = os.path.realpath(__file__)
    now = datetime.now()
    dt_string = now.strftime('%Y-%m-%d-%H-%M-%S')
    tmp_script_path = F'/tmp/{dt_string}.neopredictor.py'
    tmp_output_path = F'/tmp/{dt_string}.neopredictor.out'
    os.system(F'cp {script_path} {tmp_script_path}')
    logger.info(F'Will print predictor in json format to {tmp_output_path}')
    tmp_output = open(tmp_output_path, 'w')
    pp = pprint.PrettyPrinter(indent=2, stream=(tmp_output if tmp_output else sys.stdout))
    
    if args.train:
        dfs = []
        for infile in args.train:
            df = pd.read_csv(infile, sep='\t')
            df, are_in_cum = compute_are_in_cum(df)
            if len(df) > 0: dfs.append(df)
            logger.info(F'Finished reading {infile}')
        big_train_df = pd.concat(dfs)
        big_train_df['VALIDATED'] = big_train_df['VALIDATED'].astype(int)
        big_train_df = filtdf(big_train_df, peplens)
        #big_train_df.loc[:,'VALIDATED'] = [(0 if (-1==v) else v) for v in big_train_df['VALIDATED']]
        are_validated = ((big_train_df['VALIDATED'] == 0) | (big_train_df['VALIDATED'] == 1))
        big_train_df = big_train_df.loc[are_validated,:]
        big_train_y = big_train_df['VALIDATED'].astype(int)
        
        logger.info(F'Finished processing input. ')
        
        pipeline_names = []
        pipelines = []
        big_train_X = big_train_df.loc[:, listof_features[0]].copy()
        if args.mintrain:
            pd.concat([big_train_X, big_train_y], axis=1).to_csv(args.mintrain, sep='\t', header=1, index=0, na_rep='NA')
        big_train_X = big_train_X.round(5)
        big_train_X = pd.DataFrame(big_train_X)
        big_train_y = pd.Series(big_train_y)
        if ('method' in args.baseline.split(',')):
            for predictor_name, predictor in PREDICTORS.items():
                for scaler_name, scaler in SCALERS.items():            
                    pipe = make_pipeline(copy.deepcopy(scaler), copy.deepcopy(predictor))
                    pipelines.append(pipe)
                    pipeline_names.append((scaler_name + '/' +  predictor_name))
        map_args = [(pipename, pipe, big_train_X, big_train_y) for (pipename, pipe) in zip(pipeline_names, pipelines)]
        if args.ncores == -2:
            pipelines = list(map(mapfunc, map_args))
        else:
            pipelines = Parallel(n_jobs=args.ncores)(delayed(mapfunc)(arg) for arg in map_args)
        ilrs = []
        for features in listof_features:
            big_train_X = big_train_df.loc[:, features].copy().round(5)
            iso_scaler = IsotonicLogisticRegression(excluded_cols=['ln_NumTested']) # excluded_cols is used for better extrapolation
            iso_scaler.fit(big_train_X, big_train_y)
            ilrs.append(iso_scaler)
        
        logger.info(F'Finished training. ')

    if args.model and args.train:
        logger.info(F'Saving the model in pickle format to {args.model}')
        with open(args.model, 'wb') as file:
            pickle.dump([ilrs, pipeline_names, pipelines], file)
    elif args.model:
        logger.info(F'Loading the model in pickle format from {args.model}')
        with open(args.model, 'rb') as file:
            ilrs, pipeline_names, pipelines = pickle.load(file)
    else:
        ilrs, pipeline_names, pipelines = ([], [], [])
    pp.pprint([(i, ilr.get_info()) for i, ilr in enumerate(ilrs)])
    
    logger.info(F'Finished fitting IsotonicLogisticRegressions. ')
    
    map_args = [(ilrs, pipeline_names, pipelines, infile, args.suffix, peplens, args.baseline, listof_features, args.additonal_vars) for infile in args.test]
    if args.ncores == -2:
        ret = list(map(patientwise_predict, map_args))
    else:
        ret = Parallel(n_jobs=args.ncores)(delayed(patientwise_predict)(arg) for arg in map_args)

    if tmp_output: tmp_output.close()
    logger.info(F'Finished running {sys.argv[0]} with ret={ret}')
    
if __name__ == '__main__':
    main()

