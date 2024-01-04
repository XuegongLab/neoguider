import argparse, json, logging, multiprocessing, os, pickle, pprint, sys

import numpy as np
import pandas as pd

from collections import namedtuple
from scipy import stats

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler #, PowerTransformer, MinMaxScaler, FunctionTransformer

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from IsotonicLogisticRegression import IsotonicLogisticRegression

BIG_INT = 2**30
SCALERS = [QuantileTransformer(), StandardScaler()]
PREDICTORS = [LogisticRegression(), GradientBoostingClassifier()]

logging.basicConfig(format=('nhh_train %(asctime)s - %(message)s'), level=logging.DEBUG)

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
# features = ['%Rank_EL', 'MT_BindAff', 'Quantification', 'BindStab', 'Agretopicity', 'Foreignness']
featuresA0 = (['%Rank_EL', 'MT_BindAff', 'Quantification', 'BindStab', 'Agretopicity'])
featuresA1 = (['%Rank_EL', 'MT_BindAff', 'Quantification', 'BindStab', 'Agretopicity', 
# 'F_RankEL_LT0.5_ValF', 'F_RankEL_LT2.0_ValF',
'F_LOG_N_VALIDATION',
#'F_0_agr_TPM_SPEARMAN_sgn_inv_P',
#'F_0_agr_TPM_SPEARMAN_statistic',
#'F_1_RankEL_TPM_SPEARMAN_sgn_inv_P',
#'F_1_RankEL_TPM_SPEARMAN_statistic',
]
# + [F'F_{idx}_RankEL_agr_SPEARMAN_sgn_inv_P' for idx in range(2)]
# + [F'F_{idx}_RankEL_agr_SPEARMAN_statistic' for idx in range(2)]
)
listof_features = [featuresA0, featuresA1]

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
    auprc = (roc_auc_score(df0['VALIDATED'], -df0['Rank']) if (set(df0['VALIDATED']) == set([0, 1])) else np.nan)
    PerformanceResult = namedtuple("PerformanceResult", "top20 top50 top100 TTIF FR AUPRC TFA_mean")
    return PerformanceResult(top20, top50, top100, ttif, fr, auprc, np.mean([ttif, fr, auprc]))

def filtdf(df):
    df = df.loc[~pd.isna(df['PepTrace']),:].copy()
    df['Agretopicity'] = df['Agretopicity'].fillna(1e-9)
    return df

def filter_testdf(df):
    is_SNV_detected_from_RNA = df['Identity'].str.startswith('SNV_R')
    is_FSV_detected_from_RNA = df['Identity'].str.startswith('FSV_R')
    is_INS_detected_from_RNA = df['Identity'].str.startswith('INS_R')
    is_DEL_detected_from_RNA = df['Identity'].str.startswith('DEL_R')
    is_FUS = (df['Identity'].str.startswith('FU_') | df['Identity'].str.startswith('FUS_'))
    is_SP  = (df['Identity'].str.startswith('SP_') | df['Identity'].str.startswith('SPL_'))
    is_kept = ~(is_SNV_detected_from_RNA | is_FSV_detected_from_RNA | is_INS_detected_from_RNA | is_DEL_detected_from_RNA | is_FUS | is_SP)
    ret = df.loc[is_kept, :]
    return ret

def compute_are_in_cum(df):
    if 'VALIDATED' in df.columns: 
        are_in_cum = np.where(df['VALIDATED'] >= 0, 1, 0)
    else:
        # are_in_cum = np.where(df['%Rank_EL'] < 2.0, 1, 0)
        are_in_cum = np.where(df['MT_BindAff'] < 500.0, 1, 0)
    # df['are_in_cum'] = are_in_cum
    df['F_RankEL_LT0.5_ValF'] = ((sum(np.where(df['%Rank_EL'] < 0.5, 1, 0) * are_in_cum) / sum(are_in_cum)) if sum(are_in_cum) else -1)
    df['F_RankEL_LT2.0_ValF'] = ((sum(np.where(df['%Rank_EL'] < 2.0, 1, 0) * are_in_cum) / sum(are_in_cum)) if sum(are_in_cum) else -1)
    df['F_LOG_N_VALIDATION'] = (np.log(sum(are_in_cum)) if sum(are_in_cum) else 0)
    for idx, selector in enumerate([[1] * len(df), are_in_cum]):
        res = stats.spearmanr(
            [x for (x,y) in zip(df['%Rank_EL'],     selector) if y],
            [x for (x,y) in zip(df['Agretopicity'], selector) if y])
        df[F'F_{idx}_RankEL_agr_SPEARMAN_sgn_inv_P'] = nan_replace(np.sign(res.statistic) / max((1e-100, res.pvalue)), 0)
        df[F'F_{idx}_RankEL_agr_SPEARMAN_statistic'] = nan_replace(res.statistic, 0)
    return df, are_in_cum

def applyF(arr, f=sum):
    if len(arr): return f(arr)
    else: return np.nan

def patientwise_predict(tuple_arg):
    ilrs, pipelines, infile, suffix = tuple_arg
    
    outpref = infile + '.' + suffix
    logging.info(F'InFile={infile} OutPref={outpref}')
    df1 = pd.read_csv(infile, sep='\t')
    df1 = filtdf(df1)
    df1 = filter_testdf(df1)
    if len(df1) == 0: 
        logging.warning(F'Skipping file={infile} because it is empty after filtering. ')
        return -1
    
    #pipedir = (infile + '.other-pred.dir')
    #os.system(F'mkdir -p {pipedir}')
    for i, pipeline in enumerate(pipelines):
        pipedf = df1.copy()
        y = pipeline.predict_proba(pipedf[listof_features[0]])
        pipedf['OtherPredictedProb'] = [v[1] for v in y]
        pipedf['Rank'] = pipedf['OtherPredictedProb'].rank(method='first', ascending=False)
        pipedf['Rank'] = pipedf['Rank'].fillna(BIG_INT)
        pipedf = pipedf.sort_values('Rank')
        pipedf = pipedf.astype({"Rank":int})
        # other_pred = F'{pipedir}/{i}.other-pred'
        other_pred = F'{outpref}.other_pred.{i}'
        pipedf.to_csv(other_pred, sep='\t', header=1, index=0, na_rep = 'NA')
        if 'VALIDATED' in pipedf.columns:
            evalres = assess_top20_top50_top100_ttif_fr_auprc(pipedf)            
            with open(other_pred + '.performance', 'w') as file:
                json.dump(evalres._asdict(), file, indent=2)
            #for f.write(json.dumps evalres._asdict)
    df, are_in_cum = compute_are_in_cum(df1)
    #test_arr = list(filter(lambda x:(x>0), np.where(df['%Rank_EL'] < 0.5, 1, 0) * are_in_cum * (df['Quantification'] + 1e-50)))
    #print(F'test_array={test_arr}')
        
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
    
    for i, (features, ilr) in enumerate(zip(listof_features, ilrs)):
        y = ilr.predict_proba(df[features])
        if i < len(listof_features) - 1:
            df[F'PredictedProb_{i+1}'] = [v[1] for v in y]
            continue
        df['PredictedProbability'] = [v[1] for v in y]
        df['Rank'] = df['PredictedProbability'].rank(method='first', ascending=False)
        df['Rank'] = df['Rank'].fillna(BIG_INT)
        df = df.sort_values('Rank')
        df = df.astype({"Rank":int})
        _, are_in_cum = compute_are_in_cum(df)
        #data = data.drop(drop_cols, axis=1)        
        if 'VALIDATED' in df.columns: 
            df['VALID_N_TESTED'] = sum(are_in_cum)
            df['VALID_CUMSUM'] = np.cumsum(df['VALIDATED'] * are_in_cum)
        else:
            df['VALID_N_TESTED'] = -1
            df['VALID_CUMSUM'] = -1
        df['PROBA_CUMSUM'] = np.cumsum(df['PredictedProbability'] * are_in_cum)
        # df['BindAff_LessThan100_NUM'] = sum(np.where(df['MT_BindAff'] < 100, 1, 0) * are_in_cum) / sum(are_in_cum)
        #df['RankEL_LT0.5_VALFRAC'] = sum(np.where(df['%Rank_EL'] < 0.5, 1, 0) * are_in_cum) / sum(are_in_cum)
        #df['RankEL_LT2.0_VALFRAC'] = sum(np.where(df['%Rank_EL'] < 2.0, 1, 0) * are_in_cum) / sum(are_in_cum)        
    col2last(df, 'SourceAlterationDetail')
    col2last(df, 'PepTrace')
    dropcols(df, ['BindLevel', 'BindAff'])
    logging.info(F'N_rows={len(df)} N_cols={len(df.columns)} for tsv={outpref}.prediction')
    df.to_csv(outpref + '.prediction', sep='\t', header=1, index=0, na_rep = 'NA')
    if 'VALIDATED' in df.columns:
        evalres = assess_top20_top50_top100_ttif_fr_auprc(df)
        with open(outpref + '.prediction.performance', 'w') as file:
            json.dump(evalres._asdict(), file, indent=2)
    return 0

from datetime import datetime
def main():
    
    parser = argparse.ArgumentParser(description = 'ML classifier for neoepitopes', formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--train',  help = 'Train .validation files. ', required = False, nargs = '*', default=[])
    parser.add_argument('--test',   help = 'Test  .validation files. ', required = False, nargs = '*', default=[])
    parser.add_argument('--model',  help = 'Trained model file. ', required = False)
    parser.add_argument('--suffix', help = 'Suffix of the result files containing prediction. ', required = False)

    args = parser.parse_args()
    
    script_path = os.path.realpath(__file__)
    now = datetime.now()
    dt_string = now.strftime('%Y-%m-%d-%H-%M-%S')
    tmp_script_path = F'/tmp/{dt_string}.nhh_train.py'
    tmp_output_path = F'/tmp/{dt_string}.nhh_train.out'
    #tmp_script = open(tmp_script_path, 'w')
    os.system(F'cp {script_path} {tmp_script_path}')
    tmp_output = open(tmp_output_path, 'w')
    pp = pprint.PrettyPrinter(indent=2, stream=tmp_output)
    
    if args.train:
        dfs = []
        for infile in args.train:
            df = pd.read_csv(infile, sep='\t')
            df, are_in_cum = compute_are_in_cum(df)
            dfs.append(df)
            logging.info(F'Finished reading {infile}. ')
        big_train_df = pd.concat(dfs)
        big_train_df['VALIDATED'] = big_train_df['VALIDATED'].astype(int)
        big_train_df = filtdf(big_train_df)
        are_validated = ((big_train_df['VALIDATED'] == 0) | (big_train_df['VALIDATED'] == 1))
        big_train_df = big_train_df.loc[are_validated,:]
        
        # big_train_X1.to_csv('/tmp/big_train_X.tsv', sep = '\t')    
        big_train_y = big_train_df['VALIDATED'].astype(int)
        # big_train_y = np.maximum(big_train_y.astype(int), 0)
        #print(big_train_X1)
        #print(big_train_y)
        
        logging.info(F'Finished processing input. ')
        
        pipelines = []
        big_train_X = big_train_df[listof_features[0]]
        for scaler in SCALERS:
            for predictor in PREDICTORS:
                pipe = make_pipeline(scaler, predictor)
                pipe.fit(big_train_X, big_train_y)
                pipelines.append(pipe)
        
        ilrs = []
        for features in listof_features:
            big_train_X = big_train_df[features]
            iso_scaler = IsotonicLogisticRegression() # (excluded_cols=['F_LOG_N_VALIDATION']) # ()
            #log_regression = LogisticRegression(penalty=None)
            #ilr = make_pipeline(iso_scaler, log_regression)
            #ilr.fit(big_train_X, big_train_y) # #ilr.fit(big_train_X, big_train_y, is_centered = False)             
            #ilrs.append(ilr)
            iso_scaler.fit(big_train_X, big_train_y)
            ilrs.append(iso_scaler)

    if args.model and args.train:
        logging.info(F'Saving the model in pickle format to {args.model}')
        with open(args.model, 'wb') as file:
            pickle.dump([ilrs, pipelines], file)
    elif args.model:
        logging.info(F'Loading the model in pickle format from {args.model}')
        with open(args.model, 'rb') as file:
            ilrs, pipelines = pickle.load(file)
    
    pp.pprint([(i, ilr.get_info()) for i, ilr in enumerate(ilrs)])
    
    logging.info(F'Finished fitting IsotonicLogisticRegressions. ')
    
    p = multiprocessing.Pool(12)
    ret = list(map(patientwise_predict, [(ilrs, pipelines, infile, args.suffix) for infile in args.test]))
    # print(F'RET={ret}')
    # tmp_script.close()
    tmp_output.close()
    logging.info(F'Finished running {sys.argv[0]}. ')
    
if __name__ == '__main__':
    main()

