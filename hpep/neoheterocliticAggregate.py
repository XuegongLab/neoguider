#!/usr/bin/env python3
'''Antigen-level aggregation of the heteroclitic-peptide ranking benchmark.

Companion to neoheterocliticEval.py (ROC-AUC, datasets D01-D26) and
neoheterocliticEvalTCR.py (Spearman, datasets D27-D30) in the same directory.

Those two scripts report one value per (dataset, method).  The 30 datasets are
not independent: they derive from 16 parental antigens, and several datasets are
nested subsets of others (D21 pools the four AH1 positions reported separately in
D23-D26; in the Maslak and Gigoux series each single-position dataset is a subset
of the corresponding all-position dataset).  This script collapses datasets to
parental antigens and reports, per method:

  * the antigen-level mean of the metric,
  * a percentile bootstrap confidence interval resampled over antigens,
  * a one-sided Wilcoxon signed-rank test against chance (0.5 for ROC-AUC, 0 for
    Spearman),
  * the standard deviation and the minimum across antigens,

and, across methods:

  * paired one-sided Wilcoxon signed-rank comparisons with Holm correction,
  * a Mann-Whitney U test of across-antigen SD between the PWM-based and the
    neural-network-ensemble families,
  * leave-one-antigen-out and leave-one-study-out recomputation,
  * MHC-allotype-stratified dataset-level means.

It reproduces every antigen-level number reported in the manuscript, including
Table 3.

Usage
-----
    python3 neoheterocliticAggregate.py \
        --auroc  <prefix>.full.tsv \
        --scc    <prefix>.scc.tsv \
        --output /tmp/neoheterocliticAggregate_out

--auroc is the long-form table written by neoheterocliticEval.py, with columns
    Dataset IDs, Dataset source, Modified residue, MHC, Gene, Peptides,
    #Tested-positive, #Tested-total, Ranking methods, AUROC, P-value
--scc is the same layout for the continuous datasets; if
neoheterocliticEvalTCR.py has been extended to write one, pass it here, otherwise
pass a wide matrix (rows = dataset IDs, columns = methods) and it is melted.

Both inputs are optional; whichever is supplied is analysed.
'''

import argparse, itertools, logging, os, sys

import numpy as np
import pandas as pd
import scipy.stats

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(script_dir)
try:
    from neoheterocliticEval import METHODS, add_asterisks
except ImportError:                                    # allow standalone use
    METHODS = ['MixMHCPred_aff', 'MixMHCpred_motif', 'PRIME_immunogenicity',
               'MHCmotifAtlas_motif', 'MHCmotifAtlas_motif_diff',
               'NetMHCpan_aff', 'NetMHCpan_aff_ratio', 'netMHCpan_EL',
               'MHCflurry_aff', 'MHCflurry_presentation',
               'Dist_bit', 'Dist_conserv']
    def add_asterisks(p):
        return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''

# Model families.  The grouping is by model architecture, not by the name of the
# reported output: a position weight matrix scores a peptide as a sum of
# independent per-residue terms, whereas a neural-network ensemble models the
# peptide jointly and exposes only an aggregate.
PWM_METHODS = ['MixMHCPred_aff', 'MixMHCpred_motif', 'PRIME_immunogenicity',
               'MHCmotifAtlas_motif', 'MHCmotifAtlas_motif_diff']
NN_METHODS  = ['NetMHCpan_aff', 'NetMHCpan_aff_ratio', 'netMHCpan_EL',
               'MHCflurry_aff', 'MHCflurry_presentation']
DIST_METHODS = ['Dist_bit', 'Dist_conserv']

def family_of(method):
    return ('PWM-based' if method in PWM_METHODS else
            'NN ensemble' if method in NN_METHODS else 'Distance')

# Dataset -> parental antigen.  Datasets that scan different positions of the
# same parental peptide, or that are nested inside one another, share an entry.
DATASET2ANTIGEN = {
    'D01M07AA0': 'TYRP1',            'D02M07BB0': 'PSMA',
    'D03M18CB0': 'WT1-SQASSGQARMFPNAPYL',  'D05M18CC0': 'WT1-SQASSGQARMFPNAPYL',
    'D07M18CB9': 'WT1-SQASSGQARMFPNAPYL',  'D09M18CC9': 'WT1-SQASSGQARMFPNAPYL',
    'D04M18DB0': 'WT1-RMFPNAPYLL',   'D06M18DC0': 'WT1-RMFPNAPYLL',
    'D08M18DB1': 'WT1-RMFPNAPYLL',   'D10M18DC1': 'WT1-RMFPNAPYLL',
    'D11M22ED0': 'CALR-RRMMRTK-LQGWTEA', 'D13M22ED6': 'CALR-RRMMRTK-LQGWTEA',
    'D12M22FD0': 'CALR-RMMRTKMRM',   'D14M22FD5': 'CALR-RMMRTKMRM',
    'D15T01GB0': 'CEA-IMIGVLVGV',    'D16T01HB0': 'MAGEA3-KVAELVHFL',
    'D17T01IB0': 'MAGEA2-YLQLVFGIEV','D18T01JB0': 'HIVpol-ILKEPVHGV',
    'D19T01KB0': 'HBVpol-GLSRYVARL',
    'D20T22LA0': 'WT1-RMFPNAPYL',    'D22T22LA7': 'WT1-RMFPNAPYL',
    'D21T22ME0': 'AH1-SPSYVYHQF',    'D23T22ME1': 'AH1-SPSYVYHQF',
    'D24T22ME3': 'AH1-SPSYVYHQF',    'D25T22ME5': 'AH1-SPSYVYHQF',
    'D26T22ME8': 'AH1-SPSYVYHQF',
    'D27T24a':   'OVA-SIINFEKL',     'D28T24b':   'OVA-SIINFEKL',
    'D29T24c':   'RNF43-VPSVWRSSL',  'D30T24d':   'CMV-NLVPMVATV',
}

def antigen_of(dataset_id):
    if dataset_id in DATASET2ANTIGEN:
        return DATASET2ANTIGEN[dataset_id]
    raise KeyError(F'Dataset {dataset_id} has no parental-antigen assignment; '
                   F'add it to DATASET2ANTIGEN.')

def study_of(dataset_id):
    n = int(dataset_id[1:3])
    if n <= 2:  return 'Houghton'
    if n <= 10: return 'Maslak'
    if n <= 14: return 'Gigoux'
    if n <= 19: return 'Tangri'
    if n <= 26: return 'He'
    return 'Drost'

def read_long(path, value_col):
    '''Read either the long form written by neoheterocliticEval.py or a wide
    matrix of dataset x method, and return a tidy frame.'''
    df = pd.read_csv(path, sep='\t', header=0)
    if 'Ranking methods' in df.columns:
        out = df.rename(columns={'Dataset IDs': 'dataset',
                                 'Ranking methods': 'method',
                                 value_col: 'value'})
        keep = ['dataset', 'method', 'value']
        for extra in ('MHC', 'Modified residue', 'Dataset source'):
            if extra in df.columns:
                out[extra] = df[extra]
                keep.append(extra)
        return out[keep].drop_duplicates(subset=['dataset', 'method'])
    idcol = df.columns[0]
    out = df.melt(id_vars=[idcol], var_name='method', value_name='value')
    return out.rename(columns={idcol: 'dataset'})

def antigen_matrix(tidy, datasets):
    '''Collapse datasets to parental antigens by unweighted mean.'''
    sub = tidy[tidy['dataset'].isin(datasets)].copy()
    sub['antigen'] = sub['dataset'].map(antigen_of)
    wide = sub.pivot_table(index='antigen', columns='method', values='value',
                           aggfunc='mean')
    return wide.reindex(columns=[m for m in METHODS if m in wide.columns])

def bootstrap_ci(values, n_boot, rng, alpha=0.05):
    n = len(values)
    draws = rng.integers(0, n, size=(n_boot, n))
    means = np.asarray(values)[draws].mean(axis=1)
    return np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])

def signed_rank_vs(values, null_value):
    diffs = np.asarray(values, dtype=float) - null_value
    if np.allclose(diffs, 0):
        return np.nan
    return scipy.stats.wilcoxon(diffs, alternative='greater').pvalue

def holm(pvalues):
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adjusted = np.minimum(1.0, (len(p) - np.arange(len(p))) * p[order])
    adjusted = np.maximum.accumulate(adjusted)
    out = np.empty_like(adjusted)
    out[order] = adjusted
    return out

def summarize(wide, null_value, label, n_boot, rng, out_rows):
    logging.info(F'--- antigen-level summary: {label} '
                 F'({wide.shape[0]} antigens: {", ".join(wide.index)}) ---')
    for method in wide.columns:
        v = wide[method].dropna().values
        lo, hi = bootstrap_ci(v, n_boot, rng)
        p = signed_rank_vs(v, null_value)
        out_rows.append({'Regime': label, 'Method': method,
                         'Family': family_of(method),
                         'N antigens': len(v),
                         'Mean': round(float(v.mean()), 3),
                         'CI low': round(float(lo), 3),
                         'CI high': round(float(hi), 3),
                         'SD across antigens': round(float(v.std(ddof=1)), 3),
                         'Worst antigen': round(float(v.min()), 3),
                         'P vs chance': p,
                         'Significance': add_asterisks(p)})
        print(F'{method:<26}{v.mean():>8.3f}  95% CI [{lo:.3f}, {hi:.3f}]'
              F'  SD {v.std(ddof=1):.3f}  min {v.min():.3f}  p {p:.4f}')

def paired_tests(wide, pairs):
    print('\n--- paired comparisons across antigens (one-sided, Holm) ---')
    raw = []
    for a, b in pairs:
        if a not in wide.columns or b not in wide.columns:
            continue
        d = (wide[a] - wide[b]).dropna().values
        raw.append((a, b, float(d.mean()),
                    scipy.stats.wilcoxon(d, alternative='greater').pvalue))
    if not raw:
        return pd.DataFrame()
    adj = holm([r[3] for r in raw])
    for (a, b, delta, p), q in zip(raw, adj):
        print(F'{a:<26} > {b:<26} delta {delta:+.3f}  p {p:.4f}  Holm {q:.4f}')
    return pd.DataFrame(raw, columns=['Method A', 'Method B', 'Mean difference',
                                      'P raw']).assign(**{'P Holm': adj})

def family_consistency(wide):
    pwm = np.array([wide[m].std(ddof=1) for m in PWM_METHODS if m in wide.columns])
    nn  = np.array([wide[m].std(ddof=1) for m in NN_METHODS if m in wide.columns])
    if len(pwm) == 0 or len(nn) == 0:
        return np.nan
    p = scipy.stats.mannwhitneyu(pwm, nn, alternative='less').pvalue
    print(F'\n--- across-antigen SD: PWM {pwm.mean():.3f} vs NN ensemble '
          F'{nn.mean():.3f}, Mann-Whitney p = {p:.4f} ---')
    return p

def leave_one_out(tidy, datasets, key_fn, label):
    print(F'\n--- leave-one-{label}-out: best PWM minus best NN ensemble ---')
    keys = sorted({key_fn(d) for d in datasets})
    rows = []
    for k in keys:
        kept = [d for d in datasets if key_fn(d) != k]
        if not kept:
            continue
        wide = antigen_matrix(tidy, kept)
        best_pwm = max(wide[m].mean() for m in PWM_METHODS if m in wide.columns)
        best_nn  = max(wide[m].mean() for m in NN_METHODS  if m in wide.columns)
        rows.append({'Excluded': k, 'N antigens': wide.shape[0],
                     'Best PWM': round(best_pwm, 3), 'Best NN': round(best_nn, 3),
                     'Difference': round(best_pwm - best_nn, 3)})
        print(F'exclude {k:<26} antigens {wide.shape[0]:>2}  '
              F'best PWM {best_pwm:.3f}  best NN {best_nn:.3f}  '
              F'difference {best_pwm - best_nn:+.3f}')
    return pd.DataFrame(rows)

def mhc_stratified(tidy, datasets):
    if 'MHC' not in tidy.columns:
        return pd.DataFrame()
    print('\n--- MHC-allotype-stratified dataset-level means ---')
    sub = tidy[tidy['dataset'].isin(datasets)]
    rows = []
    for allotype, grp in sub.groupby('MHC'):
        wide = grp.pivot_table(index='dataset', columns='method', values='value')
        row = {'MHC': allotype, 'N datasets': wide.shape[0]}
        row.update({m: round(float(wide[m].mean()), 3)
                    for m in wide.columns if m in METHODS})
        rows.append(row)
        print(F'{allotype:<14} n={wide.shape[0]:<3} ' + '  '.join(
            F'{m}={wide[m].mean():.3f}' for m in
            ['MixMHCPred_aff', 'MHCmotifAtlas_motif', 'NetMHCpan_aff', 'MHCflurry_aff']
            if m in wide.columns))
    return pd.DataFrame(rows)

def main():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(pathname)s:%(lineno)d %(levelname)s - %(message)s')
    parser = argparse.ArgumentParser(
        description='Collapse the heteroclitic benchmark to parental antigens and '
                    'compute bootstrap confidence intervals and paired tests.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-a', '--auroc', help='Long-form ROC-AUC table from neoheterocliticEval.py '
                                              '(<prefix>.full.tsv)', default=None)
    parser.add_argument('-s', '--scc', help='Spearman table for datasets D27-D30', default=None)
    parser.add_argument('-o', '--output', help='Output prefix',
                        default='/tmp/neoheterocliticAggregate_out')
    parser.add_argument('-n', '--n-boot', type=int, default=10000,
                        help='Bootstrap resamples over parental antigens')
    parser.add_argument('--seed', type=int, default=20260821,
                        help='Seed for the bootstrap; fixed so the reported '
                             'confidence intervals are reproducible')
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    summary_rows, tables = [], {}

    if args.auroc:
        tidy = read_long(args.auroc, 'AUROC')
        anchor = sorted(d for d in tidy['dataset'].unique() if int(d[1:3]) <= 14)
        facing = sorted(d for d in tidy['dataset'].unique() if 15 <= int(d[1:3]) <= 26)

        summarize(antigen_matrix(tidy, anchor), 0.5, 'MHC-anchor ROC-AUC',
                  args.n_boot, rng, summary_rows)
        wide_facing = antigen_matrix(tidy, facing)
        summarize(wide_facing, 0.5, 'TCR-facing ROC-AUC',
                  args.n_boot, rng, summary_rows)

        pairs = [('MixMHCPred_aff', 'MHCflurry_aff'),
                 ('MixMHCPred_aff', 'NetMHCpan_aff'),
                 ('MixMHCPred_aff', 'netMHCpan_EL'),
                 ('MixMHCPred_aff', 'MHCflurry_presentation'),
                 ('MHCmotifAtlas_motif', 'NetMHCpan_aff'),
                 ('MixMHCpred_motif', 'NetMHCpan_aff'),
                 ('MixMHCPred_aff', 'Dist_bit'),
                 ('MixMHCPred_aff', 'Dist_conserv')]
        tables['paired'] = paired_tests(wide_facing, pairs)
        family_consistency(wide_facing)
        tables['loao'] = leave_one_out(tidy, facing, antigen_of, 'antigen')
        tables['loso'] = leave_one_out(tidy, facing, study_of, 'study')
        tables['mhc']  = mhc_stratified(tidy, facing)

        # label provenance: assayed versus computationally enumerated negatives
        assayed = [d for d in anchor if study_of(d) == 'Houghton']
        enumerated = [d for d in anchor if study_of(d) in ('Maslak', 'Gigoux')]
        print('\n--- anchor datasets: assayed versus enumerated negatives ---')
        for label, subset in (('assayed', assayed), ('enumerated', enumerated)):
            wide = tidy[tidy['dataset'].isin(subset)].pivot_table(
                index='dataset', columns='method', values='value')
            print(F'{label:<12} n={len(subset):<3} ' + '  '.join(
                F'{m}={wide[m].mean():.3f}' for m in
                ['MHCmotifAtlas_motif', 'MixMHCPred_aff', 'NetMHCpan_aff', 'MHCflurry_aff']
                if m in wide.columns))

    if args.scc:
        tidy_scc = read_long(args.scc, 'SCC')
        cont = sorted(tidy_scc['dataset'].unique())
        summarize(antigen_matrix(tidy_scc, cont), 0.0, 'TCR-facing Spearman',
                  args.n_boot, rng, summary_rows)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.output + '.antigen_level.tsv', sep='\t', index=False,
                   na_rep='NA')
    logging.info(F'Saved {args.output}.antigen_level.tsv')
    for name, table in tables.items():
        if len(table):
            table.to_csv(F'{args.output}.{name}.tsv', sep='\t', index=False,
                         na_rep='NA')
            logging.info(F'Saved {args.output}.{name}.tsv')

if __name__ == '__main__':
    main()
