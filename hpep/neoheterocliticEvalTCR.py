import argparse, os, sys
import numpy as np
import scipy
import pandas as pd

import matplotlib
from matplotlib import pyplot as plt

import seaborn as sns

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(script_dir)
from neoheterocliticEval import METHODS, fill_vals, heatmap2annotations, norm_mhc

parser = argparse.ArgumentParser(description='Take as input *.expansion.untraced file (from stdin) and output performance metrics (to stdout). ', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#/mnt/d/heteroclitic/v01-hpep.outdir/prioritization/v02_prelim_features_from_pmhcs.tsv.expansion.untraced
parser.add_argument('-i', '--input',  help='Output of result file', default='/mnt/d/heteroclitic/v05-hpep.outdir/prioritization/v05_prelim_features_from_pmhcs.tsv.expansion.untraced')
parser.add_argument('-o', '--output', help='Output heatmap', default='/tmp/neoheterocliticEvalTCR_out')
parser.add_argument('-m', '--motif', help='Motif directory', default=F'/mnt/d/code/neoguider/software/prime/MixMHCpred/lib/pwm/')

args = parser.parse_args()

# get data from https://www.cell.com/cell-genomics/fulltext/S2666-979X(24)00238-6
#print(sys.argv)

# Individual APL screening, Simultaneous N4 screening, Normalized data, TCR_info
df21 = pd.read_excel('mmc2.xlsx', sheet_name='Individual APL screening', header=1)
df23 = pd.read_excel('mmc2.xlsx', sheet_name='Normalized data', header=1)
df2tcrs = 'Ed5 Ed8 Ed9 Ed10 Ed16-1 Ed16-30 Ed21 Ed23 Ed28 Ed31 Ed33 Ed39 Ed40 Ed45 Ed46'.split()

df21['ET_pep'] = [x[1] for x in df21['Sequence'].str.split('-')]
df2end = pd.concat([df21[['Sequence', 'ET_pep']], df23[df2tcrs]], axis=1)
df2end[df2tcrs] = df2end[df2tcrs] / df2end.iloc[-1][df2tcrs]
df2end['Signal'] = df2end[df2tcrs].sum(axis=1) / len(df2tcrs)
print(df2end)

df31 = pd.read_excel('mmc3.xlsx', sheet_name='Individual APL screening', header=1)
df33 = pd.read_excel('mmc3.xlsx', sheet_name='Normalized data', header=1)
df3tcrs = 'B11	B15	B3	F4	E8	B13	H6	G6	F5	H5	B2	B6	B5	E9	E4	G2	B16	B14	B7	B10'.split() # the positive control 'OTI' is excluded

df31['ET_pep'] = [x[1] for x in df21['Sequence'].str.split('-')]
df3end = pd.concat([df31[['Sequence', 'ET_pep']], df33[df3tcrs]], axis=1)
df3end[df3tcrs] = df3end[df3tcrs] / df3end.iloc[-1][df3tcrs]
df3end['Signal'] = df3end[df3tcrs].sum(axis=1) / len(df3tcrs)
print(df3end)

df43 = pd.read_excel('mmc4.xlsx', sheet_name='Normalized by PC', header=0)
df4tcrs = 'R21 R23 R24 R25 R26 R27 R28'.split()

df43['ET_pep'] = df43['Peptide']
df4end = pd.concat([df43[['ET_pep']], df43[df4tcrs]], axis=1)
df4end[df4tcrs] = df4end[df4tcrs] / df4end.iloc[0][df4tcrs]
df4end['Signal'] = df4end[df4tcrs].sum(axis=1) / len(df4tcrs)
print(df4end)

df52 = pd.read_excel('mmc5.xlsx', sheet_name='Mean', header=0)
df54 = pd.read_excel('mmc5.xlsx', sheet_name='peptides', header=0)
df5tcrs = '1_4	2_4	3_4	4_4	5_2	6_2	10_4	11_4	51_10	52_10	54_10	65_8	66_8	73_14	74_14	77_14	81_14	82_14	83_3	84_3'.split()

df5end = pd.merge(left=df52, right=df54, how='inner', left_on='Peptide_ID', right_on='ID')
df5end['ET_pep'] = df5end['Peptide']
df5end[df5tcrs] = df5end[df5tcrs] / df5end.iloc[0][df5tcrs]
df5end['Signal'] = df5end[df5tcrs].sum(axis=1) / len(df5tcrs)
print(df5end)

# /mnt/d/heteroclitic/v05-hpep.outdir/prioritization/v05_prelim_features_from_pmhcs.tsv.expansion.untraced
df2end['HLA_type'] = 'H-2-Kb'      # educated OVA
df3end['HLA_type'] = 'H-2-Kb'      # naive OVA
df4end['HLA_type'] = 'HLA-B*07:02' # tumor neoepitope
df5end['HLA_type'] = 'HLA-A*02:01' # CMV

#dfend = pd.concat([df2end, df3end, df4end, df5end], axis=0)

resfile = args.input # '/mnt/d/heteroclitic/v05-hpep.outdir/prioritization/v05_prelim_features_from_pmhcs.tsv.expansion.untraced'
resdf = pd.read_csv(resfile, sep='\t', header=0)
resdf['ET_pep'] = resdf['ET_pep'].str.upper()
print(f'resdf={resdf}')

#print(f'HLA\tpeptide\tMethod\tstatistic\tpvalue')

DATA_IDS = ['D27T24a', 'D28T24b', 'D29T24c', 'D30T24d']
df_effects = pd.DataFrame(np.nan, index=DATA_IDS, columns=sorted(METHODS))
df_pvalues = pd.DataFrame(np.nan, index=DATA_IDS, columns=sorted(METHODS))

for df, dataID, HLA, pep in zip(
        [df2end, df3end, df4end, df5end],
        DATA_IDS,
        ['H-2-Kb', 'H-2-Kb', 'HLA-B*07:02', 'HLA-A*02:01'], 
        ['SIINFEKL-educated', 'SIINFEKL-naive', 'VPSVWRSSL', 'NLVPMVATV']):
    df3 = pd.merge(left=df, right=resdf, how='inner', on=['ET_pep', 'HLA_type'])
    #print(df3)
    df3 = fill_vals(df3, norm_mhc(HLA), args.motif)
    for meth in METHODS:
        stat, pval = scipy.stats.spearmanr(df3[meth], df3['Signal'])
        df_effects.loc[dataID, meth] = stat
        df_pvalues.loc[dataID, meth] = pval
    
    #df3['MHCmotifAtlas']        = (df3['ETinfo'])
    #df3['MHCmotifAtlas_diff']   = (df3['ETinfo']-df3['MTinfo'])
    #df3['NetMHCpan_Aff']        = (df3['ET_BindAff'])
    #df3['NetMHCpan_Aff_ratio']  = (df3['ET_BindAff']/df3['MT_BindAff'])
    #df3['netMHCpan_EL']         = (df3['%Rank_EL'])
    #df3['PRIME_Immunogenicity'] = (df3['PRIME_rank'])
    #df3['MixMHCPred_Aff']       = (df3['PRIME_BArank'])
    #for meth in ['MHCmotifAtlas', 'MHCmotifAtlas_diff', 'NetMHCpan_Aff', 'NetMHCpan_Aff_ratio', 'netMHCpan_EL', 'PRIME_Immunogenicity', 'MixMHCPred_Aff']:
    #    stat, pval = scipy.stats.spearmanr(df3[meth], df3['Signal'])
    #    print(f'{HLA}\t{pep}\t{meth}\t{stat}\t{pval}')

# Compute the mean for each column
mean_row = df_effects.mean(numeric_only=True)
min_row  = df_pvalues.min (numeric_only=True)

mean_row.name = 'Mean'
df_effects = pd.concat([df_effects, mean_row.to_frame().T])
df_pvalues = pd.concat([df_pvalues, min_row .to_frame().T])

annotations = heatmap2annotations(df_effects, df_pvalues)

fig, ax = plt.subplots(figsize=(10, 7.5/3), constrained_layout=True)
ax = sns.heatmap(df_effects, annot=annotations, ax=ax, fmt='', center=0) #(, fmt='.3g')

# Rotate x-axis tick labels
plt.setp(ax.get_xticklabels(), rotation=20, ha='right')
plt.setp(ax.get_yticklabels(), rotation=0 , ha='right', va='center', fontfamily='monospace')
ax.set_ylabel('Dataset IDs', labelpad=0)  # Adjust pad to bring it closer

plt.savefig(args.output + '.heatmap.pdf')
plt.savefig(args.output + '.heatmap.png', dpi=600)
plt.close()

