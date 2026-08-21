import argparse, collections, copy, io, logging, sys
import numpy as np
import scipy as sp
import pandas as pd

import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns

#sns.set(font="SimSun")
plt.figure(figsize=(10, 6))

AAS = 'ARNDCQEGHILKMFPSTWYV'
PMC2195959_17aas = AAS.replace('C', '').replace('W', '').replace('M', '')

PMC2195959_table1a = '''
A,  Av , C,  Cv , D,  Dv , E,  Ev , F,  Fv , G,  Gv , H,  Hv , I,  Iv , K,  Kv , L,  Lv
A,  10 , C,  10 , D,  10 , E,  10 , F,  10 , G,  10 , H,  10 , I,  10 , K,  10 , L,  10
S,  45 , V,  55 , N,  35 , Q,  33 , L,  38 , S,  28 , Q,  20 , L,  25 , R,  27 , I,  33
T,  48 , T,  65 , E,  40 , N,  42 , Y,  43 , A,  42 , E,  38 , M,  45 , Q,  60 , M,  40
P,  53 , A,  67 , Q,  62 , D,  47 , I,  48 , T,  47 , N,  62 , F,  52 , H,  68 , F,  45
G,  57 , S,  73 , T,  72 , H,  53 , M,  62 , D,  63 , R,  82 , V,  52 , N,  72 , V,  55
C,  93 , P,  80 , S,  77 , T,  83 , V,  67 , P,  70 , K,  87 , Y,  88 , E,  75 , Y,  82
V,  95 , I,  99 , H,  78 , K,  87 , W,  72 , N,  73 , P,  90 , T, 103 , D,  85 , H,  97
D, 100 , Y, 102 , G,  88 , P,  90 , H, 102 , E,  83 , D,  92 , H, 113 , M,  92 , Q, 105
M, 100 , N, 107 , P,  92 , R, 105 , C, 112 , Q,  93 , S, 105 , A, 115 , T, 105 , W, 110
N, 102 , F, 110 , A,  93 , S, 105 , T, 115 , H,  98 , T, 105 , C, 117 , S, 113 , T, 113
E, 113 , G, 110 , K,  95 , V, 110 , R, 118 , C, 115 , Y, 110 , K, 120 , P, 115 , A, 118
Q, 113 , M, 110 , R, 115 , G, 120 , A, 130 , V, 120 , M, 112 , P, 120 , I, 122 , K, 122
H, 123 , H, 113 , C, 120 , A, 122 , K, 133 , M, 135 , V, 122 , Q, 122 , L, 127 , P, 122
Y, 130 , D, 120 , V, 132 , M, 133 , P, 133 , K, 138 , L, 133 , R, 130 , Y, 128 , R, 132
I, 137 , L, 132 , M, 143 , I, 150 , S, 138 , Y, 142 , I, 137 , S, 132 , V, 137 , C, 137
L, 143 , E, 137 , Y, 158 , C, 152 , Q, 140 , R, 155 , A, 138 , E, 140 , A, 143 , E, 142
W, 143 , Q, 137 , I, 160 , L, 153 , N, 142 , I, 163 , G, 138 , N, 143 , G, 143 , N, 145
F, 158 , W, 150 , L, 167 , Y, 157 , E, 153 , L, 168 , F, 153 , W, 147 , F, 148 , S, 153
K, 158 , R, 163 , W, 180 , F, 178 , G, 163 , W, 173 , C, 172 , D, 157 , W, 155 , G, 168
R, 177 , K, 170 , F, 183 , W, 190 , D, 190 , F, 182 , W, 175 , G, 170 , C, 175 , D, 172
'''.strip()
PMC2195959_table1b = '''
M,  Mv,  N,  Nv,  P,  Pv , Q,  Qv , R,  Rv , S,  Sv , T,  Tv , V,  Vv , W,  Wv , Y,  Yv
M,  10 , N,  10 , P,  10 , Q,  10 , R,  10 , S,  10 , T,  10 , V,  10 , W,  10 , Y,  10
L,  38 , D,  32 , T,  35 , E,  32 , K,  20 , T,  40 , P,  47 , L,  53 , F,  42 , F,  62
I,  50 , E,  45 , S,  60 , H,  33 , H,  58 , G,  42 , S,  50 , I,  55 , Y,  45 , W,  63
V,  52 , Q,  58 , A,  63 , N,  60 , Q,  67 , A,  47 , A,  57 , M,  55 , L,  63 , H,  73
F,  70 , H,  68 , H,  72 , K,  70 , E,  87 , P,  53 , N,  65 , P,  78 , M,  65 , M,  80
K,  97 , T,  72 , Q,  72 , D,  72 , N,  87 , N,  65 , D,  78 , T,  82 , R,  87 , L,  97
Q, 103 , S,  77 , N,  78 , R,  83 , M,  92 , D,  80 , E,  83 , F,  85 , I,  92 , T, 100
R, 105 , K,  85 , D,  92 , P,  85 , D, 100 , C,  93 , G,  88 , A, 100 , H, 102 , I, 103
Y, 108 , P,  87 , E,  95 , M, 105 , P, 102 , E,  93 , H, 105 , C, 107 , V, 103 , C, 107
H, 110 , R, 105 , G, 102 , T, 108 , W, 103 , H, 102 , Q, 105 , H, 110 , K, 105 , V, 110
A, 112 , A, 107 , V, 112 , V, 113 , S, 113 , Q, 105 , V, 106 , Q, 110 , Q, 123 , P, 115
T, 113 , G, 118 , C, 114 , S, 123 , T, 120 , K, 123 , C, 122 , E, 113 , S, 123 , A, 117
P, 118 , V, 123 , M, 122 , A, 130 , I, 125 , V, 135 , K, 122 , N, 123 , A, 125 , N, 117
W, 127 , C, 133 , R, 132 , G, 133 , L, 130 , R, 137 , M, 123 , S, 127 , P, 127 , Q, 120
C, 130 , M, 138 , K, 135 , L, 138 , Y, 130 , M, 145 , I, 137 , G, 132 , T, 127 , S, 122
E, 132 , Y, 152 , Y, 135 , I, 142 , V, 137 , Y, 147 , Y, 143 , Y, 140 , N, 132 , K, 127
N, 142 , I, 160 , I, 160 , Y, 150 , F, 143 , I, 160 , R, 155 , D, 148 , C, 133 , R, 130
S, 150 , L, 165 , L, 163 , C, 163 , A, 152 , W, 165 , W, 163 , K, 152 , E, 157 , E, 142
G, 165 , W, 177 , W, 163 , F, 173 , G, 152 , L, 175 , L, 165 , W, 157 , G, 167 , G, 147
D, 168 , F, 183 , F, 178 , W, 175 , C, 173 , F, 183 , F, 177 , R, 163 , D, 173 , D, 160
'''.strip()

def create_PMC2195959_table():
    table1a = pd.read_csv(io.StringIO(PMC2195959_table1a.strip().replace(' ', '')), header=0)
    print(table1a)
    table1b = pd.read_csv(io.StringIO(PMC2195959_table1b.strip().replace(' ', '')), header=0)
    assert table1a.shape == table1b.shape, F'Tables 1a and 1b are not equal in shapes, {table1a.shape} == {table1b.shape} failed!'
    table1 = pd.concat([table1a, table1b], axis=1)
    aas_observed = [x for x in table1.columns if len(x) == 1]
    aas_expected = sorted([aa for aa in AAS])
    assert aas_observed == aas_expected, F'{aas_observed} == {aas_expected} failed for the header!'
    for colidx, colname in enumerate(table1.columns):
        if len(colname) == 2:
            assert (table1[colname] == sorted(table1[colname])).all(), F'The column {colname} is not sorted ({table1[colname]} == {sorted(table1[colname])})!'
        else:
            assert len(colname) == 1, F'The column {colname} has invalid name!'
            aas_observed = sorted(table1[colname])
            assert aas_observed == aas_expected, F'{aas_observed} == {aas_expected} failed for the column {colname}!'
    assert len(table1.columns) == 20*2, F'{len(table1.columns)} == {20*2} failed!'
    aa2conserved     = collections.defaultdict(list)
    aa2semiconserved = collections.defaultdict(list)
    aa2nonconserved  = collections.defaultdict(list)
    for aa1 in AAS:
        for aa2, aa2v in zip(table1[aa1], table1[aa1+'v']):
            if   aa2v <=  70: aa2conserved    [aa1].append(aa2)
            elif aa2v <= 130: aa2semiconserved[aa1].append(aa2)
            else:             aa2nonconserved [aa1].append(aa2)
            assert aa2v <= 200, F'{aa2v} <= 200 failed for {aa1}->{aa2}!'
    return table1, aa2conserved, aa2semiconserved, aa2nonconserved

PMC2195959_table, PMC2195959_aa2conserved, PMC2195959_aa2semiconserved, PMC2195959_aa2nonconserved = create_PMC2195959_table()

def norm_mhc(mhc): return mhc.replace('*', '').replace('HLA-', '').replace(':', '').replace('-', '')

def add_dash_foreach_startline(string, tokens):
    ret = string
    for token in tokens:
        if ret.startswith(token): ret = ret.replace(token, token+'-', 1)
    return ret
def load_motif(mhc, motiflen, motif_dir):
    #file = '/mnt/d/code/neoguider/software/prime/MixMHCpred/lib/pwm/class1_9/PWM_H2-Kb_1.csv'
    mhc2 = add_dash_foreach_startline(norm_mhc(mhc), ['H2', 'BoLA', 'DLA', 'Mamu'])
    #filename = F'/mnt/d/code/neoguider/software/prime/MixMHCpred/lib/pwm/class1_{motiflen}/PWM_{mhc2}_1.csv'
    filename = F'{motif_dir}/class1_{motiflen}/PWM_{mhc2}_1.csv'
    df = pd.read_csv(filename, index_col=0)
    return df
def pep2info(peptide, motif_df):
    assert len(peptide) >= 7 and len(peptide) <= 15, F'The peptide {peptide} has invalid length (not between 7 and 15)!'
    assert len(peptide) == len(motif_df.columns) #- 1
    ret = 0
    for i, aa in enumerate(peptide.upper()):
        ret += motif_df.loc[aa,str(i+1)]
    return ret

#import copy
def modify_string(aaseq, listof_posAAs):
    ret = []
    for posAAs in listof_posAAs:
        pos = int(posAAs[0])
        for AA in posAAs[1:]:
            aaseq2 = [x for x in aaseq]
            aaseq2[pos-1] = AA
            aaseq2 = ''.join(aaseq2)
            ret.append(aaseq2)
    return ret

def expand_string(aaseq, positions, alphabet=AAS):
    assert len(AAS) == 20
    assert len(set(AAS)) == len(AAS)
    
    if positions == 'all': positions = [(x+1) for x in range(len(aaseq))]
    if isinstance(positions, int): positions = [positions]
    
    ret = []
    for position in positions:
        zerobased_pos = position - 1
        for aa in AAS: #alphabet:
            aaseq2 = [aa2 for aa2 in aaseq]
            old_aa, new_aa = aaseq2[zerobased_pos], aa
            if aa in alphabet or (new_aa in PMC2195959_aa2conserved[old_aa]):
                aaseq2[zerobased_pos] = aa
                ret.append(''.join(aaseq2))
    assert len(ret) - len(positions) + 1 == len(set(ret))
    return sorted(list(set(ret)))

def expand_substrings(aaseq, positions, substrlens=9):
    if positions == 'all': positions = [(x+1) for x in range(len(aaseq))]
    if isinstance(positions, int): positions = [positions]
    if isinstance(substrlens, int): substrlens = [substrlens]
    
    ret = []    
    for i in range(len(aaseq)):
        positions1 = [(x - i) for x in positions]
        for substrlen in substrlens:
            j = i+substrlen
            if j <= len(aaseq):
                aaseq2 = aaseq[i:j]
                positions2 = [x for x in positions1 if 1 <= x and x <= len(aaseq2)]
                logging.info(F'In expand_substrings: expand_string({aaseq2}, {positions2})')
                if positions2 != []:
                    ret.extend(expand_string(aaseq2, positions2))
    return ret

def warp_roc_auc(n): return n + '_AUROC'

def fixed_pos(n): return n


METHODS = [
        'MixMHCPred_aff',
        'MixMHCpred_motif', 
        
        'PRIME_immunogenicity',
        
        'MHCmotifAtlas_motif', 
        'MHCmotifAtlas_motif_diff',

        'NetMHCpan_aff', 
        'NetMHCpan_aff_ratio', 
        'netMHCpan_EL', 

        'MHCflurry_aff', 
        'MHCflurry_presentation', 
        
        'Dist_bit', 
        'Dist_conserv']

def fill_vals(df3, HLA, motif_dir):
    
    hlalen2motifdf = {}
    for pep in list(df3['ET_pep']):
        if (HLA, len(pep)) not in hlalen2motifdf:
            hlalen2motifdf[(HLA, len(pep))] = load_motif(HLA, len(pep), motif_dir)
    infos = []
    for pep in list(df3['ET_pep']):
        motifdf = hlalen2motifdf[(HLA, len(pep))]
        info = pep2info(pep, motifdf)
        infos.append(info)
    df3['ETinfoFromMMP'] = infos

    aa_dists = []
    for et_pep, mt_pep in zip(list(df3['ET_pep']), list(df3['MT_pep'])):
        #assert len(et_pep) == len(mt_pep), F'{et_pep} and {mt_pep} have diff lengths!'
        if not (len(et_pep) == len(mt_pep)): 
            logging.critical(F'{et_pep} and {mt_pep} have diff lengths!')
            aa_dists.append(2**10+1)
            continue
        aa_substitutions = []
        for i in range(len(et_pep)):
            if et_pep[i].upper() != mt_pep[i].upper():
                aa_substitutions.append(et_pep[i].upper() + mt_pep[i].upper())
        aa_dist = 0
        for aa_substitution in aa_substitutions:
            aa_from, aa_into = aa_substitution[0], aa_substitution[1]
            if   aa_into in PMC2195959_aa2conserved[aa_from]:
                aa_dist += 1
            elif aa_into in PMC2195959_aa2semiconserved[aa_from]:
                aa_dist += 2
            elif aa_into in PMC2195959_aa2nonconserved[aa_from]:
                aa_dist += 3
        aa_dists.append(aa_dist)
    df3['ET_MT_conservDist'] = aa_dists

    df3['Identity3'] = df3['Identity'].str.replace('_', '/')
    df3 = df3.sort_values('Identity3') # 
    
    df3['MixMHCPred_aff']        = df3['PRIME_BArank'].rank(method='first', ascending=True)
    df3['MixMHCpred_motif']      = df3['ETinfoFromMMP'].rank(method='first', ascending=False)    
    
    df3['PRIME_immunogenicity']  = df3['PRIME_rank'].rank(method='first', ascending=True)

    df3['MHCmotifAtlas_motif']   = df3['ETinfo'].rank(method='first', ascending=False)
    df3['MHCmotifAtlas_motif_diff']    = (df3['ETinfo']-df3['MTinfo']).rank(method='first', ascending=False)

    df3['NetMHCpan_aff']         = df3['ET_BindAff'].rank(method='first', ascending=True)
    df3['NetMHCpan_aff_ratio']   = (df3['ET_BindAff']/df3['MT_BindAff']).rank(method='first', ascending=True)
    df3['netMHCpan_EL']          = df3['%Rank_EL'].rank(method='first', ascending=True)

    df3['MHCflurry_aff']          = df3['mhcflurry_aff_percentile'].rank(method='first', ascending=True)
    df3['MHCflurry_presentation'] = df3['mhcflurry_presentation_percentile'].rank(method='first', ascending=True)
    
    df3['Dist_bit']              = df3['ET_MT_bitDist'].rank(method='first', ascending=True) 
    df3['Dist_conserv']          = df3['ET_MT_conservDist'].rank(method='first', ascending=True) 
    
    return df3

# --- Function to add asterisks based on p-value ---
def add_asterisks(p_val):
    if p_val < 0.001:
        return "***"
    elif p_val < 0.01:
        return "**"
    elif p_val < 0.05:
        return "*"
    else:
        return ""  # No asterisk if p ≥ 0.05

def heatmap2annotations(df5heatmap, df5pvalues):
    # --- Create annotations (value + asterisks) ---
    annotations = df5heatmap.round(3).astype(str)  # Base annotation (just the value)
    for i in range(df5pvalues.shape[0]):
        for j in range(df5pvalues.shape[1]):
            p_val = df5pvalues.iloc[i, j]
            asterisks = add_asterisks(p_val)
            annotations.iloc[i, j] = f"{df5heatmap.iloc[i, j]:.3f}{asterisks}"
    return annotations

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(pathname)s:%(lineno)d %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(description='Take as input *.expansion.untraced file (from stdin) and output performance metrics (to stdout). ', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #/mnt/d/heteroclitic/v01-hpep.outdir/prioritization/v02_prelim_features_from_pmhcs.tsv.expansion.untraced
    parser.add_argument('-f', '--fasta',  help='Input fasta file. ', nargs='+')
    parser.add_argument('-i', '--input',  help='Input *.expansion.untraced TSV file. ', nargs='+')
    parser.add_argument('-o', '--output', help='Output rank TSV file. ')
    parser.add_argument('-m', '--motif', help='Motif directory. ', default=F'/mnt/d/code/neoguider/software/prime/MixMHCpred/lib/pwm/')

    args = parser.parse_args()    
    assert len(args.fasta) == len(args.input), F'The --fasta and --input options must take the some number of cmd-line params.'

    logging.info(f'Parser args')
    mydict = collections.defaultdict(list)
    for args_fasta, args_input in zip(args.fasta, args.input):
        with open(args_fasta) as infile:
            keys = ['HLA', 'TPM', 'gene', 'ACC']
            for line in infile:
                line = line.strip()
                if line.startswith('>'):
                    fastaID = line[1:].split()[0]
                    if fastaID.startswith('WILD_') or fastaID.startswith('SELF_'): continue
                    mydict['Identity'].append(fastaID)
                    mydict2 = collections.defaultdict(str)
                    for tok in line.split():
                        if len(tok.split('=')) == 2:
                            key, val = tok.split('=')
                            mydict2[key] = val
                    for key in keys: mydict[key].append(mydict2[key])
    fasta_df1 = pd.DataFrame(mydict)
    dfs = []
    for args_fasta, args_input in zip(args.fasta, args.input):
        df1 = pd.read_csv(args_input, sep='\t', header=0)        
        if dfs: assert list(dfs[0].columns) == list(df1.columns), F'{dfs[0].columns} == {df1.columns} failed because the input file {args_input} has a different set of column names. '
        dfs.append(df1)
    df1 = pd.concat(dfs)
    #print(df1)
    logging.info(f'Constructed df1')
    # dataIDs2ntimes = collections.defaultdict(int)
    
    authorTxts, geneTxts, hlaTxts, pepTxts = set(), set(), set(), set()

    author2ntimes = collections.defaultdict(int)
    gene2ntimes = collections.defaultdict(int)
    pep2ntimes = collections.defaultdict(int)
    hla2ntimes = collections.defaultdict(int)

    WT1_trial = 'Maslak et al.' # 'NCT01266083' # 'NCT01266083_and_NCT04229979'
    # difficulties (higher means harder)
    BY_MHC, BY_TCR, BY_BOTH = 0, 1, 0.5
    FROM_TRIAL_1, FROM_TRIAL_2, FROM_TRIAL_3, FROM_TRIAL_4, FROM_PAPER, FROM_PREPRINT = -1,-2,-3,-4,1,2
    FROM_TRIAL = FROM_TRIAL_3
    IN_MOUSE, IN_HUMAN = 0, 1
    WITH_COMMON_MHC, WITH_RARE_MHC = 0, 1
    AT_EVERY_POS, AT_FIXED_POS = 0, 1

    WT1_YEAR = 2018
    def stats2order(by_, from_, in_, with_, at_, year, *vargs): return (by_, year, (1 if at_ else 0), from_, in_, with_, *vargs)
    geneHLA_to_posneg_peplist = {

# pubyear=2001, PMC2195959
'b01.CEA_691.HLA-A02:01.IMIGVLVGV_from-table1': [
    (BY_TCR, FROM_PAPER, IN_HUMAN, WITH_COMMON_MHC, AT_EVERY_POS, 2001),
    ('Tangri et al.', 'HLA-A0201', 'CEA_691', 'One~mismatch~from~IMIGVLVGV,~selected~from~Tangri~et~al.', 'TAA'),
    modify_string('IMIGVLVGV', ['3M', '5H']),
    expand_string('IMIGVLVGV', [1,3,4,5,6,7,8], alphabet=PMC2195959_17aas)
],
'b01.MAGEA3_112.HLA-A02:01.KVAELVHFL_from-table1': [
    (BY_TCR, FROM_PAPER, IN_HUMAN, WITH_COMMON_MHC, AT_EVERY_POS, 2001),
    ('Tangri et al.', 'HLA-A0201', 'MAGEA3_112', 'One~mismatch~from~KVAELVHFL,~selected~from~Tangri~et~al.', 'TAA'),
    modify_string('KVAELVHFL', ['5I', '7W']),
    expand_string('KVAELVHFL', [1,3,4,5,6,7,8], alphabet=PMC2195959_17aas)
],
'b01.MAGEA2_157.HLA-A02:01.YLQLVFGIEV_from-table1': [
    (BY_TCR, FROM_PAPER, IN_HUMAN, WITH_COMMON_MHC, AT_EVERY_POS, 2001),
    ('Tangri et al.', 'HLA-A0201', 'MAGEA2_157', 'One~mismatch~from~YLQLVFGIEV,~selected~from~Tangri~et~al.', 'TAA'),
    modify_string('YLQLVFGIEV', ['5IF']),
    expand_string('YLQLVFGIEV', [1,3,4,5,6,7,8,9], alphabet=PMC2195959_17aas)
],
'b01.HBV_Pol_455.HLA-A02:01.GLSRYVARL_from-table1': [
    (BY_TCR, FROM_PAPER, IN_HUMAN, WITH_COMMON_MHC, AT_EVERY_POS, 2001),
    ('Tangri et al.', 'HLA-A0201', 'HBV_Pol_455', 'One~mismatch~from~GLSRYVARL,~selected~from~Tangri~et~al.', 'Viral'),
    modify_string('GLSRYVARL', ['7P']),
    expand_string('GLSRYVARL', [1,3,4,5,6,7,8], alphabet=PMC2195959_17aas)
],
'b01.HIV_Pol_476.HLA-A02:01.ILKEPVHGV_from-table1': [
    (BY_TCR, FROM_PAPER, IN_HUMAN, WITH_COMMON_MHC, AT_EVERY_POS, 2001),
    ('Tangri et al.', 'HLA-A0201', 'HIV_Pol_476', 'One~mismatch~from~ILKEPVHGV,~selected~from~Tangri~et~al.', 'Viral'),
    modify_string('ILKEPVHGV', ['3HL']),
    expand_string('ILKEPVHGV', [1,3,4,5,6,7,8], alphabet=PMC2195959_17aas)
],

'##b01.CEA_691.HLA-A02:01.IMIGVLVGV_positions-all-nonanchor': [
    (BY_TCR, FROM_PAPER, IN_HUMAN, WITH_COMMON_MHC, AT_EVERY_POS, 2001),
    ('Tangri et al.', 'HLA-A0201', 'CEA_691', 'One~mismatch~from~IMIGVLVGV~at~nonanchor~positions', 'TAA'),
    modify_string('IMIGVLVGV', ['3M', '4L', '4P', '5H','5L','6H','6T','7I']),
    expand_string('IMIGVLVGV', [1,3,4,5,6,7,8], alphabet=PMC2195959_17aas)
],
'##b01.MAGEA3_112.HLA-A02:01.IMIGVLVGV_positions-all-nonanchor': [
    (BY_TCR, FROM_PAPER, IN_HUMAN, WITH_COMMON_MHC, AT_EVERY_POS, 2001),
    ('Tangri et al.', 'HLA-A0201', 'MAGEA3_112', 'One~mismatch~from~KVAELVHFL~at~nonanchor~positions', 'TAA'),
    modify_string('KVAELVHFL', ['5I', '7WY']),
    expand_string('KVAELVHFL', [1,3,4,5,6,7,8], alphabet=PMC2195959_17aas)
],
'##b01.MAGEA2_157.HLA-A02:01.YLQLVFGIEV_positions-all-nonanchor': [
    (BY_TCR, FROM_PAPER, IN_HUMAN, WITH_COMMON_MHC, AT_EVERY_POS, 2001),
    ('Tangri et al.', 'HLA-A0201', 'MAGEA2_157', 'One~mismatch~from~YLQLVFGIEV~at~nonanchor~positions', 'TAA'),
    modify_string('YLQLVFGIEV', ['1F','3HKEN','5IFTY','7N','8M']),
    expand_string('YLQLVFGIEV', [1,3,4,5,6,7,8,9], alphabet=PMC2195959_17aas)
],
'##b01.HBV_Pol_455.HLA-A02:01.GLSRYVARL_positions-all-nonanchor': [
    (BY_TCR, FROM_PAPER, IN_HUMAN, WITH_COMMON_MHC, AT_EVERY_POS, 2001),
    ('Tangri et al.', 'HLA-A0201', 'HBV_Pol_455', 'One~mismatch~from~GLSRYVARL~at~nonanchor~positions', 'Viral'),
    modify_string('GLSRYVARL', ['3N','5F','6I','7P']),
    expand_string('GLSRYVARL', [1,3,4,5,6,7,8], alphabet=PMC2195959_17aas)
],
'##b01.HIV_Pol_476.HLA-A02:01.ILKEPVHGV_positions-all-nonanchor': [
    (BY_TCR, FROM_PAPER, IN_HUMAN, WITH_COMMON_MHC, AT_EVERY_POS, 2001),
    ('Tangri et al.', 'HLA-A0201', 'HIV_Pol_476', 'One~mismatch~from~ILKEPVHGV~at~nonanchor~positions', 'Viral'),
    modify_string('ILKEPVHGV', ['3HLD','8T']),
    expand_string('ILKEPVHGV', [1,3,4,5,6,7,8], alphabet=PMC2195959_17aas)
],

# pubyear=2007, 10.1016/j.vaccine.2007.05.008 (EpitOptimizer), from Fig. 4 at https://www.sciencedirect.com/science/article/abs/pii/S0264410X07005646
'a3.TYRP1.H-2-Db.TYRP1-from-EpitOptimizer': [
    (BY_MHC, FROM_PAPER, IN_MOUSE, WITH_COMMON_MHC, AT_EVERY_POS, 2007),
    ('Houghton et al.', 'H2-Db', 'TYRP1', 'Selected~by~EpitOptimizer', 'TAA'),
    [
        x.strip() for x in '''
        RMCANIEAL
        RGVCNPDLL
        ISVYNYFVL
        FSHENPAFL
        GSRSNFDSL
        RMLHNLAHL
        THLSNNDPI
        TAPDNLGYL
        IAVVNALLL
        AALLNVAAI
        '''.strip().split() if not x.startswith('#')
    ],
    [
        x.strip() for x in '''
        MMSYNVLPL
        GMACNQKIL
        TMRRNLLDL
        '''.strip().split() if not x.startswith('#')
    ]
],

# pubyear=2007, 10.1016/j.vaccine.2007.05.008 (EpitOptimizer), from Fig. 5 at https://www.sciencedirect.com/science/article/abs/pii/S0264410X07005646
'a3.PSMA.HLA-A02:01.PSMA-from-EpitOptimizer': [
    (BY_MHC, FROM_PAPER, IN_HUMAN, WITH_COMMON_MHC, AT_EVERY_POS, 2007),
    ('Houghton et al.', 'HLA-A0201', 'PSMA', 'Selected~by~EpitOptimizer', 'TAA'),
    [
        x.strip() for x in '''
        YLFTQIPHL #N76L
        VLFDSLFSA #S628L
        MLNDQLMFL #M664L
        '''.strip().split() if not x.startswith('#')
    ],
    [
        x.strip() for x in '''
        YLHETDSAV #L4Y
        YLAGGFFLL #V27L
        LLFLFGWFI #G36L
        SLFEPPPPV #G150V
        ALTEDFFKL #R181L
        KLYPDGWNL #S241L
        KLLEKMGGV #S312V
        LLGSTEWAV #E436V
        GLYDALFDI #I708L
        QLYVAAFTV #I732L
        '''.strip().split() if not x.startswith('#')
    ]
],

# pubyear=2021, PMC8201969, from the understanding of the full text
'b3.AH1.H-2-Ld.SPSYVYHQF_positions-all': [
    (BY_TCR, FROM_PAPER, IN_MOUSE, WITH_COMMON_MHC, AT_EVERY_POS, 2021),
    ('Wei et al.', 'H2-LD', 'AH1', 'One~mismatch~from~SPSYVYHQF', 'Viral'),
    [x.strip().upper() for x in '''SPSYaYHQF'''.strip().split() if not x.startswith('#')],
    expand_string('SPSYVYHQF', 'all')
],
'b3.AH1.H-2-Ld.SPSYVYHQF_position-5': [
    (BY_TCR, FROM_PAPER, IN_MOUSE, WITH_COMMON_MHC, fixed_pos(5), 2021),
    ('Wei et al.', 'H2-LD', 'AH1', 'One~mismatch~from~SPSYVYHQF~at~position~5', 'Viral'),
    [x.strip().upper() for x in '''SPSYaYHQF'''.strip().split() if not x.startswith('#')],
    expand_string('SPSYVYHQF', 5)
],

# The heteroclitic variants that were not used in the clinical trial are excluded
# https://doi.org/10.1182/bloodadvances.2017014175,uniprotkb/P19544,https://www.tandfonline.com/doi/full/10.1080/21645515.2023.2296735#d1e152,NCT04229979
# PMID: 39606837 PMCID: PMC11760237 (available on 2025-11-28)
'a1.WT1.HLA-A02:01.SQASSGQARMFPNAPYL_positions-all': [
    (BY_MHC, FROM_TRIAL, IN_HUMAN, WITH_COMMON_MHC, AT_EVERY_POS, WT1_YEAR),
    # 'Maslak et al. 2018, Ogasawara et al., and Jamy et al.'
    (WT1_trial, 'HLA-A0201', 'WT1', 'One~mismatch~from~SQASSGQARMFPNAPYL', 'TAA'),
    [x.strip().upper() for x in '''yMFPNAPYL'''.strip().split() if not x.startswith('#')],
    expand_substrings('SQASSGQARMFPNAPYL', 'all', 9)
],
'a1.WT1.HLA-A02:01.SQASSGQARMFPNAPYL_position-9': [
    (BY_MHC, FROM_TRIAL, IN_HUMAN, WITH_COMMON_MHC, fixed_pos(9), WT1_YEAR),
    (WT1_trial, 'HLA-A0201', 'WT1', 'One~mismatch~from~SQASSGQARMFPNAPYL~at~position~9', 'TAA'),
    [x.strip().upper() for x in '''yMFPNAPYL'''.strip().split() if not x.startswith('#')],
    expand_substrings('SQASSGQARMFPNAPYL', 9, 9)
],
'a1.WT1.HLA-A02:01.RMFPNAPYLL_positions-all': [
    (BY_MHC, FROM_TRIAL, IN_HUMAN, WITH_COMMON_MHC, AT_EVERY_POS, WT1_YEAR),
    (WT1_trial, 'HLA-A0201', 'WT1', 'One~mismatch~from~RMFPNAPYLL', 'TAA'),
    [x.strip().upper() for x in '''yMFPNAPYL'''.strip().split() if not x.startswith('#')],
    expand_string('RMFPNAPYL', 'all')
],
'a1.WT1.HLA-A02:01.RMFPNAPYLL_position-1': [
    (BY_MHC, FROM_TRIAL, IN_HUMAN, WITH_COMMON_MHC, fixed_pos(1), WT1_YEAR),
    (WT1_trial, 'HLA-A0201', 'WT1', 'One~mismatch~from~RMFPNAPYLL~at~position~1', 'TAA'),
    [x.strip().upper() for x in '''yMFPNAPYL'''.strip().split() if not x.startswith('#')],
    expand_string('RMFPNAPYL', 1)
],
'a1.WT1.HLA-A24:02.SQASSGQARMFPNAPYL_positions-all': [
    (BY_MHC, FROM_TRIAL, IN_HUMAN, WITH_RARE_MHC, AT_EVERY_POS, WT1_YEAR),
    (WT1_trial, 'HLA-A2402', 'WT1', 'One~mismatch~from~SQASSGQARMFPNAPYL', 'TAA'),
    [x.strip().upper() for x in '''yMFPNAPYL'''.strip().split() if not x.startswith('#')],
    expand_substrings('SQASSGQARMFPNAPYL', 'all', 9)
],
'a1.WT1.HLA-A24:02.SQASSGQARMFPNAPYL_position-9': [
    (BY_MHC, FROM_TRIAL, IN_HUMAN, WITH_RARE_MHC, fixed_pos(9), WT1_YEAR),
    (WT1_trial, 'HLA-A2402', 'WT1', 'One~mismatch~from~SQASSGQARMFPNAPYL~at~position~9', 'TAA'),
    [x.strip().upper() for x in '''yMFPNAPYL'''.strip().split() if not x.startswith('#')],
    expand_substrings('SQASSGQARMFPNAPYL', 9, 9)
],
'a1.WT1.HLA-A24:02.RMFPNAPYLL_positions-all': [
    (BY_MHC, FROM_TRIAL, IN_HUMAN, WITH_RARE_MHC, AT_EVERY_POS, WT1_YEAR),
    (WT1_trial, 'HLA-A2402', 'WT1', 'One~mismatch~from~RMFPNAPYLL', 'TAA'),
    [x.strip().upper() for x in '''yMFPNAPYL'''.strip().split() if not x.startswith('#')],
    expand_string('RMFPNAPYL', 'all')
],
'a1.WT1.HLA-A24:02.RMFPNAPYLL_position-1': [
    (BY_MHC, FROM_TRIAL, IN_HUMAN, WITH_RARE_MHC, fixed_pos(1), WT1_YEAR),
    (WT1_trial, 'HLA-A2402', 'WT1', 'One~mismatch~from~RMFPNAPYLL~at~position~1', 'TAA'),
    [x.strip().upper() for x in '''yMFPNAPYL'''.strip().split() if not x.startswith('#')],
    expand_string('RMFPNAPYL', 1)
],

# The non-investigated heteroclitic variants are excluded
# https://www.science.org/doi/10.1126/scitranslmed.aba4380
'a2.CALR_MUT.H-2-Kb.RRMMRTK-LQGWTEA_positions-all': [
    (BY_MHC, FROM_TRIAL_1, IN_MOUSE, WITH_COMMON_MHC, AT_EVERY_POS, 2022),
    ('Giroux et al.', 'H2-Kb', 'CALR_MUT', 'One~mismatch~from~RRMMRTK[..]LQGWTEA', 'Neo'),
    [x.strip().upper() for x in '''RMMRfKMRM'''.strip().split() if not x.startswith('#')],
    expand_substrings('RRMMRTKMRMRRMRRTRRKMRRKMSPARPRTSCREACLQGWTEA', 'all', 9)
],
'a2.CALR_MUT.H-2-Kb.RRMMRTK-LQGWTEA_position-6': [
    (BY_MHC, FROM_TRIAL_1, IN_MOUSE, WITH_COMMON_MHC, fixed_pos(6), 2022),
    ('Giroux et al.', 'H2-Kb', 'CALR_MUT', 'One~mismatch~from~RRMMRTK[..]LQGWTEA~at~position~6', 'Neo'),
    [x.strip().upper() for x in '''RMMRfKMRM'''.strip().split() if not x.startswith('#')],
    expand_substrings('RRMMRTKMRMRRMRRTRRKMRRKMSPARPRTSCREACLQGWTEA', 6, 9)
],
'a2.CALR_MUT.H-2-Kb.RMMRTKMRM_positions-all': [
    (BY_MHC, FROM_TRIAL_1, IN_MOUSE, WITH_COMMON_MHC, AT_EVERY_POS, 2022),
    ('Giroux et al.', 'H2-Kb', 'CALR_MUT', 'One~mismatch~from~RMMRTKMRM', 'Neo'),
    [x.strip().upper() for x in '''RMMRfKMRM'''.strip().split() if not x.startswith('#')],
    expand_string('RMMRTKMRM', 'all')
],
'a2.CALR_MUT.H-2-Kb.RMMRTKMRM_position-5': [
    (BY_MHC, FROM_TRIAL_1, IN_MOUSE, WITH_COMMON_MHC, fixed_pos(5), 2022),
    ('Giroux et al.', 'H2-Kb', 'CALR_MUT', 'One~mismatch~from~RMMRTKMRM~at~position~5', 'Neo'),
    [x.strip().upper() for x in '''RMMRfKMRM'''.strip().split() if not x.startswith('#')],
    expand_string('RMMRTKMRM', 5)
],

# 10.1158/2326-6066.CIR-21-0332 : Fig. 1 at https://aacrjournals.org/cancerimmunolres/article/10/3/314/681732/An-In-Vivo-Screen-to-Identify-Short-Peptide
'b3.TRP2.H-2-Kb.SVYDFFVWL_positions-all': [
    (BY_TCR, FROM_PAPER, IN_MOUSE, WITH_COMMON_MHC, AT_EVERY_POS, 2022, -1),
    ('He et al.', 'H2-Kb', 'TRP2', 'One~mismatch~from~SVYDFFVWL', 'TAA'),
    [
        x.strip().upper() for x in '''
        SVYDFFVcL
        '''.strip().split() if not x.startswith('#')
    ],
    expand_string('SVYDFFVWL', 'all')
],
'b3.TRP2.H-2-Kb.SVYDFFVWL_position-8': [
    (BY_TCR, FROM_PAPER, IN_MOUSE, WITH_COMMON_MHC, fixed_pos(8), 2022, -1),
    ('He et al.', 'H2-Kb', 'TRP2', 'One~mismatch~from~SVYDFFVWL~at~position~8', 'TAA'),
    [
        x.strip().upper() for x in '''
        SVYDFFVcL
        '''.strip().split() if not x.startswith('#')
    ],
    expand_string('SVYDFFVWL', 8)
],

# 10.1158/2326-6066.CIR-21-0332 : Fig. S9 at https://aacrjournals.org/cancerimmunolres/article/10/3/314/681732/An-In-Vivo-Screen-to-Identify-Short-Peptide
'b2.gp70/AH1.H-2-Ld.SPSYVYHQF_positions-1-3-5-8': [
    (BY_TCR, FROM_PAPER, IN_MOUSE, WITH_COMMON_MHC, AT_EVERY_POS, 2022, -19),
    ('He et al.', 'H2-Ld', 'gp70/AH1', 'One~mismatch~from~SPSYVYHQF~at~positions~1,~3,~5,~and_8', 'Viral'),
    [
        x.strip().upper() for x in '''
        cPSYVYHQF
        qPSYVYHQF
        hPSYVYHQF
        iPSYVYHQF
        lPSYVYHQF
        mPSYVYHQF
        fPSYVYHQF       
        wPSYVYHQF
        yPSYVYHQF
        vPSYVYHQF
        SPeYVYHQF
        SPhYVYHQF
        SPpYVYHQF
        SPSYaYHQF
        SPSYcYHQF
        SPSYgYHQF
        SPSYtYHQF
        SPSYVYHnF
        SPSYVYHdF
        SPSYVYHhF
       #SVYDFFVxL #fromFig1
        '''.strip().split() if not x.startswith('#')
    ],
    expand_string('SPSYVYHQF', [1,3,5,8])
],
'b2.gp70/AH1.H-2-Ld.SPSYVYHQF_position-1': [
    (BY_TCR, FROM_PAPER, IN_MOUSE, WITH_COMMON_MHC, fixed_pos(1), 2022, -19, 1),
    ('He et al.', 'H2-Ld', 'gp70/AH1', 'One~mismatch~from~SPSYVYHQF~at~position~1', 'Viral'),
    [
        x.strip().upper() for x in '''
        cPSYVYHQF
        qPSYVYHQF
        hPSYVYHQF
        iPSYVYHQF
        lPSYVYHQF
        mPSYVYHQF
        fPSYVYHQF
        wPSYVYHQF
        yPSYVYHQF
        vPSYVYHQF
        '''.strip().split() if not x.startswith('#')
    ], 
    expand_string('SPSYVYHQF', 1)
],
'b2.gp70/AH1.H-2-Ld.SPSYVYHQF_position-3': [
    (BY_TCR, FROM_PAPER, IN_MOUSE, WITH_COMMON_MHC, fixed_pos(3), 2022, -19, 3),
    ('He et al.', 'H2-Ld', 'gp70/AH1', 'One~mismatch~from~SPSYVYHQF~at~position~3', 'Viral'),
    [
        x.strip().upper() for x in '''
        SPeYVYHQF
        SPhYVYHQF
        SPpYVYHQF
        '''.strip().split() if not x.startswith('#')
    ], 
    expand_string('SPSYVYHQF', 3)
],
'b2.gp70/AH1.H-2-Ld.SPSYVYHQF_position-5': [
    (BY_TCR, FROM_PAPER, IN_MOUSE, WITH_COMMON_MHC, fixed_pos(5), 2022, -19, 5),
    ('He et al.', 'H2-Ld', 'gp70/AH1', 'One~mismatch~from~SPSYVYHQF~at~position~5', 'Viral'),
    [
        x.strip().upper() for x in '''
        SPSYaYHQF
        SPSYcYHQF
        SPSYgYHQF
        SPSYtYHQF
        '''.strip().split() if not x.startswith('#')
    ],
    expand_string('SPSYVYHQF', 5)
],
'b2.gp70/AH1.H-2-Ld.SPSYVYHQF_position-8': [
    (BY_TCR, FROM_PAPER, IN_MOUSE, WITH_COMMON_MHC, fixed_pos(8), 2022, -19, 8),
    ('He et al.', 'H2-Ld', 'gp70/AH1', 'One~mismatch~from~SPSYVYHQF~at~position~8', 'Viral'),
    [
        x.strip().upper() for x in '''
        SPSYVYHnF
        SPSYVYHdF
        SPSYVYHhF
        '''.strip().split() if not x.startswith('#')
    ],
    expand_string('SPSYVYHQF', 8)
],

# 10.1158/2326-6066.CIR-21-0332 : Fig. S12 at https://aacrjournals.org/cancerimmunolres/article/10/3/314/681732/An-In-Vivo-Screen-to-Identify-Short-Peptide
'b3.mWT1/WT1.H-2-Db.RMFPNAPYL_position-7': [
    (BY_TCR, FROM_PAPER, IN_MOUSE, WITH_COMMON_MHC, fixed_pos(7), 2022, -112),
    ('He et al.', 'H2-Db', 'mWT1/WT1', 'One~mismatch~from~RMFPNAPYL~at~position~7', 'TAA'),
    [
        x.strip().upper() for x in '''
        RMFPNAdYL
        RMFPNAeYL
        RMFPNAwYL
        '''.strip().split() if not x.startswith('#')
    ],
    expand_string('RMFPNAPYL', 7)
],
'b3.mWT1/WT1.H-2-Db.RMFPNAPYL_positions-7-8': [
    (BY_TCR, FROM_PAPER, IN_MOUSE, WITH_COMMON_MHC, AT_EVERY_POS, 2022, -112),
    ('He et al.', 'H2-Db', 'mWT1/WT1', 'One~mismatch~from~RMFPNAPYL~at~positions~7~and~8', 'TAA'),
    [
        x.strip().upper() for x in '''
        RMFPNAdYL
        RMFPNAeYL
        RMFPNAwYL
        '''.strip().split() if not x.startswith('#')
    ], 
    expand_string('RMFPNAPYL', [7, 8])
],

# https://doi.org/10.1158/2767-9764.CRC-23-0384
# PMC10986479: https://aacrjournals.org/view-large/figure/15636124/CRC-23-0384_fig2.png
'##b9.p15E.H-2-Kb.KSPWFTTL_position-3': [
    (BY_TCR, FROM_PAPER, IN_MOUSE, WITH_COMMON_MHC, fixed_pos(3), 2024),
    ('Zhou et al.', 'H2-Kb', 'p15E', 'One~mismatch~from~KSPWFTTL~at~position~3', 'TAA'),
    [
        x.strip().upper() for x in '''
        KScWFTTL
        KSmWFTTL
        '''.strip().split() if not x.startswith('#')
    ],
    expand_string('KSPWFTTL', 3)
],

# https://www.biorxiv.org/content/10.1101/2024.11.27.624796v1.full.pdf
'##b8.NY-ESO-1/CTAG1B.HLA-A02:01.SLLMWITQC_positions-all': [
    (BY_BOTH, FROM_PREPRINT, IN_HUMAN, WITH_COMMON_MHC, AT_EVERY_POS, 2024),
    ('Johansen', 'HLA-A0201', 'NY-ESO-1/CTAG1B', 'One~mismatch~from~SLLMWITQC', 'TAA'),
    [
        x.strip().upper() for x in '''
        SLLMWITQC
        SLLaWITQC
        SLLMWITQv
        '''.strip().split() if not x.startswith('#')
    ],
    expand_string('SLLMWITQC', 'all')
],
'##b8.NY-ESO-1/CTAG1B.HLA-A02:01.from-BinderDesign': [
    (BY_BOTH, FROM_PREPRINT, IN_HUMAN, WITH_COMMON_MHC, AT_EVERY_POS, 2024),
    ('Johansen', 'HLA-A0201', 'NY-ESO-1/CTAG1B', 'from~BinderDesign', 'TAA'),
    [
        x.strip().upper() for x in '''
        SLLMWITQC
        SLLaWITQC
        SLLMWITQv
        '''.strip().split() if not x.startswith('#')
    ],
    [
        x.strip().upper() for x in '''
        SLLdWITQC
        SLLeWITQC
        '''.strip().split() if not x.startswith('#')
    ],
],
'##b8.NY-ESO-1/CTAG1B.HLA-A02:01.from-BinderDesign-nonanchor': [
    (BY_TCR, FROM_PREPRINT, IN_HUMAN, WITH_COMMON_MHC, fixed_pos(-1), 2024),
    ('Johansen', 'HLA-A0201', 'NY-ESO-1/CTAG1B', 'from~BinderDesign~nonanchor', 'TAA'),
    [
        x.strip().upper() for x in '''
        SLLMWITQC
        SLLaWITQC
        '''.strip().split() if not x.startswith('#')
    ],
    [
        x.strip().upper() for x in '''
        SLLdWITQC
        SLLeWITQC
        '''.strip().split() if not x.startswith('#')
    ],
],


    }
    geneHLA_to_posneg_peplist = {k:v for k,v in geneHLA_to_posneg_peplist.items() if not k.startswith('##')}
    
    hlalen2motifdf = {}
    from sklearn.metrics import roc_auc_score
    dicts4df = []
    #dicts4heatmap = []
    logging.info(f'Start')
    dataID_ord = 0
    for dataID_orig, (geneHLA, _) in enumerate(sorted(geneHLA_to_posneg_peplist.items(), key=lambda x: (stats2order(*(x[1][0])), -len(x[1][2])))):
        try:
            hard_level, gene, HLA, loci = geneHLA.split('.') # (geneHLA.split('.')[0], geneHLA.split('.')[1])
        except ValueError as err:
            print(F'Illegal string: {geneHLA}')
            raise err
        #fasta_df2 = fasta_df1.loc[(fasta_df1['gene']==gene) & (fasta_df1['HLA']==HLA), :]
        fasta_df2 = fasta_df1.loc[(fasta_df1['gene']==gene), :]
        df2 = df1.loc[df1['Identity'].isin(fasta_df2['Identity']), :].copy()
        if len(df2) == 0:
            logging.warning(F'Skipping {geneHLA} because it is not found in the files {args.fasta} + {args.input}')
            continue
        '''
        df2['RankInf1']    = df2['ETinfo'].rank(method='first', ascending=False)
        df2['RankAff1']    = df2['ET_BindAff'].rank(method='first', ascending=True)
        df2['RankInfDif1'] = (df2['ETinfo']-df2['MTinfo']).rank(method='first', ascending=False)
        df2['RankAffDiv1'] = (df2['ET_BindAff']/df2['MT_BindAff']).rank(method='first', ascending=True)
        '''

        difficulty, descs, pos_peps, neg_peps = geneHLA_to_posneg_peplist[geneHLA]
        pos_pep_set = set(pos_peps)
        neg_peps = [p for p in neg_peps if p not in pos_pep_set]
        immuno_peps = pos_peps + neg_peps
        
        df3 = df1.loc[df1['ET_pep'].str.upper().isin(immuno_peps) & (df1['HLA_type'].apply(norm_mhc) == norm_mhc(HLA)), :].copy() #.groupby(['ET_pep', 'B']).first().reset_index()
        df3 = fill_vals(df3, HLA, motif_dir=args.motif)

        df3['IsTrue'] = df3['ET_pep'].str.upper().isin(pos_peps)
        #print(df3)
        if len(df3) <= 2:
            logging.warning(F'Skipping {geneHLA} because no records are found for it. ')
            continue
        
        assert len(set(immuno_peps)) == len(immuno_peps), F'The peptide list {collections.Counter(immuno_peps)} contains duplicates!'
        for pep in immuno_peps:
            assert pep.upper() in list(df3["ET_pep"].str.upper()), F'The pre-given peptide {pep.upper()} is not in the observed series {sorted(list(df3["ET_pep"].str.upper()))} for {geneHLA}'
        #assert len(df3) == len(immuno_peps), F'The rows\n{collections.Counter(list(df3["ET_pep"]))}\nof length {len(df3)}\nhas fewer peptides than the expected set of peptides\n{collections.Counter(immuno_peps)}\nof length {len(immuno_peps)}'        
        dataID_ord += 1
        pre_dataID = ('M' if BY_MHC == difficulty[0] else 'T') + str(difficulty[5]-2000).zfill(2) # + descs[2].split('-')[0]
        #pre2_dataID = pre_dataID + 'p' + str(1 if difficulty[4] else 0)
        print(F'descs={descs} loci={loci}')
        authorTxt, geneTxt, hlaTxt, pepTxt = descs[0], descs[2], descs[1], loci.split('_')[0]

        authorTxts.add(authorTxt)
        geneTxts.add(geneTxt)
        pepTxts.add(pepTxt)
        hlaTxts.add(hlaTxt)

        if authorTxt not in author2ntimes: author2ntimes[authorTxt] = len(authorTxts) - 1
        if geneTxt   not in gene2ntimes:   gene2ntimes  [geneTxt]   = len(geneTxts) - 1
        if pepTxt    not in pep2ntimes:    pep2ntimes   [pepTxt]    = len(pepTxts) - 1
        if hlaTxt    not in hla2ntimes:    hla2ntimes   [hlaTxt]    = len(hlaTxts) - 1

        authorInID = author2ntimes[authorTxt]
        geneInID   = gene2ntimes  [geneTxt]
        pepInID    = pep2ntimes   [pepTxt]
        hlaInID    = hla2ntimes   [hlaTxt]

        dataID = 'D' + str(dataID_ord).zfill(2) + pre_dataID + (
                #chr(ord('A')+geneInID) + 
                chr(ord('A')+pepInID) + chr(ord('A')+hlaInID)
                ) + str(difficulty[4])
        # dataIDs2ntimes[pre_dataID] += 1
        # author, gene, hla
        
        roc_aucs = []
        for colname in METHODS:
            pepranks = list(zip(df3['ET_pep'], df3[colname]))
            logging.info(F'gene={gene} HLA={HLA} ColumnName={colname} ranks={pepranks}')
            roc_auc = roc_auc_score(df3['IsTrue'], -df3[colname])
            x, y = -df3[colname], df3['IsTrue']
            x1 = x[y == 1]  # Values where y is 1
            x0 = x[y == 0]  # Values where y is 0
            sigtest_res = sp.stats.mannwhitneyu(x0, x1)
            nsamples = len(df3)
            npositives = sum(df3['IsTrue'])
            nnegatives = nsamples - npositives
            print(F'Gene={gene}\tHLA={HLA}\tLoci={loci:<40}\tNumExamples={nsamples}\tNumPosNeg={npositives}/{nnegatives}\tmethod={colname:<20}\tROCAUC={roc_auc}')
            
            dicts4df.append({
                'Dataset IDs': dataID, 
                'Dataset publication year' : difficulty[5],
                'Dataset source': descs[0],
                'Modified residue': ('MHC-anchored' if BY_MHC == difficulty[0] else 'TCR-facing'),
                'Antigen type': descs[4],
                'MHC': descs[1],
                'Gene': descs[2],
                'Peptides': descs[3],
                '#Tested-positive': npositives,
                '#Tested-total': nsamples,
                #'Gene': gene,
                #'HLA' : HLA,
                #'Loci': loci,
                #'NumExamples' : nsamples,
                'Ranking methods': colname,
                'AUROC'          : roc_auc,
                'P-value'        : sigtest_res.pvalue,
                })
            roc_aucs.append(roc_auc)
        df3.to_csv(args.output + '.' + geneHLA.replace('/', '--') + '.tsv', sep='\t', header=1, index=0, na_rep='NA')
        logging.info(f'Saved df3=' + args.output + '.' + geneHLA.replace('/', '--') + '.tsv')
        '''
        dict4heatmap = {
                'Dataset IDs': dataID, 
                'Dataset publication year' : difficulty[5],
                'Dataset source': descs[0],
                'Modified residue': ('MHC-anchored' if BY_MHC == difficulty[0] else 'TCR-facing'),
                'MHC': descs[1],
                'Gene': descs[2],
                'Peptides': descs[3],
                '#Tested-positive': npositives,
                '#Tested-total': nsamples,
        }
        for colidx, colname in enumerate(METHODS):
            dict4heatmap[warp_roc_auc(colname)] = roc_aucs[colidx]
        dicts4heatmap.append(dict4heatmap)
        '''
    logging.info(f'Process df4')
    df4a = pd.DataFrame.from_records(dicts4df)
    df4b = df4a.loc[~df4a['Peptides'].str.contains('~at~position~') & ~df4a['Peptides'].str.contains('~nonanchor'), :]
    df4b = df4b.loc[(df4b['#Tested-total'] >= 10),:]

    #df5 = pd.DataFrame.from_records(dicts4heatmap)
    for dfform, df4 in [('full', df4a), ('brief', df4b)]:
        df4a.to_csv(args.output + '.' + dfform + '.tsv', sep='\t', header=1, index=0, na_rep='NA')
        
        df5table = df4[['Dataset IDs', 'Dataset source', 'Dataset publication year', 'Modified residue', 'Antigen type', 'MHC', 'Gene', 'Peptides', '#Tested-positive', '#Tested-total']].drop_duplicates()
        df5heatmap = df4.pivot(index='Dataset IDs', columns='Ranking methods', values='AUROC')
        df5pvalues = df4.pivot(index='Dataset IDs', columns='Ranking methods', values='P-value')

        # Compute the mean for each column
        mean_row = df5heatmap.mean(numeric_only=True)
        min_row  = df5pvalues.min (numeric_only=True)

        # Optional: set a name for the row (e.g., "Average")
        mean_row.name = 'Mean'
        # Append the new row
        df5heatmap = pd.concat([df5heatmap, mean_row.to_frame().T])
        df5pvalues = pd.concat([df5pvalues, min_row .to_frame().T])
        
        annotations = heatmap2annotations(df5heatmap, df5pvalues)

        df5table.to_csv(args.output + '.' + dfform + '.data_summary.tsv', sep='\t', index=False)
        
        fig, ax = plt.subplots(figsize=(10, 7.5), constrained_layout=True)
        ax = sns.heatmap(df5heatmap, annot=annotations, ax=ax, fmt='', center=0.5) #(, fmt='.3g')
        # Rotate x-axis tick labels
        plt.setp(ax.get_xticklabels(), rotation=20, ha='right')
        plt.setp(ax.get_yticklabels(), rotation=0 , ha='right', va='center', fontfamily='monospace')
        ax.set_ylabel('Dataset IDs', labelpad=0)  # Adjust pad to bring it closer
        
        #plt.tight_layout()
        # Save figure
        plt.savefig(args.output + '.' + dfform + '.heatmap.pdf', )
        plt.savefig(args.output + '.' + dfform + '.heatmap.png', dpi=600)
        plt.close()
        
        

        '''
        df5.columns = [x.replace('_', ' ') for x in df5.columns]
        #df5 = df5.str.replace('_', ' ')
        df5['MHC'] = df5['MHC'].str.replace('-', '_')
        df5.reset_index(drop=True)
        cm = 'Greens' #'YlOrRd' # sns.light_palette("green", as_cmap=True)
        styled_table = df5.style.background_gradient(cmap=cm, axis=None, subset=[warp_roc_auc(x).replace('_', ' ') for x in METHODS])
        html = styled_table.to_html()
        with open(args.output + '.html', 'w') as file: file.write(html)
        df6 = df5.loc[~df5['Peptides'].str.contains('~at~position~') & ~df5['Peptides'].str.contains('~nonanchor'), [x for x in df5.columns if x not in ['MHCmotifAtlas_diff']]]
        df6 = df6.loc[(df6['#Tested-positive'] + df6['#Tested-negative'] >= 10),:]
        styled_table = df6.style.background_gradient(cmap=cm, axis=None, subset=[warp_roc_auc(x).replace('_', ' ') for x in METHODS])
        html = styled_table.to_html()
        with open(args.output + '.brief.html', 'w') as file: file.write(html)
        logging.info(f'Saved df6={args.output}.brief.html')
        df6b_colidxs = df6.columns.str.contains('_AUROC')
        df6a = df6.loc[:,~df6b_colidxs]
        df6b = df6.loc[:, df6b_colidxs]
        df6.to_csv(args.output + '.brief.tsv', sep='\t', index=False)
        df6b.columns = df6b.columns.str.replace('_AUROC', '')
        sns.heatmap(df6b, )
        '''
if __name__ == '__main__': main()

