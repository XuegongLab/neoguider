import argparse, collections, copy, datetime, json, itertools, logging, os, pickle, pprint, random, sys
from collections import Counter, defaultdict, namedtuple

import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import QuantileTransformer

def match_col(df, colnames, dup='raise'):
    valid_dups = ['raise', 'warn', 'skip']
    assert dup in valid_dups, F'The dup value {dup} should be in {valid_dups}!'
    ret = ''
    for colname in colnames:
        if colname in df.columns:
            if ret != '':
                msg = F'The columns {ret} and {colname} are both found'
                if dup == 'raise':
                    raise ValueError(F'{msg}, aborting!')
                elif dup == 'warn':
                    logging.warning(F'{msg}, skpping {colname} with warning!')
                    continue
                elif dup == 'skip':
                    logging.debug(F'{msg}, skipping {colname} silently')
                    continue
                else:
                    raise RuntimeError(F'{msg}, this code should not execute!')
            ret = colname
    return ret

def transform(df):
    return QuantileTransformer(random_state=0).fit_transform(df)

def assert_non_na(df, dfname):
    if df.isna().any().any():
        # Find the first NA location for error message
        na_rows, na_cols = np.where(df.isna())
        first_na_row = na_rows[0]
        first_na_col = na_cols[0]
        raise AssertionError(
            f'The df {dfname} at row {first_na_row} col {first_na_col} is NA '
            f'(row={df.iloc[first_na_row]})'
            f'(dataframe={df})'
        )

scriptpath = (os.path.realpath(__file__))
scriptdir = (os.path.dirname(os.path.realpath(__file__)))
parser1 = argparse.ArgumentParser()
parser1.add_argument('-i', '--input', nargs='+', type=str, required=True, help='Input file')
parser1.add_argument('-o', '--output', nargs='*', type=str, default=[], help='Output file (default to <input>.pwqt, patient-wise quantile-transformed)')
parser1.add_argument('-I', '--isolib', default=scriptdir+'/../IsotonicLogisticRegression#IsotonicLogisticRegression',
        help='The NeoGuider feature transformation library file')
parser1.add_argument('--sep', type=str, default=None, help='Column separator chraracter in the input file')

args = parser1.parse_args()
print(args)

for fidx, infilename in enumerate(args.input):
    
    if not args.sep:
        testfile = infilename
        with open(testfile) as file:
            firstline = file.readline()
            if   firstline.count('\t') > 3: csvsep = '\t'
            elif firstline.count(',')  > 3: csvsep = ','
            elif firstline.count(' ')  > 3: csvsep = ' '
            else: raise RuntimeError(F'Cannot infer the column separator string from the first line of the file {testfile}!')
    else:
        csvsep = args.sep

    if not args.output:
        outfilename = infilename + '.pwqt'
    else:
        assert len(args.input) == len(args.output)
        outfilename = args.output[fidx]

    df = pd.read_csv(infilename, sep=csvsep)
    df = df.reset_index(drop=True)
    indexcols = []
    for indexcol in ['Unnamed: 0', 'Unnamed: 1']:
        if indexcol in df.columns:
            indexcols.append(indexcol)
    #if indexcols: df = df.drop(indexcols, axis=1)
    numeric_cols = df.select_dtypes(include=['number']).columns

    patientcol = match_col(df, ['Patient', 'PatientID', 'patient'])
    labelcol = match_col(df, ['Label', 'response', 'VALIDATED', 'response_type'])
    partition_cols = ['Partition', 'Patient', 'MT_pep', 'ET_pep', 'Epitope']
    df['groupby_patient'] = df[patientcol]
    
    transform_cols = [str(x) for x in numeric_cols if str(x) not in ([patientcol, labelcol, 'ln_NumTested'] + partition_cols)]

    out_dfs = []
    # patient-specific dataframe
    for patient, ps_df in df.groupby('groupby_patient'):
        ps_df2 = ps_df.copy()
        assert_non_na(ps_df[transform_cols], F'FirstDF.patient={patient}')
        if (ps_df[labelcol] != 1).all():
            # example: NCI_4348
            logging.warning(F'No positive label ({labelcol}) is found for patient {patient}, please manually check this patient if needed. ')
        preqt_df = ps_df[transform_cols].copy()
        assert_non_na(preqt_df[transform_cols], F'PreQT.patient={patient}')
        qt_df = transform(preqt_df)
        qt_df = pd.DataFrame(qt_df, columns=transform_cols)
        assert_non_na(qt_df[transform_cols], F'PostQT.patient={patient}')
        assert_non_na(ps_df2[transform_cols], F'PreFinalQT.patient={patient}')
        for col in transform_cols:
            ps_df2[col] = qt_df[col].values
        assert_non_na(ps_df2[transform_cols], F'FinalQT.patient={patient}')
        out_dfs.append(ps_df2)
    outdf = pd.concat(out_dfs, axis=0)
    assert_non_na(outdf[transform_cols], F'Outdf at {outfilename}')
    outdf.reset_index(drop=True).to_csv(outfilename, header=True, index=None, sep=csvsep)

