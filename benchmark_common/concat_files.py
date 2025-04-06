import argparse, os, warnings
import pandas as pd

DROP_COLS = (
    'WT_Foreignness ForeignDiff XSinfo MTinfo STinfo WTinfo EXinfo MTlen EXpep NmAffMT NmAffET NlAffMT NlAffET NmInfMT NmInfET NlInfMT NlInfET InTested_RankEL_LT0.5_frac'
    + ' ' + 'Rank VALID_N_TESTED VALID_CUMSUM PROBA_CUMSUM ML_pipeline PredictedProbWithOtherFeatureSet_1 SourceAlterationDetail PepTrace').split()

scriptdir = (os.path.dirname(os.path.realpath(__file__)))
parser = argparse.ArgumentParser(description='This script concatenate different files with the same header together', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-i', '--inputs', nargs='+', help='Input files')
parser.add_argument('-o', '--output', help='Output file')
parser.add_argument('-s', '--sep', default=None, help='Column separator, such as comma for CSV and tab for TSV. Auto infer if not provided')

args = parser.parse_args()
header = args.inputs[0]

assert len(args.inputs) >= 1, F'At least one input file should be provided'
testfile = args.inputs[0]
with open(testfile) as file:
    firstline = file.readline()
    if not args.sep:
        if   firstline.count('\t') > 3: csvsep = '\t'
        elif firstline.count(',')  > 3: csvsep = ','
        elif firstline.count(' ')  > 3: csvsep = ' '
        else: raise RuntimeError(F'Cannot infer the column separator string from the first line of the file {testfile}!')
    else:
        csvsep = args.sep
column_names = pd.read_csv(testfile, nrows=0, sep=csvsep).columns.tolist()
dfs = []
for file in args.inputs:
    #print('Reading the file {file}')
    df = pd.read_csv(file, sep=csvsep)
    if len(df) == 0:
        warnings.warn(F'Skipping the file {file} because it represents an empty dataframe. ')
        continue
    drop_cols =[col for col in DROP_COLS if col in df.columns]    
    assert list(df.columns) == column_names, F'The file {file} with columns\n{list(df.columns)}\ndoes not have the columns\n{column_names}\n!'
    df = df.drop(columns=drop_cols)
    dfs.append(df)
df = pd.concat(dfs)
df.to_csv(args.output, sep=csvsep)

