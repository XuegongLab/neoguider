import argparse, os, sys
import pandas as pd

scriptdir = (os.path.dirname(os.path.realpath(__file__)))
parser = argparse.ArgumentParser(description='This script analyzes features (the features are typically the output of relevant software packages, such as kallisto, netMHCpan, mhcflurry, PRIME, ERGO, and netTCR). ', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-i', '--input', nargs='+', default=[], help='Input files, where each file has the following columns: patient ID, peptide sequence, and 4-digit HLA name. ')
#parser.add_argument('-f', '--features', default=['deephlaan'], help='Feature names')
parser.add_argument('--sep', default='', help='Column separator, empty-string means auto-infer')

args = parser.parse_args()

# https://github.com/jiujiezz/deephlapan
  
example_input = '''
  Annotation,HLA,peptide
  NCI-3784,HLA-A01:01,MKRFVQWL
  NCI-3784,HLA-A03:01,MKRFVQWL
  NCI-3784,HLA-B07:02,MKRFVQWL
  NCI-3784,HLA-B07:02,MKRFVQWL
  NCI-3784,HLA-C07:02,MKRFVQWL
  NCI-3784,HLA-C07:02,MKRFVQWL
  NCI-3784,HLA-A01:01,KRFVQWLK
  NCI-3784,HLA-A03:01,KRFVQWLK
  NCI-3784,HLA-B07:02,KRFVQWLK
  NCI-3784,HLA-B07:02,KRFVQWLK
  NCI-3784,HLA-C07:02,KRFVQWLK
  NCI-3784,HLA-C07:02,KRFVQWLK
'''
# cmd = deephlapan -F [file] -O [output directory]

column_set_0 = 'PatientID,HLA_type_y,MT_pep_y'.split(',')
column_set_1 = 'patient,mutant_best_alleles_netMHCpan,mutant_seq'.split(',')
column_set_2 = 'Patient,HLA_allele,Mut_peptide'.split(',')

def allin(X, Y): return all([(x in Y) for x in X])

for infilename in args.input:
    with open(infilename) as file:
        firstline = file.readline()
        if not args.sep:
            if   firstline.count('\t') > 3: csvsep = '\t'
            elif firstline.count(',')  > 3: csvsep = ','
            elif firstline.count(' ')  > 3: csvsep = ' '
            else: raise RuntimeError(F'Cannot infer the column separator string from the first line of the file {infilename}!')
        else:
            csvsep = args.sep
    df = pd.read_csv(infilename, sep=csvsep)
    if   allin(column_set_0, df.columns): column_set = column_set_0
    elif allin(column_set_1, df.columns): column_set = column_set_1
    elif allin(column_set_2, df.columns): column_set = column_set_2
    else: raise RuntimeError(F'Invalid columns `{df.columns}` encountered in the file {infilename}!')
    df_2 = df[column_set]
    df_2.columns = 'Annotation,HLA,peptide'.split(',')
    df_2['HLA'] = df_2['HLA'].str.replace('*', '')
    dhp_in = infilename + '.dhp_in'
    dhp_out = infilename + '.dhp_out'
    
    dhp_dir = os.path.dirname(dhp_in)
    dhp_in_base = os.path.basename(dhp_in)
    dhp_out_base = os.path.basename(dhp_out)
    #ISO_MODULE, ISO_EXT = os.path.splitext(ISO_NAME)
    
    df_2.to_csv(dhp_in, sep=',', index=False)
    #cmd = F'deephlapan -F {dhp_in} -O {dhp_out}'
    #os.makedirs(dhp_out_base)
    cmd = F'/usr/bin/podman run -v {dhp_dir}:/data -it --rm biopharm/deephlapan:v1.1 deephlapan -F /data/{dhp_in_base} -O /data/'
    print(cmd)

