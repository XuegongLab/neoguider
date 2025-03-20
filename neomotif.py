import argparse,collections,logging,sys
from collections import Counter, defaultdict
import numpy as np
import pandas as pd

AAS = 'ARNDCQEGHILKMFPSTWYV'

def calc_nbits(count, totcount):
    assert count <= totcount
    if count == 0: return 0
    p = count / totcount
    return -p * np.log2(p)

def main():
    parser = argparse.ArgumentParser(description='This program takes as input a motif file and outputs a TSV file. ', formatter_class=argparse.ArgumentDefaultsHelpFormatter)    
    parser.add_argument('-i', '--input', help='Input motif file of the format from http://mhcmotifatlas.org/data/classI/MS/Peptides/all_peptides.txt', required=True)
    parser.add_argument('-o', '--output', help=
        'Prefix ($prefix) of two output files. \n'
        'File ${prefix}.motif.tsv: TSV file with the header columns: MHC allele, motif length, position in the motif, and amino acids (under each amino acid is an odds ratio). '
        'File ${prefix}.motif-length-distr.tsv: TSV file with the header columns: MHC allele and motif lengths (under each motif length is its count). ', required=True)
    parser.add_argument('-p', '--prot', help='Proteome FASTA file used for calculating background amino acid frequency. ', required=True)

    args = parser.parse_args()
    
    aacounter = Counter()
    with open(args.prot) as file:
        for line in file:
            if line.startswith('>') or line.startswith(';'): continue
            for aa in line.strip(): aacounter[aa] += 1
    totcount = sum(aacounter.values())
    aa2nbits = {aa : calc_nbits(count, totcount) for (aa, count) in aacounter.items()}
    aa2bgfreq = {aa : (count / float(totcount))  for (aa, count) in aacounter.items()}
    
    mhclen2poscounter = {}
    mhc2lencounter = {}
    with open(args.input) as file:
        for lineno, line in enumerate(file):
            if lineno == 0 and line.startswith('Allele'): continue
            mhc, pep = line.split()
            key = (mhc, len(pep))
            if key not in mhclen2poscounter:
                mhclen2poscounter[key] = [Counter() for i in range(len(pep))]
            for pos, aa in enumerate(pep):
                mhclen2poscounter[key][pos][aa] += 1
            if mhc not in mhc2lencounter:
                mhc2lencounter[mhc] = Counter()
            mhc2lencounter[mhc][len(pep)] += 1

    dfdict = defaultdict(list)
    dfdict['MHC'].append('Proteome')
    dfdict['Len'].append(-1)
    dfdict['Pos'].append(-1)
    
    for aa in AAS:
        dfdict[aa].append(aa2nbits[aa])
    dfdict['X'].append(sum([(aa2nbits[aa] * aa2bgfreq[aa]) for aa in AAS]))
    for (mhc, peplen) in mhclen2poscounter:
        for pos, aa2count1 in enumerate(mhclen2poscounter[(mhc, peplen)]):
            aa2count = {aa: count/aa2bgfreq[aa] for aa, count in aa2count1.items()}
            totcount = sum(aa2count.values())
            dfdict['MHC'].append(mhc)
            dfdict['Len'].append(peplen)
            dfdict['Pos'].append(pos)
            avg_nbits_gain = 0
            for aa in AAS:
                count = aa2count.get(aa, 0)
                nbits_gain = calc_nbits(count, totcount) # - aa2nbits[aa]
                dfdict[aa].append(nbits_gain)
                avg_nbits_gain += nbits_gain * count / totcount
            dfdict['X'].append(avg_nbits_gain)
    df = pd.DataFrame(dfdict)
    df.to_csv(args.output + '.motif.tsv', header=1, sep='\t', index=0, na_rep='NA', float_format='%.5f')
    
    dfdict = defaultdict(list)
    for mhc in mhc2lencounter:
        dfdict['MHC'].append(mhc)
        for motiflen in range(7-1,14+1+1):
             dfdict[(motiflen)].append(mhc2lencounter[mhc][motiflen])
    df = pd.DataFrame(dfdict)
    df.to_csv(args.output + '.motif-length-distr.tsv', header=1, sep='\t', index=0, na_rep='NA', float_format='%.5f')

if __name__ == '__main__': main()

