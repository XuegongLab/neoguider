#!/usr/bin/env python

import argparse,logging,sys

#          '12345678901234567890'
ALPHABET = 'ARNDCQEGHILKMFPSTWYV'

def aaseq2canonical(aaseq): return aaseq.upper().replace('U', 'X').replace('O', 'X')

def get_val_by_key(fhdr, key):
    val = None
    for i, tok in enumerate(fhdr.split()):
        if i > 0 and len(tok.split('=')) == 2:
            k, v = tok.split('=')
            if k == key:
                assert val == None, 'The header {} has duplicated key {}'.format(fhdr, key)
                val = v
    return val

def output(fhdr, fseq, alphabet, tpm_thres, hla):
    tpm = 0
    for i, tok in enumerate(fhdr.split()):
        if i > 0 and len(tok.split('=')) == 2:
            k, v = tok.split('=')
            if k == 'TPM': tpm = float(v)
    if tpm < tpm_thres: return 1
    
    fseq = ''.join(fseq)
    for ch in fseq:
        if ch not in alphabet:
            logging.warning(F'The FASTA record (header={fhdr}, seq={fseq}) does not use the alphabet ({alphabet}), so skipping this record. ')
            return ch
    if hla: print(fhdr + ' HLA=' + hla)
    else: print(fhdr)
    print(fseq)
    return 0

def main():
    #pp = pprint.PrettyPrinter(indent=4)
    #pp.pprint(MatrixInfo.blosum62)
    parser = argparse.ArgumentParser(description = 'Read FASTA records from stdin, keep records with sequences in --alphabet, and write FASTA records to stdout. ',
            formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-a', '--alphabet', default = ALPHABET, help = 'The alphabet that each kept FASTA sequence must conform to. ')
    parser.add_argument(      '--hla', default = '', type = str, help = 'String of comma-separated HLA alleles to be added as comment to each FASTA sequence. ')
    parser.add_argument('-t', '--tpm', default = -1, type = float, help = 'Transcript per million (TPM) below which the FASTA sequence is filtered out. ')
    
    args = parser.parse_args()
    fhdr = None
    for line in sys.stdin:
        if line.startswith('>'):
            if fhdr: output(fhdr, fseq, args.alphabet, args.tpm, args.hla)
            fhdr = line.strip()
            fseq = []
        else:
            fseq.append(line.strip())
    if fhdr: output(fhdr, fseq, args.alphabet, args.tpm, args.hla)

if __name__ == '__main__':
    main()

