#!/usr/bin/env python

import argparse,collections,json,logging,sys

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

def output(fhdr, fseq, alphabet, tpm_thres, hla, edit_dist_thres, rmdups, visited, seqlen):
    tpm = 0
    edit_dist = -1
    for i, tok in enumerate(fhdr.split()):
        if i > 0 and len(tok.split('=')) == 2:
            k, v = tok.split('=')
            if k == 'TPM': tpm = float(v)
            if k == 'EDIT_DIST': edit_dist = int(v)
    if tpm < tpm_thres: return 1
    if edit_dist > edit_dist_thres: return 2
    fseq = ''.join(fseq)
    if rmdups:
        ret = (fseq in visited)
        visited[fseq].append(fhdr)
        if ret: return 3
    if seqlen < len(fseq):
        logging.warning(F'The FASTA record (header={fhdr}, seq={fseq}) is filtered out due to sequence length ({seqlen} < {len(fseq)})!')
        return 4
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
    parser.add_argument('-e', '--edit-dist-thres', default = 9, type = int, help = 'Edit-distance threshold above which heteroclitic peptides (enhanced mimotopes) are discarded. ')
    parser.add_argument('-r', '--rmdups', default = '', type = str, help = 'File to store records with duplicated sequences.  If set to empty string, then do not remove these records.')
    parser.add_argument('-l', '--len', default=120, type = int, help = 'Records with sequence length below this threshold are discarded.')

    args = parser.parse_args()
    visited = collections.defaultdict(list)
    fhdr = None
    for line in sys.stdin:
        if line.startswith('>'):
            if fhdr: output(fhdr, fseq, args.alphabet, args.tpm, args.hla, args.edit_dist_thres, args.rmdups, visited, args.len)
            fhdr = line.strip()
            fseq = []
        else:
            fseq.append(line.strip())
    if fhdr: output(fhdr, fseq, args.alphabet, args.tpm, args.hla, args.edit_dist_thres, args.rmdups, visited, args.len)
    if args.rmdups:
        with open(args.rmdups, 'w') as file:
            json.dump(visited, file, indent=2)
if __name__ == '__main__':
    main()

