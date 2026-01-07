#!/usr/bin/env python

import argparse, collections, json, os, sys

def get_val_by_key(fhdr, key):
    val = None
    for i, tok in enumerate(fhdr.split()):
        if i > 0 and len(tok.split('=')) == 2:
            k, v = tok.split('=')
            if k == key:
                assert val == None, 'The header {} has duplicated key {}'.format(fhdr, key)
                val = v
    return val

def rm_key_val(fhdr, key):
    ret = []
    for i, tok in enumerate(fhdr.split()):
        if i > 0 and len(tok.split('=')) == 2:
            k, v = tok.split('=')
            if k != key:
                ret.append(tok)
        else:
            ret.append(tok)
    return ' '.join(ret)

def main():
    parser = argparse.ArgumentParser(description = 'Read FASTA records from stdin, merge records with duplicated sequences while summing values of the key (where the key-value pair is of the form <key>=<val> in comments), and write the merged FASTA records to stdout. ',
            formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--key', default = '', type = str,
        help = 'The key by which the summation is performed (e.g., TPM for FASTA header with TPM=... in its comment). ')
    parser.add_argument('--seq2hdrs', default = '', type = str,
        help = 'The original sequence-to-header mapping is stored to this json file. Skip generating this json file if this param is set to the empty string. ')

    args = parser.parse_args()
    
    fhdr = None
    uhdrs = []
    uhdr2seq = {}
    seq2hdrs = collections.defaultdict(list)
    seq2val = collections.defaultdict(int)
    for line in sys.stdin:
        if line.startswith('>'):
            if fhdr:
                fseq = ''.join(fseq)
                if fseq not in seq2hdrs:
                    uhdrs.append(uhdr)
                    uhdr2seq[uhdr] = fseq
                seq2hdrs[fseq].append(fhdr)
                seq2val[fseq] += val
            fhdr = line.strip()
            val = get_val_by_key(fhdr, args.key)
            assert val, F'The header {line} does not have the key {args.key}!'
            val = float(val)
            uhdr = rm_key_val(fhdr, args.key)
            fseq = []
        else:
            fseq.append(line.strip())
    if fhdr:
        fseq = ''.join(fseq)
        if fseq not in seq2hdrs:
            uhdrs.append(uhdr)
            uhdr2seq[uhdr] = fseq
        seq2hdrs[fseq].append(fhdr)
        seq2val[fseq] += val
    for uhdr in uhdrs:
        fseq = uhdr2seq[uhdr]
        print(F'''{uhdr} {args.key}={seq2val[fseq]}\n{(fseq)}''')
    if args.seq2hdrs:
        with open(args.seq2hdrs, 'w') as file:
            json.dump(seq2hdrs, file, indent=2)

if __name__ == '__main__': main()

