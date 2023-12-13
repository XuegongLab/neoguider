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
    
def main():
    parser = argparse.ArgumentParser(description = 'Read FASTA records from stdin, fill in records without any token of the form <key>=<val> in is comments with <--key>=<--default>, and write FASTA records to stdout. ',
            formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--key', default = '', type = str,
        help = 'Partition the fasta input from stdin into multiple fasta files with this key (e.g., HLA for FASTA header with HLA=... in its comment). ')
    parser.add_argument('--val', default = '', type = str,
        help = 'Default value of the key to be used if the key is not found in the FASTA header. ')
    args = parser.parse_args()
    
    fhdr = None
    for line in sys.stdin:
        if line.startswith('>'):
            if fhdr: print(F'''{fhdr}\n{''.join(fseq)}''')
            fhdr = line.strip()
            val = get_val_by_key(fhdr, args.key)
            if None == val: 
                assert args.val != '', F'The header {fhdr} does not have the key {key} and empty string --val is provided. '
                fhdr += F' {args.key}={args.val}'
            fseq = []
        else:
            fseq.append(line.strip())
    if fhdr: print(F'''{fhdr}\n{''.join(fseq)}''')

if __name__ == '__main__': main()

