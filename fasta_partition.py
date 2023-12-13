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

def get_val_to_id_seq(key):
    val2records = collections.defaultdict(list)
    fhdr = None
    for line in sys.stdin:
        if line.startswith('>'):
            if fhdr:
                val = get_val_by_key(fhdr, key)  # output(fhdr, fseq, args.alphabet, args.tpm, args.hla)
                assert val, 'The header {} does not have the key {}'.format(fhdr, key)
                val2records[val].append((fhdr, ''.join(fseq)))
            fhdr = line.strip()
            fseq = []
        else:
            fseq.append(line.strip())
    if fhdr:
        val = get_val_by_key(fhdr, key)
        assert val, 'The header {} does not have the key {}'.format(fhdr, key)
        val2records[val].append((fhdr, ''.join(fseq)))
    return val2records

def partition(val2records, outpref):
    outdir = '{}.partition'.format(outpref)
    os.system('rm -r {} ; mkdir -p {}'.format(outdir, outdir))
    val2outfname = {}
    for val, records in sorted(val2records.items()):
        val2 = ''.join([ c if c.isalnum() else '_' for c in val])
        outfname='{}.fasta'.format(val2)
        outfpath='{}/{}'.format(outdir, outfname)
        with open(outfpath, 'w') as outfile:
            for fid, fseq in records:
                outfile.write('{}\n{}\n'.format(fid, fseq))
        val2outfname[val] = outfname
    with open('{}/mapping.json'.format(outdir), 'w') as sumfile:
        json.dump(val2outfname, sumfile, indent=2)

def main():
    parser = argparse.ArgumentParser(description = 'Read FASTA records from stdin, partition FASTA records according to --key, and write FASTA records to --out. ',
            formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--key', default = '', type = str,
        help = 'Partition the fasta input from stdin into multiple fasta files with this key (e.g., HLA for FASTA header with HLA=... in its comment). ')
    parser.add_argument('--out', default = '', type = str, 
        help = 'Output prefix of the partition such that result fasta files are in the <out>.partition directory. ')

    args = parser.parse_args()
    val_to_id_seq = get_val_to_id_seq(args.key)
    partition(val_to_id_seq, args.out)

if __name__ == '__main__': main()

