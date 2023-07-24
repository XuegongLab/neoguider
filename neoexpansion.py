#!/usr/bin/env python

import logging

#import pprint
import argparse,multiprocessing,sys
from Bio.SubsMat import MatrixInfo

#          '12345678901234567890'
ALPHABET = 'ARNDCQEGHILKMFPSTWYV'

def aaseq2canonical(aaseq): return aaseq.upper().replace('U', 'X').replace('O', 'X')

def get_neighbour_seqs(in_string_seq, alphabet = 'ARNDCQEGHILKMFPSTWYV'):
    ret = []
    for i in range(len(in_string_seq)):
        sequence1 = [x for x in in_string_seq]
        for letter in alphabet: 
            sequence1[i] = letter
            ret.append(''.join(sequence1))
    return ret

def alnscore_penalty(sequence1, neighbour1):
    assert len(sequence1) == len(neighbour1)
    sequence = aaseq2canonical(sequence1)
    neighbour = aaseq2canonical(neighbour1)
    ret = 0
    for i in range(len(sequence)):
        if sequence[i] != neighbour[i]:
            scoremax = MatrixInfo.blosum62[(sequence[i], sequence[i])]
            ab = (sequence[i], neighbour[i])
            ba = (neighbour[i], sequence[i])
            if ab in MatrixInfo.blosum62: 
                score = MatrixInfo.blosum62[ab]
            else:
                score = MatrixInfo.blosum62[ba]
            ret += scoremax - score
    return ret

def pep2simpeps(bioseq, nbits):
    queue = [ bioseq ]
    seq2penalty = { bioseq : 0 }
    while queue:
        nextseq = queue.pop(0)
        for neighbour in get_neighbour_seqs(nextseq):
            penalty = alnscore_penalty(bioseq, neighbour)
            if not neighbour in seq2penalty and penalty * 0.5 <= nbits * (1.0 + sys.float_info.epsilon):
                queue.append(neighbour)
                seq2penalty[neighbour] = penalty * 0.5
    return seq2penalty

def faa2newfaa(arg):
    hdr, pep, nbits = arg
    pep2 = aaseq2canonical(pep)
    for aa in pep2:
        assert aa in ALPHABET, (F'The FASTA record (header={hdr}, sequence={pep}) contains non-standard amino-acid {aa} which is not in {ALPHABET}')
    logging.debug(F'Processing {hdr} {pep}')
    ret = []
    seq2penalty = pep2simpeps(pep, nbits)
    seqs = sorted(seq2penalty.keys())
    seqs.remove(pep)
    for i, simpep in enumerate([pep] + seqs):
        new_ID = hdr.split()[0] + str(i)
        pep_comment = ' '.join([tok for (j, tok) in enumerate(hdr.split()) if j > 0])
        new_hdr = (F'{new_ID} {pep_comment} SOURCE={pep} MAX_BIT_DIST={seq2penalty[simpep]}')
        new_seq = (simpep)
        ret.append((new_hdr, new_seq))
    return ret

def main():
    #pp = pprint.PrettyPrinter(indent=4)
    #pp.pprint(MatrixInfo.blosum62)
    parser = argparse.ArgumentParser(description = 'Experimental work (please do not use output sequences with MAX_BIT_DIST>0 for now). ',
            formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-b', '--nbits', type = float, help = 'hamming distance by the number of bits', default = 1.0)
    parser.add_argument('-c', '--ncores', type = int, help = 'number of processes to use for computation', default = 8)
    args = parser.parse_args()
    hdr = None
    hdr_pep_list = []
    for line in sys.stdin:
        if line.startswith('>'):
            if hdr: hdr_pep_list.append((hdr, pep, args.nbits))
            hdr = line.strip()
        else:
            pep = line.strip()
    if hdr: hdr_pep_list.append((hdr, pep, args.nbits))
    if args.ncores >= 0:
        with multiprocessing.Pool(args.ncores) as p:
            new_hdr_seq_list_list = p.map(faa2newfaa, hdr_pep_list, chunksize = 10)
    else:
            new_hdr_seq_list_list =   map(faa2newfaa, hdr_pep_list)
    for new_hdr_seq_list in new_hdr_seq_list_list:
        for new_hdr, new_seq in new_hdr_seq_list:
            print(new_hdr)
            print(new_seq)
if __name__ == '__main__':
    main()

