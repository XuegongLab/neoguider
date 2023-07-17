#!/usr/bin/env python

#import pprint
import argparse,sys
from Bio.SubsMat import MatrixInfo

#          '12345678901234567890'
ALPHABET = 'ARNDCQEGHILKMFPSTWYV'

def get_neighbour_seqs(in_string_seq, alphabet = 'ARNDCQEGHILKMFPSTWYV'):
    ret = []
    for i in range(len(in_string_seq)):
        sequence1 = [x for x in in_string_seq]
        for letter in alphabet: 
            sequence1[i] = letter
            ret.append(''.join(sequence1))
    return ret

def alnscore_penalty(sequence, neighbour):
    assert len(sequence) == len(neighbour)
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
    
def main():
    #pp = pprint.PrettyPrinter(indent=4)
    #pp.pprint(MatrixInfo.blosum62)
    parser = argparse.ArgumentParser(description = 'Experimental work (please do not use this script). ', formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-b', '--nbits', type = float, help = 'hamming distance by the number of bits', default = 1.0)
    args = parser.parse_args()
    for line in sys.stdin:
        if line.startswith('>'): pepID = line.strip()
        else: 
            pep = line.strip()
            seq2penalty = pep2simpeps(pep, args.nbits)
            seqs = sorted(seq2penalty.keys())
            seqs.remove(pep)
            for i, simpep in enumerate([pep] + seqs):
                new_pepID = pepID.split()[0] + str(i)
                pep_comment = ' '.join([tok for (j, tok) in enumerate(pepID.split()) if j > 0])
                print(F'{new_pepID} {pep_comment} SOURCE={pep} MAX_BIT_DIST={seq2penalty[simpep]}')
                print(simpep)
    
if __name__ == '__main__':
    main()

