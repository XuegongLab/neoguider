import pprint
import sys
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

def pep2simpeps(bioseq):
    queue = [ bioseq ]
    seq2penalty = { bioseq : 0 }
    while queue:
        nextseq = queue.pop(0)  
        for neighbour in get_neighbour_seqs(nextseq):
            penalty = alnscore_penalty(bioseq, neighbour)
            if not neighbour in seq2penalty and penalty <= 2:
                queue.append(neighbour)
                seq2penalty[neighbour] = penalty
    return seq2penalty
    
def main():
    #pp = pprint.PrettyPrinter(indent=4)
    #pp.pprint(MatrixInfo.blosum62)
    #print((MatrixInfo.blosum62))
    for line in sys.stdin:
        pep = line.strip()
        seq2penalty = pep2simpeps(pep)
        for simpep in sorted(seq2penalty.keys()):
            print(F'>{simpep}_from_{pep}_score_{seq2penalty[simpep]}')
            print(simpep)
    
if __name__ == '__main__':
    main()

