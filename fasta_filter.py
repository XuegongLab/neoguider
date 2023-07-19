#!/usr/bin/env python

import argparse,logging,sys

#          '12345678901234567890'
ALPHABET = 'ARNDCQEGHILKMFPSTWYV'

def aaseq2canonical(aaseq): return aaseq.upper().replace('U', 'X').replace('O', 'X')

def faa2newfaa(arg):
    hdr, pep, nbits = arg
    pep2 = aaseq2canonical(pep)
    for aa in pep2:
        if aa not in ALPHABET:
            logging.warning(F'The FASTA record (header={hdr}, sequence={pep}) contains non-standard amino acids, so skipping this record. ')
            return []
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

def output(fhdr, fseq, alphabet):
    fseq = ''.join(fseq)
    for ch in fseq:
        if ch not in alphabet:
            logging.warning(F'The FASTA record (header={fhdr}, seq={fseq}) does not use the alphabet ({alphabet}), so skipping this record. ')
            return ch
    print(fhdr)
    print(fseq)

def main():
    #pp = pprint.PrettyPrinter(indent=4)
    #pp.pprint(MatrixInfo.blosum62)
    parser = argparse.ArgumentParser(description = 'Read FASTA records from stdin, keep records with sequences in --alphabet, and write FASTA records to stdout. ',
            formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-a', '--alphabet', default = ALPHABET, help = 'The alphabet that each kept FASTA sequence must conform to. ')
    args = parser.parse_args()
    fhdr = None
    for line in sys.stdin:
        if line.startswith('>'):
            if fhdr: output(fhdr, fseq, args.alphabet)
            fhdr = line.strip()
            fseq = []
        else:
            fseq.append(line.strip())
    output(fhdr, fseq, args.alphabet)

if __name__ == '__main__':
    main()

