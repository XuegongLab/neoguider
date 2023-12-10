#!/usr/bin/env python

import argparse, logging, multiprocessing, os, sys

from Bio.Align import substitution_matrices
#from Bio.SubsMat import MatrixInfo

def isna(arg): return arg in [None, '', 'NA', 'Na', 'None', 'none', '.']

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

def alnscore_penalty(sequence1, neighbour1, blosum62):
    assert len(sequence1) == len(neighbour1)
    sequence = aaseq2canonical(sequence1)
    neighbour = aaseq2canonical(neighbour1)
    ret = 0
    for i in range(len(sequence)):
        if sequence[i] != neighbour[i]:
            scoremax = blosum62[(sequence[i], sequence[i])]
            ab = (sequence[i], neighbour[i])
            ba = (neighbour[i], sequence[i])
            if ab in blosum62: 
                score = blosum62[ab]
            else:
                score = blosum62[ba]
            ret += scoremax - score
    return ret

def pep2simpeps(bioseq, nbits, blosum62):
    queue = [ bioseq ]
    seq2penalty = { bioseq : 0 }
    while queue:
        nextseq = queue.pop(0)
        for neighbour in get_neighbour_seqs(nextseq):
            penalty = alnscore_penalty(bioseq, neighbour, blosum62)
            if not neighbour in seq2penalty and penalty * 0.5 <= nbits * (1.0 + sys.float_info.epsilon):
                queue.append(neighbour)
                seq2penalty[neighbour] = penalty * 0.5
    return seq2penalty

def faa2newfaa(arg):
    blosum62 = substitution_matrices.load("BLOSUM62")
    hdr, pep, nbits = arg
    pep2 = aaseq2canonical(pep)
    for aa in pep2:
        assert aa in ALPHABET, (F'The FASTA record (header={hdr}, sequence={pep}) contains non-standard amino-acid {aa} which is not in {ALPHABET}')
    logging.debug(F'Processing {hdr} {pep}')
    ret = []
    seq2penalty = pep2simpeps(pep, nbits, blosum62)
    seqs = sorted(seq2penalty.keys())
    seqs.remove(pep)
    for i, simpep in enumerate([pep] + seqs):
        new_ID = hdr.split()[0] + (('_' + str(i)) if (i > 0) else '')
        pep_comment = ' '.join([tok for (j, tok) in enumerate(hdr.split()) if j > 0])
        new_hdr = (F'{new_ID} {pep_comment} SOURCE={pep} MAX_BIT_DIST={seq2penalty[simpep]}')
        new_seq = (simpep)
        ret.append((new_hdr, new_seq))
    return ret

NA_STRING = ''
def runblast(query_seqs, target_fasta, output_file, ncores):
    query_fasta = F'{output_file}.query_seqs.fasta.tmp'
    with open(query_fasta, 'w') as query_fasta_file:
        for query_seq in query_seqs:
            query_fasta_file.write(F'>{query_seq}\n{query_seq}\n')
    # from https://github.com/andrewrech/antigen.garnish/blob/main/R/antigen.garnish_predict.R
    cmd = F'''blastp -word_size 3 -qcov_hsp_perc 100 \
        -query {query_fasta} -db {target_fasta} \
        -evalue 100000000 -matrix BLOSUM62 -gapopen 11 -gapextend 1 \
        -out {query_fasta}.blastp_out.tmp -num_threads {ncores} \
        -outfmt '10 qseqid sseqid qseq qstart qend sseq sstart send length mismatch pident evalue bitscore' '''
    logging.info(cmd)
    os.system(cmd)
    ret = []
    qseq2sseq = {}
    qseq2maxb = {}
    qseq2tpm  = {}
    with open(F'{query_fasta}.blastp_out.tmp') as blastp_csv:
        for line in blastp_csv:
            tokens = line.strip().split(',')
            qseq = tokens[2]
            sseq = tokens[5]
            bitscore = float(tokens[-1])
            is_canonical = all([(aa in 'ARNDCQEGHILKMFPSTWYV') for aa in sseq])
            if is_canonical and qseq2maxb.get(qseq, -1) < bitscore:
                qseq2sseq[qseq] = sseq
                qseq2maxb[qseq] = bitscore
    ret = []
    for qseq in query_seqs:
        ret.append(qseq2sseq.get(qseq, NA_STRING))
    return ret

def blast_search(hdr_pep_list, reference, output_file, ncores):
    pep_list = [pep for (hdr, pep, _) in hdr_pep_list]
    selfpeps1 = runblast(pep_list, reference, output_file, ncores)
    selfpeps = sorted(list(set(selfpeps1)))
    for (hdr, pep, _), selfpep in zip(hdr_pep_list, selfpeps1):
        print(F'{hdr} ST={selfpep} IsNeoPeptide=1')
        print(pep)
    for i, selfpep in enumerate(selfpeps):
        if isna(selfpep):
            logging.warning('The string (ST={selfpep}) is found in the list of peptides and is skipped. ')
            continue
        print(F'>SELF_{i+1} ST={selfpep} IsHelperPeptide=1 IsSearchedFromSelfProteome=1')
        print(selfpep)

def process_WT(hdr_pep_list):
    wt2peps = set([]) # collections.defaultdict(list)
    for (hdr, pep, _) in hdr_pep_list:
        toks = hdr.split()
        if len(toks) == 1: continue
        for tok in toks[1:]:
            if 2 == len(tok.split('=')):
                key, val = tok.split('=')
                if key == 'WT': wt2peps.add(val) # [val].append(pep) # append((len(toks[0]), toks[0], hdr, pep))
    wildpeps = sorted(list(wt2peps))
    for i, wildpep in enumerate(wildpeps):
        if isna(wildpep):
            logging.warning('The string (WT={wildpep}) is found in the list of peptides and is skipped. ')
            continue
        print(F'>WILD_{i+1} WT={wildpep} IsHelperPeptide=1 IsGeneratedFromWT=1')
        print(wildpep)
    
def main():
    #pp = pprint.PrettyPrinter(indent=4)
    #pp.pprint(MatrixInfo.blosum62)
    parser = argparse.ArgumentParser(description = 'Experimental work (please do not use output sequences with MAX_BIT_DIST>0 for now). ',
            formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-b', '--nbits', type = float, help = 'hamming distance by the number of bits. If this param is set, then do not expand by --reference', default = 1.0)
    parser.add_argument('-c', '--ncores', type = int, help = 'number of processes to use for computation. ', default = 8)
    parser.add_argument('-r', '--reference', type = str, help = 'reference proteome fasta to blast against to match neo-peptides with self-peptides. If this param is set, then do not expand by --nbits. ', default = '')
    parser.add_argument('-t', '--tmp', type = str, help = 'temporary file path. ', default = '')
    
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
    if args.reference != '':        
        blast_search(hdr_pep_list, args.reference, args.tmp, args.ncores)
        process_WT(hdr_pep_list)
        exit(0)
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

