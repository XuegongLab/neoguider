#!/usr/bin/env python

import argparse, collections, json, logging, multiprocessing, os, sys

from Bio.Align import substitution_matrices
#from Bio.SubsMat import MatrixInfo

def isna(arg): return arg in [None, '', 'NA', 'Na', 'N/A', 'None', 'none', '.']

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
            # neighbour is the original sequence whose TCR can cross react with bioseq
            penalty = alnscore_penalty(neighbour, bioseq, blosum62)
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

NA_REP = 'N/A'
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
    #            -1 0      1      2    3      4    5    6      7    8      9        10     11     12
    logging.info(cmd)
    os.system(cmd)
    qseq2sseq = {}
    qseq2maxb = {}
    qseq2info = collections.defaultdict(list)
    with open(F'{query_fasta}.blastp_out.tmp') as blastp_csv:
        for line in blastp_csv:
            tokens = line.strip().split(',')
            qseq = tokens[2].replace('-', '')
            sseq = tokens[5].replace('-', '')
            bitscore = float(tokens[12])
            qseq2info[qseq].append((bitscore, tokens[1], sseq, tokens[9], tokens[10]))
            is_canonical = all([(aa in 'ARNDCQEGHILKMFPSTWYV') for aa in sseq])
            if is_canonical and qseq2maxb.get(qseq, -1) < bitscore:
                qseq2sseq[qseq] = sseq
                qseq2maxb[qseq] = bitscore
    ret1 = []
    ret2 = []
    for qseq in query_seqs:
        ret1.append(qseq2sseq.get(qseq, NA_REP))
        ret2.append(qseq2info.get(qseq, []))
    return ret1, ret2

def blast_search(hdr_pep_list, reference, output_file, ncores):
    pep_list = [pep                        for (hdr, pep, _) in hdr_pep_list]
    hla_list = [get_val_by_key(hdr, 'HLA') for (hdr, pep, _) in hdr_pep_list]
    selfpeps1, alninfos1  = runblast(pep_list, reference, output_file, ncores)
    pep2hla = {}
    for hla, selfpep in zip(hla_list, selfpeps1):
        if selfpep in pep2hla and pep2hla[selfpep] != hla:
            logging.warning(F'The peptide {selfpep} has multiple mutant sequences with different HLA alleles {hla} and {pep2hla[selfpep]}')
            pep2hla[selfpep] = pep2hla[selfpep] + ',' + hla
        elif not (selfpep in pep2hla):
            pep2hla[selfpep] = hla
    selfpeps = sorted(list(set(selfpeps1)))
    for (hdr, pep, _), selfpep, alninfo in zip(hdr_pep_list, selfpeps1, alninfos1):
        alninfo_str = json.dumps(sorted(alninfo)[::-1], separators=(',', ':'), sort_keys=True)
        print(F'{hdr} ST={selfpep} IsNeoPeptide=1 SelfAlnInfo={alninfo_str}')
        print(pep)
    for i, selfpep in enumerate(selfpeps):
        if isna(selfpep):
            logging.warning(F'The string (ST="{selfpep}") is not found in the list of peptides and is skipped. ')
            continue
        hla = pep2hla[selfpep]
        print(F'>SELF_{i+1} HLA={hla} ST={selfpep} IsHelperPeptide=1 IsSearchedFromSelfProteome=1')
        print(selfpep)

def process_WT(hdr_pep_list):
    wt2hlas = collections.defaultdict(set)
    for (hdr, pep, _) in hdr_pep_list:
        hla = get_val_by_key(hdr, 'HLA')
        wtpep = get_val_by_key(hdr, 'WT')
        # toks = hdr.split()
        # if len(toks) == 1: continue
        # for tok in toks[1:]:
        #    if 2 == len(tok.split('=')):
        #        key, val = tok.split('=')
        #        if key == 'WT': wt2hlas[val].add() # [val].append(pep) # append((len(toks[0]), toks[0], hdr, pep))
        if wtpep: wt2hlas[wtpep].add(hla)
    wildpeps = sorted(list(wt2hlas.keys()))
    for i, wildpep in enumerate(wildpeps):
        if isna(wildpep):
            logging.warning(F'The string (WT="{wildpep}") is found in the list of peptides and is skipped. ')
            continue
        hlas = sorted(list(wt2hlas[wildpep]))
        if len(hlas) > 1:
            logging.warning('The wild-type peptide (WT={wildpep}) is associated with multiple sets of HLA alleles ({hlas}). ')
            hlastr = ','.join(hlas)
        else:
            hlastr = hlas[0]
        print(F'>WILD_{i+1} HLA={hlastr} WT={wildpep} IsHelperPeptide=1 IsGeneratedFromWT=1')
        print(wildpep)
    
def main():
    #pp = pprint.PrettyPrinter(indent=4)
    #pp.pprint(MatrixInfo.blosum62)
    parser = argparse.ArgumentParser(description = 'Experimental work (please do not use output sequences with MAX_BIT_DIST>0 for now). ',
            formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-b', '--nbits', type = float, help = 'max hamming distance by the number of bits. If this param is set, then do not expand by --reference', default = 0.75)
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

