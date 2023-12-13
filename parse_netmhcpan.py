#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import argparse,collections,csv,getopt,json,logging,math,os,sys
import pandas as pd 

#from Bio import pairwise2
#from Bio.SubsMat import MatrixInfo as matlist

from Bio import Align
from Bio.Align import substitution_matrices

#from Bio.Align import MultipleSeqAlignment
#from Bio.Alphabet import IUPAC, Gapped
#from Bio.Seq import Seq
#from Bio.SeqRecord import SeqRecord

# from Bio.SubsMat import MatrixInfo

# import edlib
BIG_INT = 2**30
NA_REP = 'N/A'
#NAN_REP = 'nan' # https://en.wikipedia.org/wiki/NaN#Display
NUM_INFO_INDEXES = 4
INFO_WT_IDX, INFO_MT_IDX, INFO_ET_IDX, INFO_TPM_IDX = tuple(range(NUM_INFO_INDEXES))

#          '12345678901234567890'
ALPHABET = 'ARNDCQEGHILKMFPSTWYV'

# def aaseq2canonical(aaseq): return aaseq.upper().replace('U', 'X').replace('O', 'X')
def isna(arg): return arg in [None, '', 'NA', 'Na', 'N/A', 'None', 'none', '.']
def col2last(df, colname): return (df.insert(len(df.columns)-1, colname, df.pop(colname)) if colname in df.columns else -1)

def str2str_show_empty(s, empty_str = NA_REP): return (s if s else empty_str)
def str2str_hide_empty(s, empty_str = NA_REP): return (s if (s != empty_str) else '')

def dedup_vals(key2vals): return {k : sorted(set(vs)) for (k, vs) in key2vals.items()} # removed duplicated values

def pep_norm(pep):
    ret = []
    for aa in pep:
        #assert aa in ALPHABET, (F'The amino-acid sequence ({toks[2]}) from ({toks}) does not use the alphabet ({ALPHABET})')
        if aa in ALPHABET: ret.append(aa)
        else: ret.append('X')
    if 'X' in ret: logging.warning(F'{pep} contains non-standard amino acid and is replaced by {ret}')
    return ''.join(ret)

def hamming_dist(a, b):
    assert not (pd.isna(a) and pd.isna(b)), F'The Hamming distance between {a} and {b} cannot be computed. '
    if pd.isna(a) or '' == a: return len(b)
    if pd.isna(b) or '' == b: return len(a)
    assert len(a) == len(b)
    return sum([(0 if (a1 == b1) else 1) for (a1, b1) in zip(a, b)])

def alnscore_penalty(sequence, neighbour, blosum62, gap_open=-11, gap_ext=-1):
    assert len(sequence) == len(neighbour)
    ret = 0
    prev_gap_pos = -1-1
    for i in range(len(sequence)):
        if sequence[i] != neighbour[i]:
            if sequence[i] == '-' or neighbour[i] == '-':
                if i != (prev_gap_pos + 1): # and i != 0 and i != len(sequence)-1:
                    ret += -gap_open
                else:
                    ret += -gap_ext
                prev_gap_pos = i
            else:
                scoremax = blosum62[(sequence[i], sequence[i])]
                ab = (sequence[i], neighbour[i])
                ba = (neighbour[i], sequence[i])
                if ab in blosum62:
                    score = blosum62[ab]
                else:
                    score = blosum62[ba]
                ret += scoremax - score
    return ret
''' 
def bio_substr(s, subpos, sublen, is_zerolen_str_set_to_empty=True, skipchars=['-', '*']):
    ret = []
    i = subpos
    while i < len(s) and len(ret) != sublen:
        if not s[i] in skipchars: ret.append(s[i])
        i += 1
    if is_zerolen_str_set_to_empty and len(ret) != sublen: return ''
    return ''.join(ret)
'''
def bio_endpos(s, subpos, sublen, skipchars=['-', '*']):
    p = subpos
    rlen = 0
    while p < len(s) and rlen < sublen:
        if not s[p] in skipchars: rlen += 1
        p += 1
    if rlen == sublen: return p
    else: return -1
def bio_strfilter(s, skipchars=['-', '*']): return ''.join([c for c in s if (not (c in skipchars))])

def aln_2(aligner, pep1, pep2):
    alns = aligner.align(pep1, pep2)
    best_aln = alns[0]
    best_faa = format(best_aln)
    toks = best_faa.split('\n')
    aln1_str = toks[0] # ret1
    aln2_str = toks[2] # ret2
    i = 0
    j = len(aln1_str)
    while aln1_str[i]   == '-': i += 1
    while aln1_str[j-1] == '-': j -= 1
    aln1_str = aln1_str[i:j]
    aln2_str = aln2_str[i:j]
    return (aln1_str, aln2_str, hamming_dist(aln1_str, aln2_str), alnscore_penalty(aln1_str, aln2_str, aligner.substitution_matrix))

def threewise_aln(aligner, wt_fpep, mt_fpep, et_fpep):
    if wt_fpep != '':
        # aln = pairwise2.align.globalxx(wt_fpep, mt_fpep)
        #result = edlib.align(wt_fpep, mt_fpep, mode="HW", task="path")
        #nice = edlib.getNiceAlignment(result, wt_fpep, mt_fpep)
        #wt_aln_str = nice['query_aligned']
        #mt_aln_str = nice['target_aligned']
        alns = aligner.align(wt_fpep, mt_fpep)
        best_aln = alns[0]        
        best_faa = format(best_aln)
        toks = best_faa.split('\n')
        wt_aln_str = toks[0] # ret1
        mt_aln_str = toks[2] # ret2
        et_aln = []
        j = 0
        for i in range(len(mt_aln_str)):
            if mt_aln_str[i] != '-':
                et_aln.append(et_fpep[j])
                j += 1
            else:
                et_aln.append('-')
        assert j == len(et_fpep), F'{j} == {len(et_fpep)} failed for et_fpep={et_fpep} mt_fpep={mt_fpep} wt_fpep={wt_fpep} wt_aln_str={wt_aln_str} mt_aln_str={mt_aln_str}. '        
        et_aln_str = ''.join(et_aln) # ret3
    else:
        et_aln_str = et_fpep # ret1
        mt_aln_str = mt_fpep # ret2
        wt_aln_str = '-' * len(et_fpep) # ret3
    return (wt_aln_str, mt_aln_str, et_aln_str)

def build_pep_ID_to_seq_info_TPM_dic(fasta_filename, aligner):
    """ This function assumes that there is a one-to-one correspondence between peptide and RNA transcript in fasta_filename. """
    blosum62 = aligner.substitution_matrix
    
    etpep_to_mtpep_list_dic = collections.defaultdict(list)
    mtpep_to_stpep_list_dic = collections.defaultdict(list)
    mtpep_to_wtpep_list_dic = collections.defaultdict(list)
    
    wtpep_to_fpep_list = collections.defaultdict(list)
    stpep_to_fpep_list = collections.defaultdict(list)
    mtpep_to_fpep_list = collections.defaultdict(list)
    etpep_to_fpep_list = collections.defaultdict(list)

    fpep_to_fid_list = collections.defaultdict(list)
    fid_to_seqs = {}
    
    with open(fasta_filename) as file:
        for line in file:
            line = line.strip()
            if line.startswith('>'):
                fid = line.split()[0][1:]
                wt_fpep, st_fpep, mt_fpep, tpm, is_helper_peptide, is_searched_from_self_proteome, = '', '', '', 0, 0, 0
                tkeys = []
                for i, tok in enumerate(line.split()):
                    if i > 0 and len(tok.split('=')) == 2:
                        #print(tok)
                        key, val = tok.split('=')
                        if key == 'WT':
                            if not isna(val): wt_fpep = val # aaseq2canonical(val)
                            tkeys.append(key)
                        if key == 'ST':
                            if not isna(val): st_fpep = val # aaseq2canonical(val)
                            tkeys.append(key)
                        if key == 'MT':
                            if not isna(val): mt_fpep = val # aaseq2canonical(val)
                            tkeys.append(key)
                        if key == 'TPM': 
                            tpm = float(val)
                            tkeys.append(key)
                        if key == 'IsHelperPeptide':
                            is_helper_peptide = int(val)
                            tkeys.append(key)
                        if key == 'IsSearchedFromSelfProteome':
                            is_searched_from_self_proteome = int(val)
                            tkeys.append(key)
                # assert len(wt_fpep) == len(mt_fpep), F'{wt_fpep} == {mt_fpep} failed'
                if len(wt_fpep) != len(mt_fpep) and wt_fpep != '' and not is_helper_peptide:
                    logging.warning(F'{wt_fpep} and {mt_fpep} (wt and mt fasta peptides) have different lengths, so set wt to empty string. ')
                    # wt_fpep = ''
                if '*' in wt_fpep:
                    logging.warning(F'{wt_fpep} (wt fasta peptide) has stop codon (*) in it, so set wt to empty string. ')
                    # wt_fpep = ''
            else:
                et_fpep = line # aaseq2canonical(line)
                if ('WT' not in tkeys) and ('MT' not in tkeys) and ('ST' not in tkeys): mt_fpep = et_fpep
                wt_fpep = pep_norm(wt_fpep)
                st_fpep = pep_norm(st_fpep)
                mt_fpep = pep_norm(mt_fpep)
                et_fpep = pep_norm(et_fpep)
                assert fid not in fid_to_seqs, F'{fid} is duplicated in {fasta_filename}. '
                fid_to_seqs[fid] = ((wt_fpep, st_fpep), mt_fpep, et_fpep, tpm) # {'WT': wt_fpep, 'MT': mt_fpep, 'ET': et_fpep, 'TPM': tpm}
                fpep_to_fid_list[et_fpep].append(fid)
                
                if is_helper_peptide: continue # INFO_WT_IDX, INFO_MT_IDX, INFO_ET_IDX, INFO_TPM_IDX (NUM_INFO_INDEXES)
                
                assert len(et_fpep) == len(mt_fpep), F'len({et_fpep}) == len({mt_fpep}) failed'
                assert len(et_fpep) > 3
                #aln = pairwise2.align.globalxx(wt_fpep, mt_fpep) # format_alignment
                #aln.format_alignment()
                
                wt_aln_str, mt_aln_str, et_aln_str = threewise_aln(aligner, wt_fpep, mt_fpep, et_fpep)
                st_aln_st2, mt_aln_st2, et_aln_st2 = threewise_aln(aligner, st_fpep, mt_fpep, et_fpep)

                #wt_sr = SeqRecord(Seq(wt_fpep), id='WT')
                #mt_sr = SeqRecord(Seq(mt_fpep), id='MT')
                #et_sr = SeqRecord(Seq(et_fpep), id='ET')
                
                #align = MultipleSeqAlignment([wt_sr, mt_sr, et_sr])
                #align.add_sequence("WT", wt_fpep)
                #align.add_sequence("MT", mt_fpep)
                #align.add_sequence("ET", et_fpep)
                #toks = format(align, 'fasta').strip().split('\n')
                #wt_fpep = toks[1]
                #mt_fpep = toks[3]
                #et_fpep = toks[5]
                # assert len(wt_fpep) == 0 or len(wt_fpep) == len(et_fpep)
                
                assert len(et_aln_str) == len(mt_aln_str)
                assert len(et_aln_str) == len(wt_aln_str)
                
                logging.debug(F'ET={et_fpep} MT={mt_fpep} ST={st_fpep} WT={wt_fpep}')
                for peplen in [7,8,9,10,11,12,13]:
                    for beg in range(len(et_fpep)):
                        end = beg + peplen
                        if end >= len(et_fpep): continue
                        et_pep = et_fpep[beg:end]
                        
                        mt_pep = st_pep = wt_pep = ''
                        if mt_fpep != '':
                            et_mt_pep1_aln, et_mt_pep2_aln, et_mt_hamdist, et_mt_scoredist = aln_2(aligner, et_pep, mt_fpep)
                            mt_pep = et_mt_pep2_aln.replace('-', '')
                        if st_fpep != '':
                            mt_st_pep1_aln, mt_st_pep2_aln, mt_st_hamdist, mt_st_scoredist = aln_2(aligner, mt_pep, st_fpep)
                            st_pep = mt_st_pep2_aln.replace('-', '')
                        if wt_fpep != '':
                            mt_wt_pep1_aln, mt_wt_pep2_aln, mt_wt_hamdist, mt_wt_scoredist = aln_2(aligner, mt_pep, wt_fpep)
                            wt_pep = mt_wt_pep2_aln.replace('-', '')
                        
                        if et_pep != '' and mt_pep != '': 
                            etpep_to_mtpep_list_dic[et_pep].append((et_mt_hamdist, et_mt_scoredist*0.5, mt_pep, et_mt_pep1_aln, et_mt_pep2_aln))
                        if mt_pep != '' and st_pep != '': 
                            mtpep_to_stpep_list_dic[mt_pep].append((mt_st_hamdist, mt_st_scoredist*0.5, st_pep, mt_st_pep1_aln, mt_st_pep2_aln))
                        if mt_pep != '' and wt_pep != '': 
                            mtpep_to_wtpep_list_dic[mt_pep].append((mt_wt_hamdist, mt_wt_scoredist*0.5, wt_pep, mt_wt_pep1_aln, mt_wt_pep2_aln))
                        
                        if wt_pep != '': wtpep_to_fpep_list[wt_pep].append(wt_fpep)
                        if mt_pep != '': mtpep_to_fpep_list[mt_pep].append(mt_fpep)
                        if et_pep != '': etpep_to_fpep_list[et_pep].append(et_fpep)
                    '''
                    for pepbeg in range(len(et_aln_str)):
                        # if et_aln_str[pepbeg] == '-': continue
                        #et_pep = et_fpep[pepbeg:pepend]
                        pepend = bio_endpos(et_aln_str, pepbeg, peplen)
                        et_pep_aln = et_aln_str[pepbeg:pepend]
                        et_pep     = bio_strfilter(et_pep_aln)
                        mt_pep_aln = mt_aln_str[pepbeg:pepend]
                        mt_pep     = bio_strfilter(mt_pep_aln)
                        wt_pep_aln = wt_aln_str[pepbeg:pepend]
                        wt_pep     = bio_strfilter(wt_pep_aln)
                        #wt_pep = bio_substr(wt_aln_str, pepbeg, peplen, True)
                        #mt_pep = bio_substr(mt_aln_str, pepbeg, peplen, True)
                        #et_pep = bio_substr(et_aln_str, pepbeg, peplen, True)
                        mt_wt_hdist = hamming_dist(mt_pep_aln, wt_pep_aln)
                        mt_wt_bdist = alnscore_penalty(mt_pep_aln, wt_pep_aln, blosum62) # timeline: mt is after wt

                        if mt_pep != '' and wt_pep != '': mtpep_to_wtpep_list_dic[mt_pep].append((mt_wt_hdist, mt_wt_bdist, wt_pep, mt_pep_aln, wt_pep_aln))
                        if wt_pep != '': wtpep_to_fpep_list[wt_pep].append(wt_fpep)
                        
                        et_mt_hdist = hamming_dist(et_pep_aln, mt_pep_aln)
                        et_mt_bdist = alnscore_penalty(et_pep_aln, mt_pep_aln, blosum62) # timeline: mt is after et
                        if et_pep != '' and mt_pep != '': etpep_to_mtpep_list_dic[et_pep].append((et_mt_hdist, et_mt_bdist, mt_pep, et_pep_aln, mt_pep_aln))
                        if mt_pep != '': mtpep_to_fpep_list[mt_pep].append(mt_fpep)
                        if et_pep != '': etpep_to_fpep_list[et_pep].append(et_fpep)
                    
                    for pepbeg in range(len(et_aln_st2)):
                        pepend = bio_endpos(et_aln_st2, pepbeg, peplen)
                        et_pep_aln = et_aln_st2[pepbeg:pepend]
                        et_pep     = bio_strfilter(et_pep_aln)
                        mt_pep_aln = mt_aln_st2[pepbeg:pepend]
                        mt_pep     = bio_strfilter(mt_pep_aln)
                        st_pep_aln = st_aln_st2[pepbeg:pepend]
                        st_pep     = bio_strfilter(st_pep_aln)
                        #st_pep = bio_substr(st_aln_st2, pepbeg, peplen, True)
                        #mt_pep = bio_substr(mt_aln_st2, pepbeg, peplen, True)
                        #et_pep = bio_substr(et_aln_st2, pepbeg, peplen, True)
                        mt_st_hdist = hamming_dist(mt_pep_aln, st_pep_aln)
                        mt_st_bdist = alnscore_penalty(st_pep_aln, mt_pep_aln, blosum62) # timeline: mt is after st                        
                        if mt_pep != '' and st_pep != '': mtpep_to_stpep_list_dic[mt_pep].append((mt_st_hdist, mt_st_bdist, st_pep, mt_pep_aln, st_pep_aln))
                        if st_pep != '': stpep_to_fpep_list[st_pep].append(st_fpep)
                    '''
                # INFO_WT_IDX, INFO_MT_IDX, INFO_ET_IDX, INFO_TPM_IDX (NUM_INFO_INDEXES)
            
    logging.debug(etpep_to_mtpep_list_dic)
    logging.debug(mtpep_to_wtpep_list_dic)
    
    etpep_to_mtpep_list_dic = dedup_vals(etpep_to_mtpep_list_dic)
    mtpep_to_stpep_list_dic = dedup_vals(mtpep_to_stpep_list_dic)
    mtpep_to_wtpep_list_dic = dedup_vals(mtpep_to_wtpep_list_dic)    
    etpep_to_fpep_list = dedup_vals(etpep_to_fpep_list)
    mtpep_to_fpep_list = dedup_vals(mtpep_to_fpep_list)
    stpep_to_fpep_list = dedup_vals(stpep_to_fpep_list)
    wtpep_to_fpep_list = dedup_vals(wtpep_to_fpep_list)
    fpep_to_fid_list = dedup_vals(fpep_to_fid_list)
    
    return ((etpep_to_mtpep_list_dic, mtpep_to_stpep_list_dic, mtpep_to_wtpep_list_dic), # mutation tracing
            (etpep_to_fpep_list, mtpep_to_fpep_list, stpep_to_fpep_list, wtpep_to_fpep_list), # superstring tracing
            (fpep_to_fid_list, fid_to_seqs)) # ID-TPM tracing

# We need (et_subseq -> mt_subseq) (mt_subseq -> wt_subseq) (subseq -> listof_seqs) (seq -> listof_fastaID) (fastaID -> TPM)
# OUT_CSV_HEADER = ['HLA_type', 'MT_pep', 'WT_pep', 'BindAff', 'WT_BindAff', 'BindLevel', 'Identity', 'Quantification']
OUT_HEADER = ['HLA_type',
    'ET_pep',     'MT_pep',     'ST_pep',     'WT_pep', 
    'ET_BindAff', 'MT_BindAff', 'ST_BindAff', 'WT_BindAff',
    'ET_MT_pairAln', 'ET_ST_pairAln', 'ET_WT_pairAln', 'MT_ST_pairAln', 'MT_WT_pairAln',
    'ET_MT_bitDist', 'ET_ST_bitDist', 'ET_WT_bitDist', 'MT_ST_bitDist', 'MT_WT_bitDist',
    'ET_MT_hamDist', 'ET_ST_hamDist', 'ET_WT_hamDist', 'MT_ST_hamDist', 'MT_WT_hamDist',
    'Identity', 'Quantification', 'BindLevel', 
    'Core', 'Of', 'Gp', 'Gl', 'Ip', 'Il', 'Icore', 'Score_EL', '%Rank_EL', 'Score_BA', '%Rank_BA',
    'PepTrace']

def netmhcpan_result_to_df(infilename, et2mt_mt2wt_2tup_pep2pep, et_mt_wt_3tup_pep2fpep, fpep2fid_fid2finfo_3tup, aligner): 
    # https://stackoverflow.com/questions/35514214/create-nested-dictionary-on-the-fly-in-python
    def fix(f): return lambda *args, **kwargs: f(fix(f), *args, **kwargs)
    etpep_to_mtpep_list_dic, mtpep_to_stpep_list_dic, mtpep_to_wtpep_list_dic = et2mt_mt2wt_2tup_pep2pep
    etpep_to_fpep_list, mtpep_to_fpep_list, stpep_to_fpep_list, wtpep_to_fpep_list = et_mt_wt_3tup_pep2fpep
    fpep_to_fid_list, fid2finfo = fpep2fid_fid2finfo_3tup
    #pep_to_fpep_list, fpep_to_fid_list, fid_to_tpm = pep_fpep_fid_tpm_dic_3tup 
    inheader = None
    rows = []
    with open(infilename) as file:
        for line in file:
            if not line.startswith(' '): continue
            # print(line)
            toks = line.strip().split()
            if toks[0] == 'Pos': 
                assert inheader == None or inheader == toks
                inheader = toks
                assert len(inheader) == 17 or len(inheader) == 15, F'The header-line {line} is invalid.'
            else:
                assert (len(toks) == (len(inheader) - 1) or len(toks) == (len(inheader) + 1)), F'The content-line {line} is invalid'
                if len(toks) == (len(inheader) - 1): row = toks + ['NB'] # no-binding
                if len(toks) == (len(inheader) + 1): row = toks[0:(len(inheader) - 1)] + [toks[(len(inheader))]]
                row[2] = pep_norm(row[2])
                rows.append(row)
    print(F'File={infilename} inheader={inheader}')
    df = pd.DataFrame(rows, columns = inheader)
    df.columns = df.columns.str.replace('HLA', 'MHC')
    mtpep_wtpep_fpep_fid_tpm_ddic_json_dflist = []
    #mtpep_pipesep_dflist = []
    #wtpep_pipesep_dflist = []
    best_mtpep_dflist = []
    best_stpep_dflist = []
    best_wtpep_dflist = []
    etpep_tpm_dflist = []
    etpep_mhc_to_aff = {}
    # etpep_to_mhc_to_aff = collections.defaultdict(dict)
    #def fid_convert(fid, type1 = 'R', type2 = 'D'):
    #    ts = fid.split('_')
    #    if len(ts) >= 2:
    #        if ts[1].startswith(type1):
    #            ts[1] = type2 + ts[1:]
    #    return '_'.join(ts)
    def fid_is_moltype(fid, moltype):
        ts = fid.split('_')
        return len(ts) >= 2 and ts[1].startswith(moltype)
    
    # def fid_to_dna_rna_equiv_fid(fid): return fid.replace('SNV_R','SNV_D').replace('INS_R', 'INS_D').replace('DEL_R', 'DEL_D').replace('FSV_R', 'FSV_D')
    for identity, etpep, mhc, aff in zip(df['Identity'], df['Peptide'], df['MHC'], df['Aff(nM)']):
        etpep_mhc_to_aff[(etpep, mhc)] = float(aff)
        # etpep_to_mhc_to_aff[etpep][mhc] = float(aff)
    df = df.loc[df['Peptide'].isin(set(etpep_to_mtpep_list_dic.keys())),:]
    for identity, etpep, mhc, aff in zip(df['Identity'], df['Peptide'], df['MHC'], df['Aff(nM)']):
        aff = float(aff)
        fids = set(fid for (_, _, mtpep, _, _) in etpep_to_mtpep_list_dic[etpep] for (fpep) in mtpep_to_fpep_list[mtpep] for fid in fpep_to_fid_list[fpep])
        # fids = set()
        # for (_, mtpep) in etpep_to_mtpep_list_dic[etpep]:
        #    for (fpep) in mtpep_to_fpep_list[mtpep]:
        #        for fid in fpep_to_fid_list[fpep]:
        #            fids.add(fid)
        # # fids= set(fid for                                             fpep in etpep_to_fpep_list[etpep] for fid in fpep_to_fid_list[fpep])
        not_dna_fids = set(fid for fid in fids if not fid_is_moltype(fid, 'D'))
        not_rna_fids = set(fid for fid in fids if not fid_is_moltype(fid, 'R'))
        # dna_rna_equiv_fids = set(fid_to_dna_rna_equiv_fid(fid) for fpep in etpep_to_fpep_list[etpep] for fid in fpep_to_fid_list[fpep])
        #etpep_tpm = sum(fid_to_tpm[fid] for fid in dna_rna_equiv_fids)
        not_dna_etpep_tpm = sum(fid2finfo[fid][INFO_TPM_IDX] for fid in not_dna_fids) # (if fid2finfo[fid][INFO_MT_IDX] == fid2finfo[fid][INFO_ET_IDX])
        not_rna_etpep_tpm = sum(fid2finfo[fid][INFO_TPM_IDX] for fid in not_rna_fids) # (if fid2finfo[fid][INFO_MT_IDX] == fid2finfo[fid][INFO_ET_IDX])
        etpep_tpm = max((not_dna_etpep_tpm, not_rna_etpep_tpm))
        
        mtpep_list = []
        stpep_list = []
        wtpep_list = []
        #mtpep2stpeplist = {}
        #mtpep2wtpeplist = {}
        mtpep2peplist = {}
        pep2fidlist = {}
        for et_mt_hdist, et_mt_bdist, mtpep, et_aln1, mt_aln1 in etpep_to_mtpep_list_dic[etpep]:
            mtpep_list.append(mtpep)
            mtpep_aff = etpep_mhc_to_aff.get((mtpep,mhc), BIG_INT)
            mtpep_key = (et_mt_hdist, et_mt_bdist, mtpep_aff, mtpep, 'MT', et_aln1, mt_aln1)
            #mtpep2stpeplist[mtpep_key] = []
            #mtpep2wtpeplist[mtpep_key] = []
            mtpep2peplist[mtpep_key] = []
            for mt_st_hdist, mt_st_bdist, stpep, mt_aln, st_aln in mtpep_to_stpep_list_dic.get(mtpep, []):
                assert len(mt_aln) >= len(mtpep), F'{mt_aln} is not at least as long as {mtpep}'
                assert len(st_aln) >= len(stpep), F'{wt_aln} is not at least as long as {wtpep}'
                stpep_list.append(stpep)
                stpep_aff = etpep_mhc_to_aff.get((stpep,mhc), BIG_INT)
                mtpep2peplist[mtpep_key].append((mt_st_hdist, mt_st_bdist, stpep_aff, stpep, 'ST', mt_aln, st_aln))
                #mtpep2stpeplist[mtpep_key].append((mt_st_hdist, stpep_aff, stpep, 'ST'))
            for mt_wt_hdist, mt_wt_bdist, wtpep, mt_aln, wt_aln in mtpep_to_wtpep_list_dic.get(mtpep, []):
                assert len(mt_aln) >= len(mtpep), F'{mt_aln} is not at least as long as {mtpep}'
                assert len(wt_aln) >= len(wtpep), F'{wt_aln} is not at least as long as {wtpep}'
                wtpep_list.append(wtpep)
                wtpep_aff = etpep_mhc_to_aff.get((wtpep,mhc), BIG_INT)
                mtpep2peplist[mtpep_key].append((mt_wt_hdist, mt_wt_bdist, wtpep_aff, wtpep, 'WT', mt_aln, wt_aln))
                #mtpep2wtpeplist[mtpep_key].append((mt_wt_hdist, wtpep_aff, wtpep, 'WT'))
        for pep in sorted(set([etpep] + mtpep_list + stpep_list + wtpep_list)):
            pep2fidlist[pep] = []
            fpep_list = (etpep_to_fpep_list.get(pep, []) + mtpep_to_fpep_list.get(pep, []) + wtpep_to_fpep_list.get(pep, []))
            # logging.warning(fpep_list)
            for fpep in fpep_list:
                for fid in fpep_to_fid_list[fpep]:
                    pep2fidlist[pep].append(fid)
        pep2fidlist = dedup_vals(pep2fidlist)
        #mtpep_pipesep_dflist.append(str2str_show_empty('|'.join(sorted(list(set(mtpep_list))))))
        #wtpep_pipesep_dflist.append(str2str_show_empty('|'.join(sorted(list(set(wtpep_list))))))
        etpep_tpm_dflist.append(etpep_tpm)
        mtpep2peplist_json = {'/'.join(str(x) for x in k) : '/'.join(str(x) for x in v) for (k,v) in sorted(mtpep2peplist.items())}
        mtpep_wtpep_fpep_fid_tpm_ddic_json = json.dumps((mtpep2peplist_json, pep2fidlist), separators=(',', ':'), sort_keys=True).replace('"', "'")
        mtpep_wtpep_fpep_fid_tpm_ddic_json_dflist.append(mtpep_wtpep_fpep_fid_tpm_ddic_json)
        best_mtpep_key = sorted(mtpep2peplist.keys())[0]
        peplist = mtpep2peplist[best_mtpep_key]
        sorted_st_peplist = sorted(p for p in peplist if p[4] == 'ST')
        sorted_wt_peplist = sorted(p for p in peplist if p[4] == 'WT')
        best_stpep_key = (sorted_st_peplist[0] if sorted_st_peplist else (BIG_INT, BIG_INT, BIG_INT, pd.NA, 'ST', '', ''))
        best_wtpep_key = (sorted_wt_peplist[0] if sorted_wt_peplist else (BIG_INT, BIG_INT, BIG_INT, pd.NA, 'WT', '', ''))
        best_mtpep_dflist.append(best_mtpep_key)
        best_stpep_dflist.append(best_stpep_key)
        best_wtpep_dflist.append(best_wtpep_key)
    #wtpep_aff_dflist = []
    #mtpep_aff_dflist = []
    '''
    bit_dist_dflist = []
    for identity, etpep, mhc, aff in zip(df['Identity'], df['Peptide'], df['MHC'], df['Aff(nM)']):
        #min_wt_aff = BIG_INT
        #min_mt_aff = BIG_INT
        min_bit_dist = BIG_INT
        for mtpep in etpep_to_mtpep_list_dic[etpep]:
            #mt_aff = float(etpep_mhc_to_aff.get((mtpep, mhc), BIG_INT))
            #min_mt_aff = min((min_mt_aff, mt_aff))
            bit_dist = alnscore_penalty(mtpep, etpep) * 0.5
            min_bit_dist = min((min_bit_dist, bit_dist))
            #for wtpep in mtpep_to_wtpep_list_dic[mtpep]:
            #    #wt_aff = float(etpep_mhc_to_aff.get((wtpep, mhc), BIG_INT))
            #    #min_wt_aff = min((min_wt_aff, wt_aff))
        #if min_wt_aff == BIG_INT: min_wt_aff = math.nan
        #if min_mt_aff == BIG_INT: min_mt_aff = math.nan
        if min_bit_dist == BIG_INT: min_bit_dist = math.nan
        #wtpep_aff_dflist.append(min_wt_aff)
        #mtpep_aff_dflist.append(min_mt_aff)
        bit_dist_dflist.append(min_bit_dist)
    '''
    
    df['Quantification'] = etpep_tpm_dflist
    df['ET_pep'] = df['Peptide']
    df['MT_pep'] = [x[3] for x in best_mtpep_dflist] # mtpep_pipesep_dflist
    df['ST_pep'] = [x[3] for x in best_stpep_dflist] # wtpep_pipesep_dflist
    df['WT_pep'] = [x[3] for x in best_wtpep_dflist] # wtpep_pipesep_dflist
    
    df['ET_BindAff'] = df['Aff(nM)'].astype(float)
    df['MT_BindAff'] = [x[2] for x in best_mtpep_dflist]
    df['ST_BindAff'] = [x[2] for x in best_stpep_dflist]
    df['WT_BindAff'] = [x[2] for x in best_wtpep_dflist]
    
    def aln_2a(aligner, a, b): return (('', '', BIG_INT, BIG_INT) if (pd.isna(a) or pd.isna(b) or isna(a) or isna(b)) else aln_2(aligner, a, b))
    # ET crw MT, MT crw (ST and/or WT), where crw denotes "cross-react with"
    aln_ET_MT = [aln_2a(aligner, a, b) for (a, b) in zip(df['ET_pep'], df['MT_pep'])]
    aln_ET_ST = [aln_2a(aligner, a, b) for (a, b) in zip(df['ET_pep'], df['ST_pep'])]
    aln_ET_WT = [aln_2a(aligner, a, b) for (a, b) in zip(df['ET_pep'], df['WT_pep'])]
    aln_MT_ST = [aln_2a(aligner, a, b) for (a, b) in zip(df['MT_pep'], df['ST_pep'])]
    aln_MT_WT = [aln_2a(aligner, a, b) for (a, b) in zip(df['MT_pep'], df['WT_pep'])]
    
    # This info should be concordant with the one from PepTrace
    df['ET_MT_pairAln'] = [(x[0] + '/' + x[1]) for x in aln_ET_MT]
    df['ET_ST_pairAln'] = [(x[0] + '/' + x[1]) for x in aln_ET_ST]
    df['ET_WT_pairAln'] = [(x[0] + '/' + x[1]) for x in aln_ET_WT]
    df['MT_ST_pairAln'] = [(x[0] + '/' + x[1]) for x in aln_MT_ST]
    df['MT_WT_pairAln'] = [(x[0] + '/' + x[1]) for x in aln_MT_WT]
    df['ET_MT_hamDist'] = [x[2] for x in aln_ET_MT]
    df['ET_ST_hamDist'] = [x[2] for x in aln_ET_ST]
    df['ET_WT_hamDist'] = [x[2] for x in aln_ET_WT]
    df['MT_ST_hamDist'] = [x[2] for x in aln_MT_ST]
    df['MT_WT_hamDist'] = [x[2] for x in aln_MT_WT]
    df['ET_MT_bitDist'] = [x[3] for x in aln_ET_MT]
    df['ET_ST_bitDist'] = [x[3] for x in aln_ET_ST]
    df['ET_WT_bitDist'] = [x[3] for x in aln_ET_WT]
    df['MT_ST_bitDist'] = [x[3] for x in aln_MT_ST]
    df['MT_WT_bitDist'] = [x[3] for x in aln_MT_WT]
    
    #df['ET_MT_hamdist'] = [x[0] for x in best_mtpep_dflist]
    #df['ET_ST_hamdist'] = [hamming_dist(a,b) for (a,b) in zip(df['ET_pep'], df['ST_pep'])]
    #df['ET_WT_hamdist'] = [hamming_dist(a,b) for (a,b) in zip(df['ET_pep'], df['WT_pep'])]
    #df['MT_ST_hamdist'] = [x[0] for x in best_stpep_dflist]
    #df['MT_WT_hamdist'] = [x[0] for x in best_wtpep_dflist]

    #df['MT_BindAff'] = best_mtpep_dflist #mtpep_aff_dflist
    #df['WT_BindAff'] = #wtpep_aff_dflist
    # blosum62 = substitution_matrices.load("BLOSUM62")
    # df['ET_MT_bitdist'] = [(alnscore_penalty(mtpep, etpep, blosum62) * 0.5) for (mtpep, etpep) in zip(df['MT_pep'], df['ET_pep'])]  # bit_dist_dflist    
    df['PepTrace'] = mtpep_wtpep_fpep_fid_tpm_ddic_json_dflist
    df['HLA_type'] = df['MHC']
    # df['BindAff'] = df['Aff(nM)'].astype(float)
    # 'BindLevel', 'Identity' are kept as they are
    return df[OUT_HEADER]

def main():
    description = 'This script parses the output of netMHCpan into a tsv file'
    epilog = 'Abbreviations: WT for wild-type, MT for mutant-type, ET for experimental-type (please just ignore ET for now)'
    parser = argparse.ArgumentParser(description = description, epilog = epilog) # formatter_class = argparse.ArgumentDefaultsHelpFormatter is not used    
    parser.add_argument('-f', '--fasta-file',     help = 'fasta file that netMHCpan took as input to generate the netmhcpan-file', required = True)
    parser.add_argument('-n', '--netmhcpan-file', help = 'file containing the output of netMHCpan', required = True)
    parser.add_argument('-o', '--out-tsv-file',   help = F'''output TSV file with the following columns: {', '.join(OUT_HEADER)}''', required = True)
    parser.add_argument('-a', '--binding-affinity-thres', help = F'binding affinity threshold in nanoMolar above which the peptide-MHC is filtered out '
            F'(higher nanoMolar value means lower binding affinity)', required = True, type = float)
    #parser.add_argument('-l', '--bind-levels', help = F'comma-separated tokens describing bind levels '
    #        F'(a combination of SB/WB/NB denoting strong/weak/no binding)', required = False, default = 'SB,WB,NB')
    parser.add_argument('-l', '--lengths', help = F'comma-separated tokens describing peptide lengths. '
            F'(same as the ones used by netMHCpan series)', required = False, default = '8,9,10,11,12')
    
    args = parser.parse_args()
    peplens = [int(x) for x in args.lengths.split(',')]
    
    blosum62 = substitution_matrices.load("BLOSUM62")
    aligner = Align.PairwiseAligner()
    aligner.mode = 'global'
    #aligner.match_score = 2
    #aligner.mismatch_score = -1
    aligner.open_gap_score = -11
    aligner.extend_gap_score = -1
    aligner.target_end_gap_score = 0.0 # first is in second
    aligner.query_end_gap_score = -(2**10) # first is in second
    #aligner.query_end_gap_score = 0.0
    aligner.substitution_matrix = blosum62
    
    et2mt_mt2wt_2tup_pep2pep, et_mt_wt_3tup_pep2fpep, fpep2fid_fid2finfo_3tup = build_pep_ID_to_seq_info_TPM_dic(args.fasta_file, aligner)
    df1 = netmhcpan_result_to_df(args.netmhcpan_file, et2mt_mt2wt_2tup_pep2pep, et_mt_wt_3tup_pep2fpep, fpep2fid_fid2finfo_3tup, aligner)
    df2 = df1[(df1['ET_BindAff'] <= args.binding_affinity_thres) 
            #& (df1['BindLevel'].isin(args.bind_levels.split(',')))
            & (df1['ET_pep'].str.len().isin(peplens))
            & (df1['MT_pep'] != df1['WT_pep']) 
            & (df1['ET_pep'] != df1['WT_pep'])]
    df3 = df2.drop_duplicates(subset=['HLA_type','ET_pep'])
    df4 = df3.copy()
    col_ET_BindAff = df4['ET_BindAff'] #.copy()
    col_ST_BindAff = df4['ST_BindAff'] #.copy()
    col_WT_BindAff = df4['WT_BindAff'] #.copy()
    df4['ET_MT_Agretopicity'] = (col_ET_BindAff / df4['MT_BindAff'])
    df4['ST_Agretopicity']    = (col_ET_BindAff / col_ST_BindAff)
    df4['Agretopicity']       = (col_ET_BindAff / col_WT_BindAff)
    col2last(df4, 'PepTrace')
    df4.to_csv(args.out_tsv_file, header = 1, sep = '\t', index = 0, na_rep = NA_REP) # not NAN_REP # used tsv instead of csv to prevent conflict with min-json encoding
    
if __name__ == '__main__': main()

