import argparse,collections,copy,csv,getopt,logging,math,multiprocessing,os,statistics,sys,subprocess # ,os
import pandas as pd
import numpy as np
import scipy

from Bio import Align
from Bio.Align import substitution_matrices

from Bio.SeqIO.FastaIO import SimpleFastaParser

from math import log, exp

import pysam

NA_REP = 'N/A'
BIG_INT = 2**32
DROP_COLS = [
    'ET_pep',     #'MT_pep',     'ST_pep',     'WT_pep',
    'ET_BindAff', # 'MT_BindAff', 'ST_BindAff', 'WT_BindAff',
    'ET_MT_pairAln', 'ET_ST_pairAln', 'ET_WT_pairAln', # 'MT_ST_pairAln', 'MT_WT_pairAln',
    'ET_MT_bitDist', 'ET_ST_bitDist', 'ET_WT_bitDist', # 'MT_ST_bitDist', 'MT_WT_bitDist',
    'ET_MT_hamDist', 'ET_ST_hamDist', 'ET_WT_hamDist', # 'MT_ST_hamDist', 'MT_WT_hamDist',
    'ET_MT_Agretopicity']

def u2d(s): return '--' + s.replace('_', '-')

def aaseq2canonical(aaseq): return aaseq.upper().replace('U', 'X').replace('O', 'X')

def col2last(df, colname): return (df.insert(len(df.columns)-1, colname, df.pop(colname)) if colname in df.columns else -1) # return operation status
def dropcols(df, colnames = DROP_COLS):
    xs = [x for x in colnames if x in df.columns]
    return df.drop(xs, axis = 1) # return DataFrame

NA_VALS = [None, '', 'NA', 'Na', 'N/A', 'None', 'none', '.']
def isna(arg): return arg in NA_VALS
    
# Please note that ref2 is not used in identity check
def var_vcf2vep(vcfrecord):
    def replace_emptystring_by_dash(ch): return (ch if (0 < len(ch)) else '-')
    chrom, pos, ref, alts = (vcfrecord.chrom, vcfrecord.start, vcfrecord.ref, vcfrecord.alts)
    alts = [a for a in alts if not a.startswith('<')]
    if not alts: return (F'{chrom}_{pos}_-/-', -1, -1, -1)
    pos = int(pos)
    alts_len = min([len(alt) for alt in alts])
    i = 0
    while i < len(ref) and i < alts_len and all([(ref[i] == alt[i]) for alt in alts]): i += 1
    chrom2 = chrom
    pos2 = pos + i + 1
    ref2 = replace_emptystring_by_dash(ref[i:])
    alt2 = '/'.join([replace_emptystring_by_dash(alt[i:]) for alt in alts])
    assert ((('tAD' in vcfrecord.info) and len(vcfrecord.info['tAD']) == 2) 
            or (('diRDm' in vcfrecord.info) and len(vcfrecord.info['diRDm']) == 1 and
                ('diADm' in vcfrecord.info) and len(vcfrecord.info['diADm']) == 1))
    refAD = (vcfrecord.info['tAD'][0] if ('tAD' in vcfrecord.info) else vcfrecord.info['diRDm'][0])
    tumorAD = (vcfrecord.info['tAD'][1] if ('tAD' in vcfrecord.info) else vcfrecord.info['diADm'][0])
    return ('_'.join([chrom2, str(pos2), ref2 + '/' + alt2]), vcfrecord.qual, int(refAD), int(tumorAD)) # tumor allele depths

def vep_lenient_equal(vep1, vep2):
    chrom1, pos1, alleles1 = vep1.split('_')
    alleles1 = alleles1.split('/')
    chrom2, pos2, alleles2 = vep2.split('_')
    alleles2 = alleles2.split('/')
    if chrom1 == chrom2 and pos1 == pos2:
        alen = min((len(alleles1), len(alleles2)))
        for alt1 in alleles1[1:]:
            for alt2 in alleles2[1:]:
                ref1 = alleles1[0]
                ref2 = alleles2[0]
                if ref1[0] == ref2[0] and alt1[0] == alt2[0]:
                    return True
    return False

def pep_norm(pep):
    ALPHABET = 'ARNDCQEGHILKMFPSTWYV'
    ret = []
    for aa in pep:
        #assert aa in ALPHABET, (F'The amino-acid sequence ({toks[2]}) from ({toks}) does not use the alphabet ({ALPHABET})')
        if aa in ALPHABET: ret.append(aa)
        else: ret.append('X')
    if 'X' in ret: logging.warning(F'{pep} contains non-standard amino acid and is replaced by {ret}')
    return ''.join(ret)

def build_pMHC2stab(netmhcstabpan_file):
    pmhc2halfperc = {}
    inheader = [
        'pos', 'HLA', 'peptide', 'Identity', 'Pred', 'Thalf(h)', 
        '%Rank_Stab', '1-log50k', 'Aff(nM)', '%Rank_aff', 'Combined', 'Combined_%rank', 'BindLevel']
    with open(netmhcstabpan_file) as file:
        for lineno, line in enumerate(file):
            if not line.startswith(' '): continue
            # print(line)
            toks = line.strip().split()
            if toks[0] in ['Pos', 'pos']:
                assert inheader == None or inheader == toks, F'Line ( {line.strip()} ) is invalid at lineno {lineno}. '
                inheader = toks
                assert len(inheader) == 13, F'The header-line {line} is invalid.'
            else:
                assert (len(toks) == (len(inheader) - 1) or len(toks) == (len(inheader) + 1)), F'The content-line ({toks}) of length={len(toks)} is invalid compared with the header ({inheader} of length={len(inheader)})'
                #if len(toks) == (len(inheader) - 1): row = toks + ['NB'] # no-binding
                #if len(toks) == (len(inheader) + 1): row = toks[0:(len(inheader) - 1)] + [toks[(len(inheader))]]
                pep = pep_norm(toks[2])
                mhc = toks[1]
                pmhc2halfperc[(pep,mhc)] = (toks[5], toks[6]) # half-life and its percentile
    return pmhc2halfperc

def getR(neo_seq, iedb_seqs, aligner):
    align_scores = []
    a = 26
    k = 4.86936
    for pathogen_seq in iedb_seqs:
        alns = aligner.align(neo_seq, pathogen_seq)
        if alns:
            localds_core = max([aln.score for aln in alns])
            align_scores.append(localds_core)

    bindingEnergies = list(map(lambda x: -k * (a - x), align_scores))
    ## This tweak ensure that we get similar results compared with antigen.garnish at
    ## https://github.com/andrewrech/antigen.garnish/blob/main/R/antigen.garnish_predict.R#L86
    # bindingEnergies = [max(bindingEnergies)] 
    sumExpEnergies = sum([exp(x) for x in bindingEnergies])
    Zk = (1 + sumExpEnergies)
    return float(sumExpEnergies) / Zk

def getiedbseq(iedb_path):
    iedb_seq = []
    with open(iedb_path, 'r') as fin: # from /data8t_2/zzt/antigen.garnish/iedb.fasta (antigen.garnish data file)
        for t, seq in SimpleFastaParser(fin):
            iedb_seq.append(seq)
    return iedb_seq

def iedb_fasta_to_dict(iedb_path):
    ret = {}
    with open(iedb_path, 'r') as iedb_fasta_file:
        for fasta_id, fasta_seq in SimpleFastaParser(iedb_fasta_file):
            if fasta_id in ret:
                assert fasta_seq == ret[fasta_id], F'The FASTA_ID {fasta_id} is duplicated in the file {iedb_path}'
            ret[fasta_id] = fasta_seq
    return ret

def aligner(seq1,seq2):
        
    matrix = matlist.blosum62
    gap_open = -11
    gap_extend = -1
    aln = pairwise2.align.localds(aaseq2canonical(seq1), aaseq2canonical(seq2.strip().split('+')[0]), matrix, gap_open, gap_extend)
    return aln

def allblast(query_seqs, target_fasta, output_file):
    os.system(F'mkdir -p {output_file}.tmp')
    query_fasta = F'{output_file}.tmp/foreignness_query.all.fasta'
    with open(query_fasta, 'w') as query_fasta_file:
        warned_seqs = set([])
        for pipesep_query_seq in query_seqs:
            for query_seq in pipesep_query_seq.split('|'):
                if not isna(query_seq):
                    query_fasta_file.write(F'>{query_seq}\n{query_seq}\n')
                elif not query_seq in warned_seqs:
                    logging.warning(F'The query seq {query_seq} is invalid and is therefore skipped (not put in file {query_fasta}). ')
                    warned_seqs.add(query_seq)
    # from https://github.com/andrewrech/antigen.garnish/blob/main/R/antigen.garnish_predict.R
    cmd = F'''blastp \
        -query {query_fasta} -db {target_fasta} \
        -evalue 100000000 -matrix BLOSUM62 -gapopen 11 -gapextend 1 \
        -out {query_fasta}.blastp_iedbout.csv -num_threads 8 \
        -outfmt '10 qseqid sseqid qseq qstart qend sseq sstart send length mismatch pident evalue bitscore' '''
    logging.debug(cmd)
    os.system(cmd)
    ret = collections.defaultdict(list)
    with open(F'{query_fasta}.blastp_iedbout.csv') as blastp_csv:
        for line in blastp_csv:
            tokens = line.strip().split(',')
            qseqid = tokens[0]
            sseq = tokens[5].replace('-', '')
            is_canonical = all([(aa in 'ARNDCQEGHILKMFPSTWYV') for aa in sseq])
            if is_canonical: ret[qseqid].append(sseq)
    return ret

def main():
    description = 'This script gathers the results from netMHC, netMHCstabpan and other tools. '
    
    parser = argparse.ArgumentParser(description = description, formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument(u2d('netmhcstabpan_file'), help = 'The netmhcstabpan input file. ', required = True)
    
    parser.add_argument('-i', u2d('input_file'), help = 'Input file generated by bindstab_filter.py (binding-stability filter)', required = True)
    parser.add_argument('-I', u2d('iedb_fasta'), help = 'path to IEDB reference fasta file containing pathogen-derived immunogenic peptides', required = True)
    parser.add_argument('-o', u2d('output_file'), help = 'output file to store results of neoantigen prioritization', required = True)
    
    parser.add_argument('-D', u2d('dna_detail'), help = 'Optional input file providing SourceAlterationDetail for SNV and InDel variants from DNA-seq', default = '')
    parser.add_argument('-R', u2d('rna_detail'), help = 'Optional input file providing SourceAlterationDetail for SNV and InDel variants from RNA-seq', default = '')
    parser.add_argument('-F', u2d('fusion_detail'), help = 'Optional input file providing SourceAlterationDetail for fusion variants from RNA-seq', default = '')
    parser.add_argument('-S', u2d('splicing_detail'), help = 'Optional input file providing SourceAlterationDetail for splicing variants from RNA-seq', default = '')
    
    parser.add_argument('-t',  u2d('alteration_type'), default = 'snv,indel,fsv,fusion,splicing',
            help = 'type of alterations detected, can be a combination of (snv, indel, fsv, sv, and/or fusion separated by comma)')
    
    parser.add_argument(u2d('immuno_strength_p_value'), default = 0.02, type=float,
            help = 'The p-value threshold for the mannwhitneyu test for recognized versus unrecognized neo-abundance. ')
    parser.add_argument(u2d('immuno_strength_effect_size'), default = 1.5, type=float,
            help = 'The median of recognized neo-abundance to the median of unrecognized one, '
            'above which/below the inverse of which the immuno-strength is high if p-value is also low. ')
    
    parser.add_argument(u2d('dna_vcf'), default = '',
            help = 'VCF file (which can be block-gzipped) generated by calling small variants from DNA-seq data')
    parser.add_argument(u2d('rna_vcf'), default = '',
            help = 'VCF file (which can be block-gzipped) generated by calling small variants from RNA-seq data')

    parser.add_argument(u2d('function'), default = '',
            help = 'The keyword rerank means using existing stats (affinity, stability, etc.) to re-rank the neoantigen candidates')
    parser.add_argument(u2d('truth_file'), default = '',
            help = 'File containing ground-truth of immunogenicity for some tested pMHCs')
    parser.add_argument(u2d('truth_patientID_prefix'), default = '',
            help = 'Prefix of the PatientID to select rows from the truth-file')
    parser.add_argument(u2d('truth_patientID'), default = '',
            help = 'PatientID to select rows from the truth-file')
    
    args = parser.parse_args()
    paramset = args
    logging.info(paramset)
    
    if not isna(args.truth_file):
        def norm_hla(h): return h.astype(str).str.replace('*', '', regex=False).str.replace('HLA-', '', regex=False).str.replace(':', '', regex=False).str.strip()
        origdata = pd.read_csv(args.truth_file)
        origdata['PatientID'] = origdata['PatientID'].astype(str)
        if args.truth_patientID_prefix:
            origdata = origdata[origdata['PatientID'].str.startswith(args.truth_patientID_prefix)]
        else:
            origdata = origdata[origdata['PatientID'] == args.truth_patientID]
        origdata['peptideMHC'] = origdata['MT_pep'].astype(str) + '/' + norm_hla(origdata['HLA_type'])
        
        keptdata = pd.read_csv(args.output_file + '.expansion', sep='\t')
        keptdata = keptdata[keptdata['ET_pep'] == keptdata['MT_pep']]
        keptdata['peptideMHC'] = keptdata['MT_pep'].astype(str) + '/' + norm_hla(keptdata['HLA_type'])
        
        data1 = dropcols(keptdata)
        data1 = data1.sort_values(['%Rank_EL', 'MT_BindAff'])

        combdata1 = pd.merge(origdata, data1, how = 'left', left_on = 'peptideMHC', right_on = 'peptideMHC')
        combdata1.to_csv(args.output_file + '.validation' , header=1, sep='\t', index=0, na_rep = 'NA')
        
        data2 = data1.drop_duplicates(subset=['MT_pep'], keep='first', inplace=False)
        combdata2 = pd.merge(origdata, data2, how='left', left_on='MT_pep', right_on='MT_pep')
        combdata1.to_csv(args.output_file + '.peptide-validation' , header=1, sep='\t', index=0, na_rep = 'NA')
        
        exit(0)
    
    def openif(fname): return (open(fname) if fname else [])
    
    dna_snv_indel_file = openif(args.dna_detail)
    rna_snv_indel_file = openif(args.rna_detail)
    fusion_file = openif(args.fusion_detail)
    splicing_file = openif(args.splicing_detail)
     
    dnaseq_small_variants_file = (pysam.VariantFile(args.dna_vcf, 'r') if args.dna_vcf else [])
    rnaseq_small_variants_file = (pysam.VariantFile(args.rna_vcf, 'r') if args.rna_vcf else [])

    dna_snvindel = [line for line in dna_snv_indel_file]
    rna_snvindel = [line for line in rna_snv_indel_file]
    fusion = [line for line in fusion_file]
    splicing = [line for line in splicing_file]
    
    if dna_snv_indel_file: dna_snv_indel_file.close()
    if rna_snv_indel_file: rna_snv_indel_file.close()
    if fusion_file: fusion_file.close()
    if splicing_file: splicing_file.close() 

    candidate_file = open(args.input_file)
    reader = csv.reader(candidate_file, delimiter='\t')
    fields=next(reader)
    fields.append("Foreignness")
    fields.append("DNA_QUAL")
    fields.append("DNA_refDP")
    fields.append("DNA_altDP")
    fields.append("RNA_QUAL")
    fields.append("RNA_refDP")
    fields.append("RNA_altDP")
    fields.append("SourceAlterationDetail")
    fields.append('IsFrameshift')
    data_raw = []
    agre_exist = []
    et_pep_idx = fields.index('ET_pep')
    wt_pep_idx = fields.index('WT_pep')
    identity_idx = fields.index('Identity')
    
    for line1 in reader:
        assert not '|' in line1[et_pep_idx], F'The line {line1} is characterized by invalid ET_pep {line1[et_pep_idx]}'
        if line1[et_pep_idx] in line1[wt_pep_idx].split('|'):
            logging.info(F'Skipping the peptide candidate {line1[et_pep_idx]} because it is found in the list of wild-type peptides {line1[wt_pep_idx]}')
        line = copy.deepcopy(line1)
        line.append(-1)
        dna_varqual = 0
        dna_ref_depth = 0
        dna_alt_depth = 0
        rna_varqual = 0
        rna_ref_depth = 0
        rna_alt_depth = 0
        line_info_string = ""
        is_frameshift = False
        identity = line[identity_idx]
        if dna_snvindel and rna_snvindel and ((identity.strip().split('_')[0] in ["SNV", 'INS', 'DEL', 'INDEL', 'FSV'] or identity.strip().split('_')[0].startswith("INDEL"))):
            fastaID = identity.strip().split('_')[1]
            selected_snvindel = (rna_snvindel if (fastaID[0] == 'R') else dna_snvindel)
            line_num = int(fastaID[1:])
            snv_indel_line = selected_snvindel[line_num-1]
            ele = snv_indel_line.strip().split('\t')
            if len(ele) == 14: # annotation software is vep
                annotation_info = ["Uploaded_variation","Location","Allele","Gene","Feature","Feature_type",
                                    "Consequence","cDNA_position","CDS_position","Protein_position","Amino_acids","Codons","Existing_variation","Extra"]
                for i in range(0,len(ele),1):
                    line_info_string+=annotation_info[i]+"$"+ele[i]+"#"
                    if annotation_info[i] == 'Consequence' and (ele[i].lower().startswith('frameshift') or ele[i].lower().startswith('frame_shift')):
                        is_frameshift = True
                chrom, pos, alts = ele[0].split('_')
                if dnaseq_small_variants_file:
                    for vcfrecord in dnaseq_small_variants_file.fetch(chrom, int(pos) - 6, int(pos) + 6):
                        vepvar, varqual, varRD, varAD = var_vcf2vep(vcfrecord)
                        if vep_lenient_equal(vepvar, ele[0]):
                            dna_varqual = max((dna_varqual, varqual))
                            dna_ref_depth = max((dna_ref_depth, varRD))
                            dna_alt_depth = max((dna_alt_depth, varAD))
                if rnaseq_small_variants_file:
                    for vcfrecord in rnaseq_small_variants_file.fetch(chrom, int(pos) - 6, int(pos) + 6):
                        vepvar, varqual, varRD, varAD = var_vcf2vep(vcfrecord)
                        if vep_lenient_equal(vepvar, ele[0]):
                            rna_varqual = max((rna_varqual, varqual))
                            rna_ref_depth = max((rna_ref_depth, varRD))
                            rna_alt_depth = max((rna_alt_depth, varAD))
            elif len(ele)==11:
                annotation_info = ["CHROM","POS","ID","REF","ALT","QUAL","FILTER","INFO","FORMAT","normal","tumor"]
                for i in range(0,len(ele),1):
                    line_info_string+=annotation_info[i]+"$"+ele[i]+"#"
            else:
                continue
        elif fusion and (identity.strip().split('_')[0]=="FUS"):
            line_str = identity.strip().split('_')[1]
            if not line_str.startswith('R'): 
                logging.warning(F'The record {identity} cannot be processed and is therefore skipped. ')
            else:
                line_num = int(identity.strip().split('_')[1].lstrip('DRP'))
                fusion_line = fusion[line_num-1]
                ele = fusion_line.strip().split('\t')
                annotation_info = ["FusionName","JunctionReadCount","SpanningFragCount","est_J","est_S","SpliceType","LeftGene","LeftBreakpoint",
                                    "RightGene","RightBreakpoint","LargeAnchorSupport","FFPM","LeftBreakDinuc","LeftBreakEntropy","RightBreakDinuc",
                                    "RightBreakEntropy","annots","CDS_LEFT_ID","CDS_LEFT_RANGE","CDS_RIGHT_ID","CDS_RIGHT_RANGE","PROT_FUSION_TYPE",
                                    "FUSION_MODEL","FUSION_CDS","FUSION_TRANSL","PFAM_LEFT","PFAM_RIGHT"]
                for i in range(0, len(ele),1):
                    line_info_string+=annotation_info[i]+"$"+ele[i]+"#"
        elif splicing and (identity.strip().split('_')[0]=="SP"):
            line_str = identity.strip().split('_')[1]
            if not line_str.startswith('R'): 
                logging.warning(F'The record {identity} cannot be processed and is therefore skipped. ')
            else:
                line_num = int(line_str.lstrip('R'))
                splicing_line = splicing[line_num-1]
                ele = splicing_line.strip().split('\t')
                annotation_info = ["chrom","txStart","txEnd","isoform","protein","strand","cdsStart","cdsEnd","gene","exonNum",
                                    "exonLens","exonStarts","ensembl_transcript"]
                for i in range(0, len(ele),1):
                    line_info_string+=annotation_info[i]+"$"+ele[i]+"#"
        line.append(dna_varqual)
        line.append(dna_ref_depth)
        line.append(dna_alt_depth)
        line.append(rna_varqual)
        line.append(rna_ref_depth)
        line.append(rna_alt_depth)
        line.append(line_info_string if line_info_string else NA_REP)
        line.append(is_frameshift)
        data_raw.append(line)
    
    picked_rows = []
    alt_type = args.alteration_type.replace(' ', '').strip().split(',')
    for line in data_raw:
        atype = line[identity_idx].strip().split('_')[0]
        if (atype == 'SP'): atype='SPLICING'
        if (atype == 'FUS'): atype='FUSION'
        if atype.lower() in alt_type: picked_rows.append(line)
        else: logging.debug(F'Skipping {line} because {atype} is-not-in {alt_type}')
    # data can be emtpy (https://stackoverflow.com/questions/44513738/pandas-create-empty-dataframe-with-only-column-names)
    data=pd.DataFrame(picked_rows, columns = fields)
    
    netmhcstabpan_pmhc2halfperc = build_pMHC2stab(args.netmhcstabpan_file)
    halflives = [netmhcstabpan_pmhc2halfperc[(pep, hla)][0] for (pep, hla) in zip(data['ET_pep'], data['HLA_type'])]
    data['BindStab'] = halflives
    
    qseq2sseqs = allblast(sorted(list(set(list(data['ET_pep'])))), args.iedb_fasta, args.output_file)
    wseq2sseqs = allblast(sorted(list(set(list(data['WT_pep'])))), args.iedb_fasta, args.output_file)
    ERs = []
    WRs = []
    aligner = Align.PairwiseAligner()
    aligner.mode = 'local'
    aligner.open_gap_score = -11
    aligner.extend_gap_score = -1
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    for pipesep_qseq, pipesep_wseq in zip(list(data['ET_pep']), list(data['WT_pep'])):
        pipeR = [getR(qseq, qseq2sseqs[qseq], aligner) for qseq in pipesep_qseq.split('|')]
        assert len(pipeR) == 1
        ERs.append(min(pipeR))
        pipeR = [getR(wseq, wseq2sseqs[wseq], aligner) for wseq in pipesep_wseq.split('|')]
        WRs.append(max(pipeR))
    data['Foreignness'] = ERs
    data['WT_Foreignness'] = WRs
    data['ForeignDiff'] = np.array(ERs) - np.array(WRs)
    
    data.ET_BindAff = data.ET_BindAff.astype(float)
    data.MT_BindAff = data.MT_BindAff.astype(float)
    data.WT_BindAff = data.WT_BindAff.astype(str).replace(NA_VALS, np.nan).astype(float)
    data.BindStab = data.BindStab.astype(float)
    data.Foreignness = data.Foreignness.astype(float)
    data.Agretopicity = data.Agretopicity.astype(str).replace(NA_VALS, np.nan).astype(float)
    data.Quantification = data.Quantification.astype(float)
    
    col2last(data, 'PepTrace')
    keptdata = data
    
    keptdata1 = keptdata[keptdata['ET_pep']==keptdata['MT_pep']]
    keptdata2 = dropcols(keptdata1)
    
    etpep2list = []
    for etpep, mtpep in zip(keptdata['ET_pep'], keptdata['MT_pep']):
        if len(etpep) != len(mtpep):
            etpep2 = etpep.lower()
        else:
            etpep2 = []
            for etaa, mtaa in zip(etpep, mtpep):
                etpep2.append(etaa if (etaa == mtaa) else etaa.lower())
            etpep2 = ''.join(etpep2)
        etpep2list.append(etpep2)
    keptdata['ET_pep'] = etpep2list
    keptdata['MT_peplen'] = [len(mtpep) for mtpep in keptdata['MT_pep']]
    keptdata = keptdata.sort_values(['HLA_type', 'MT_peplen', 'MT_pep', 'ET_pep'])
    col2last(keptdata, 'PepTrace')
    keptdata.to_csv(args.output_file + '.expansion', sep='\t', header=1, index=0, na_rep='NA')
    
    keptdata2.to_csv(args.output_file,               sep='\t', header=1, index=0, na_rep='NA')
    
    if dnaseq_small_variants_file: dnaseq_small_variants_file.close()
    if rnaseq_small_variants_file: rnaseq_small_variants_file.close()
    
if __name__ == '__main__':
    main()

