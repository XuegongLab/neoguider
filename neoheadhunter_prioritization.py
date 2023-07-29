import argparse,copy,csv,getopt,logging,math,multiprocessing,os,statistics,sys,subprocess # ,os
import pandas as pd
import numpy as np
import scipy
from Bio import pairwise2
from Bio.SubsMat import MatrixInfo as matlist
from Bio.SeqIO.FastaIO import SimpleFastaParser
from math import log, exp

import pysam

def u2d(s): return '--' + s.replace('_', '-')

def aaseq2canonical(aaseq): return aaseq.upper().replace('U', 'X').replace('O', 'X')

def col2last(df, colname): return (df.insert(len(df.columns)-1, colname, df.pop(colname)) if colname in df.columns else -1)
def dropcols(df, colnames):
    xs = [x for x in colnames if x in df.columns]
    df.drop(xs, axis = 1)

def isna(arg): return arg in [None, '', 'NA', 'Na', 'None', 'none', '.']

def get_avg_depth_from_rna_depth_filename(rna_depth):
    regsize = -1
    sumDP = -1
    with open(rna_depth) as file:
        for i, line in enumerate(file):
            #sumDP, regsize = line.split('\t')
            if line.startswith('exome_total_bases'): regsize = int(line.split()[1])
            if line.startswith('exome_total_depth'): sumDP   = int(line.split()[1])
    assert regsize > -1
    assert sumDP > -1
    return float(sumDP) / float(regsize)

# Result using this measure is dependent on RNA-seq read length, so it is not recommended to use it (and hence not used).
def get_total_transcript_num_from_rna_flagstat_filename(rna_flagstat):    
    prim_mapped = -1
    prim_duped = -1
    with open(rna_flagstat) as file:
        for line in file:
            pass_nreads, fail_nreads, desc = line.split('\t')
            desc = desc.strip()
            if desc == 'primary mapped': prim_mapped = int(pass_nreads)
            if desc == 'primary duplicates': prim_duped = int(pass_nreads)
    assert prim_mapped >=0, F'The key <primary mapped> is absent in the file {rna_flagstat}'
    assert prim_duped >=0, F'The key <primary duplicates> is absent in the file {rna_flagstat}'
    return prim_mapped - prim_duped
    
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

def getR(neo_seq,iedb_seq): 
    align_score = []
    a = 26
    k = 4.86936
    for seq in iedb_seq:
        aln_score = aligner(neo_seq,seq)
        if aln_score:            
            localds_core = max([line[2] for line in aln_score])
            align_score.append(localds_core)

    bindingEnergies = list(map(lambda x: -k * (a - x), align_score))
    ## This tweak ensure that we get similar results compared with antigen.garnish at
    ## https://github.com/andrewrech/antigen.garnish/blob/main/R/antigen.garnish_predict.R#L86
    # bindingEnergies = [max(bindingEnergies)] 
    sumExpEnergies = sum([exp(x) for x in bindingEnergies])
    Zk = (1 + sumExpEnergies)
    return float(sumExpEnergies) / Zk

def getiedbseq(iedb_path):
    iedb_seq = []
    with open(iedb_path, 'r') as fin: #'/data8t_2/zzt/antigen.garnish/iedb.fasta'
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

def runblast(query_seq, target_fasta, output_file):
    os.system(F'mkdir -p {output_file}.tmp')
    query_fasta = F'{output_file}.tmp/foreignness_query.{query_seq}.fasta'
    with open(query_fasta, 'w') as query_fasta_file:
        query_fasta_file.write(F'>{query_seq}\n{query_seq}\n')
    # from https://github.com/andrewrech/antigen.garnish/blob/main/R/antigen.garnish_predict.R
    cmd = F'''blastp \
        -query {query_fasta} -db {target_fasta} \
        -evalue 100000000 -matrix BLOSUM62 -gapopen 11 -gapextend 1 \
        -out {query_fasta}.blastp_iedbout.csv -num_threads 8 \
        -outfmt '10 qseqid sseqid qseq qstart qend sseq sstart send length mismatch pident evalue bitscore' '''
    logging.debug(cmd)
    os.system(cmd)
    ret = []
    with open(F'{query_fasta}.blastp_iedbout.csv') as blastp_csv:
        for line in blastp_csv:
            tokens = line.strip().split(',')
            sseq = tokens[5]
            is_canonical = all([(aa in 'ARNDCQEGHILKMFPSTWYV') for aa in sseq])
            if is_canonical: ret.append(sseq)
    return ret

def read_tesla_xls(tesla_xls, patientID):
    # cat GRCh37_gencode_v19_CTAT_lib_Mar012021.plug-n-play/ctat_genome_lib_build_dir//ref_annot.gtf.mini.sortu  | awk '{print $NF}' | sort | uniq | wc
    #  57055   57055 1688510 # 57055 genes, so averageTPM = 1e6 / 57055
    df1 = pd.read_excel(tesla_xls)
    df1.PATIENT_ID = df1.PATIENT_ID.astype(int)
    df = df1.loc[df1.PATIENT_ID == patientID,]
    if 'NETMHC_BINDING_AFFINITY' in df.columns:
        df['ET_BindAff'] = df.NETMHC_BINDING_AFFINITY.astype(float)
    elif 'NETMHC_PAN_BINDING_AFFINITY' in df.columns:
        df['ET_BindAff'] = df.NETMHC_PAN_BINDING_AFFINITY.astype(float)
    else:
        sys.stderr.write(F'Peptide-MHC binding-affinity is not present in {tesla_xls}')
        exit(1)
    df['MT_BindAff'] = df['ET_BindAff']
    df['BindStab'] = df.BINDING_STABILITY.astype(float)
    df['Quantification'] = df.TUMOR_ABUNDANCE.astype(float)
    df['Agretopicity'] = df.AGRETOPICITY.astype(str).replace('N/A', np.nan).astype(float)
    df['Foreignness'] = df.FOREIGNNESS.astype(float)
    df['RNA_normAD'] = df.Quantification # * 0.02 # 0.02 is empirical # mannwhitneyu p-values and effect sizes are not affected by this normalizer
    
    # default values
    df['Identity'] = [F'SNV_D{i}_MT' for i in range(len(df))] # most neo-epitope candidates are from SNVs
    df['BindLevel'] = 'SB' # most neo-epitope candidates are assumed to have strong-binding for pMHC
    
    # dummy values
    df['DNA_refDP'] = 100
    df['DNA_altDP'] = 100
    df['RNA_refDP'] = 100
    df['RNA_altDP'] = 100
    
    ret = df.dropna(subset=['ET_BindAff', 'BindStab', 'Quantification', 'Agretopicity', 'Foreignness'])
    # ret = df.dropna(subset=['ET_BindAff', 'BindStab', 'Quantification', 'Foreignness'])
    if len(ret) * 2 < len(df): logging.warning(F'Only {len(ret)} rows out of {len(df)} rows are kept from the file {tesla_xls} with patientID={patientID}')
    return ret

def indicator(x): return np.where(x, 1, 0)
def compute_immunogenic_probs(data, paramset):
    
    t2BindAff      = paramset.binding_affinity_hard_thres
    t1BindAff      = paramset.binding_affinity_soft_thres
    
    t2BindStab     = paramset.binding_stability_hard_thres
    t1BindStab     = paramset.binding_stability_soft_thres
    
    t2Abundance    = paramset.tumor_abundance_hard_thres
    t1Abundance    = paramset.tumor_abundance_soft_thres
    
    t0Abundance    = paramset.tumor_abundance_recognition_thres
    t0Agretopicity = paramset.agretopicity_thres
    t0Foreignness  = paramset.foreignness_thres
    
    snvindel_location_param     = paramset.snvindel_location_param
    non_snvindel_location_param = paramset.non_snvindel_location_param
    # prior_weight                = paramset.immuno_strength_null_hypothesis_prior_weight
    
    are_snvs_or_indels_bool = (data.Identity.str.startswith(('SNV_', 'INS_', 'DEL_', 'INDEL_', 'FSV_')))
    are_snvs_or_indels = indicator(are_snvs_or_indels_bool)
    
    rescued_by_bindstab_ind = indicator(data.BindStab > paramset.resue_by_bindstab_thres)

    t0foreign_nfilters = indicator(np.logical_and((t0Foreignness >= data.Foreignness), (t0Agretopicity <= data.Agretopicity)))
    t0recognized_nfilters = (
        indicator(data.ET_BindAff >= t1BindAff) +
        indicator(data.BindStab <= t1BindStab) +
        indicator(data.Quantification <= t0Abundance) + 
        t0foreign_nfilters)
    t1presented_nfilters = (
        indicator(data.ET_BindAff >= t1BindAff) +
        indicator(data.BindStab <= t1BindStab) +
        indicator(data.Quantification <= t1Abundance) 
    )
    t2presented_nfilters = (
        indicator(data.ET_BindAff >= t2BindAff) +
        indicator(data.BindStab <= t2BindStab) +
        indicator(data.Quantification <= t2Abundance) 
        # + indicator(~data.BindLevel.isin(['SB', 'WB']))
    )
    t2dna_nfilters = (
        indicator(data.DNA_altDP < 5) + 
        indicator(data.DNA_altDP < (data.DNA_refDP + data.DNA_altDP + sys.float_info.epsilon) * 0.1)
    ) * are_snvs_or_indels
    t2rna_nfilters = (
        indicator(data.RNA_altDP < 5) + 
        indicator(data.RNA_altDP < (data.RNA_refDP + data.RNA_altDP + sys.float_info.epsilon) * 0.1)
    ) * are_snvs_or_indels
    
    t0_are_foreign = (t0foreign_nfilters == 0)
    t1_are_presented = (t1presented_nfilters == 0)
    presented_not_recog = t1_are_presented * are_snvs_or_indels * indicator(t0foreign_nfilters >  0)
    presented_and_recog = t1_are_presented * are_snvs_or_indels * indicator(t0foreign_nfilters == 0)
    #presented_not_recog_sumtpm = sum(data.RNA_normAD * presented_not_recog)
    #presented_and_recog_sumtpm = sum(data.RNA_normAD * presented_and_recog)
    #presented_not_recog_maxtpm = max(data.RNA_normAD * presented_not_recog)
    #presented_and_recog_maxtpm = max(data.RNA_normAD * presented_and_recog)
    presented_not_recog_vals = sorted(ad for (ad, ind) in zip(data.RNA_normAD, presented_not_recog) if ind)
    presented_and_recog_vals = sorted(ad for (ad, ind) in zip(data.RNA_normAD, presented_and_recog) if ind)
    presented_not_recog_medtpm = (statistics.median(presented_not_recog_vals) if len(presented_not_recog_vals) else math.nan)
    presented_and_recog_medtpm = (statistics.median(presented_and_recog_vals) if len(presented_and_recog_vals) else math.nan)
    
    #def get_lower_upper_bounds(arr):
    #    npos = len(arr)
    #    # https://en.wikipedia.org/wiki/Standard_normal_table
    #    mednpos = 1.28*math.sqrt(0.5*(1-0.5)*len(arr)) # 85% -> 1.44 | 90% -> 1.645
    #    medlower = npos * 0.5 - mednpos
    #    medupper = npos * 0.5 + mednpos
    #    retlower = (np.percentile(arr, medlower / len(arr) * 100) if (medlower > 0.5) else -float('inf'))
    #    retupper = (np.percentile(arr, medupper / len(arr) * 100) if (medupper < len(arr) - 0.5) else float('inf'))
    #    retmid = (np.percentile(arr, 0.5*100) if arr else 0)
    #    return (retlower, retmid, retupper)
    
    #presented_not_recog_medtpm_3 = get_lower_upper_bounds(presented_not_recog_vals)
    #presented_and_recog_medtpm_3 = get_lower_upper_bounds(presented_and_recog_vals)
    
    if presented_not_recog_vals and presented_and_recog_vals:
        mwutest = scipy.stats.mannwhitneyu(presented_not_recog_vals, presented_and_recog_vals)
    else:
        mwutest = None
    #prior_avg_burden = (presented_and_recog_sumtpm + presented_not_recog_sumtpm + 0.5) / (sum(presented_not_recog) + sum(presented_and_recog) + 1)
    #presented_and_recog_average_burden = (prior_avg_burden * prior_weight + presented_and_recog_sumtpm) / (prior_weight + sum(presented_and_recog))
    #presented_not_recog_average_burden = (prior_avg_burden * prior_weight + presented_not_recog_sumtpm) / (prior_weight + sum(presented_not_recog))
    #presented_and_recog_sub_avg_burden = (prior_avg_burden * prior_weight + presented_and_recog_sumtpm - presented_and_recog_maxtpm) / (
    #        prior_weight + max((0, sum(presented_and_recog) - 1)))
    #presented_not_recog_sub_avg_burden = (prior_avg_burden * prior_weight + presented_not_recog_sumtpm - presented_not_recog_maxtpm) / (
    #        prior_weight + max((0, sum(presented_not_recog) - 1)))

    # The variable immuno_strength should be positive/negative for patients with high/low immune strength
    # immuno_strength = log(presented_not_recog_average_burden / presented_and_recog_average_burden) * 2 
    med_immuno_strength = 0
    if mwutest and mwutest.pvalue < paramset.immuno_strength_p_value:
        if paramset.immuno_strength_effect_size * presented_not_recog_medtpm < presented_and_recog_medtpm:
            med_immuno_strength = -1
        if paramset.immuno_strength_effect_size * presented_and_recog_medtpm < presented_not_recog_medtpm:
            med_immuno_strength = 1
    #if presented_not_recog_medtpm_3[1] < presented_and_recog_medtpm_3[0] and presented_not_recog_medtpm_3[2] < presented_and_recog_medtpm_3[1]:
    #    med_immuno_strength = -1
    #if presented_and_recog_medtpm_3[1] < presented_not_recog_medtpm_3[0] and presented_and_recog_medtpm_3[2] > presented_not_recog_medtpm_3[1]:
    #    med_immuno_strength = 1
    
    # Please be aware that t2presented_nfilters should be zero if the data were hard-filtered with t2 thresholds first. 
    
    bindlevel_penal = indicator(~data.BindLevel.isin(['SB'])) * 3 * (1 - rescued_by_bindstab_ind * indicator(med_immuno_strength >= 0))
    rnaqlevel_penal = indicator(t2rna_nfilters > 0)           * 3 * (1 - rescued_by_bindstab_ind * indicator(med_immuno_strength >= 0))
    
    # if med_immuno_strength >=  1 then neoepitopes are always recognized # this never happened in the TESLA dataset though
    # if med_immuno_strength <= -1 then heavily penalize for non-recognition by foreignness # this happened for TESLA patient 2
    log_odds_ratio = (t1BindAff / (t1BindAff + data.MT_BindAff)
            + np.maximum(indicator(t0recognized_nfilters == 0), med_immuno_strength)
            - indicator(t0Foreignness >= data.Foreignness) * indicator(med_immuno_strength < 0) * 3
            - (t1presented_nfilters)
            - (t2presented_nfilters * 3)
            - indicator(t2dna_nfilters > 0) * 3
            - bindlevel_penal
            - rnaqlevel_penal
            + (are_snvs_or_indels * snvindel_location_param) + (1 - are_snvs_or_indels) * non_snvindel_location_param
            )
    p = 1 / (1 + 2.0**(-log_odds_ratio))
    return (p, t2presented_nfilters, t1presented_nfilters, t0recognized_nfilters, 
            sum(presented_not_recog), sum(presented_and_recog),
            #presented_not_recog_sumtpm, presented_and_recog_sumtpm,
            #presented_not_recog_maxtpm, presented_and_recog_maxtpm,
            presented_not_recog_medtpm, presented_and_recog_medtpm,
            #presented_not_recog_average_burden, presented_and_recog_average_burden,
            #presented_not_recog_sub_avg_burden, presented_and_recog_sub_avg_burden,
            presented_not_recog_vals,   presented_and_recog_vals,
            #presented_not_recog_medtpm_3, presented_and_recog_medtpm_3,
            mwutest, med_immuno_strength)
   
def datarank(data, outcsv, paramset, drop_cols = []):
    
    (probs, t2presented_filters, t1presented_filters, t0recognized_filters, 
            n_presented_not_recognized, n_presented_and_recognized, 
            #presented_not_recog_sumtpm, presented_and_recog_sumtpm,
            #presented_not_recog_maxtpm, presented_and_recog_maxtpm,
            presented_not_recog_medtpm, presented_and_recog_medtpm,
            #presented_not_recog_average_burden, presented_and_recog_average_burden,
            #presented_not_recog_sub_avg_burden, presented_and_recog_sub_avg_burden, 
            presented_not_recog_vals,   presented_and_recog_vals,
            #presented_not_recog_medtpm_3, presented_and_recog_medtpm_3,
            mwutest, med_immuno_strength) = compute_immunogenic_probs(data, paramset)
    
    #immuno_strength_lo = log(presented_not_recog_sub_avg_burden / presented_and_recog_average_burden) * 2
    #immuno_strength_hi = log(presented_not_recog_average_burden / presented_and_recog_sub_avg_burden) * 2
    
    data['Probability'] = probs
    data['PresentationPreFilters'] = t2presented_filters
    data['PresentationFilters'] = t1presented_filters
    data['RecognitionFilters'] = t0recognized_filters
    data["Rank"]=data["Probability"].rank(method="first", ascending=False)
    
    data=data.sort_values("Rank")
    data=data.astype({"Rank":int})
    data=data.drop(drop_cols, axis=1)
    col2last(data, 'SourceAlterationDetail')
    col2last(data, 'PepTrace')
    dropcols(data, ['BindLevel', 'BindAff'])
    data.to_csv(outcsv, header=1, sep='\t', index=0, float_format='%6g', na_rep = 'NA')
    with open(outcsv + ".extrainfo", "w") as extrafile:
        # Presented: expected to be bound pMHC
        # Recognized: expected to be immunogenic if bound pMHC
        extrafile.write(F'N_presented_and_recognized={n_presented_and_recognized}\n')
        extrafile.write(F'N_presented_not_recognized={n_presented_not_recognized}\n')
        extrafile.write(F'Presented_and_recog_vals={sorted(presented_and_recog_vals)}\n')
        extrafile.write(F'Presented_not_recog_vals={sorted(presented_not_recog_vals)}\n')

        #extrafile.write(F'Presented_not_recog_sumtpm={presented_not_recog_sumtpm}\n')
        #extrafile.write(F'Presented_and_recog_sumtpm={presented_and_recog_sumtpm}\n')
        #extrafile.write(F'Presented_not_recog_maxtpm={presented_not_recog_maxtpm}\n')
        #extrafile.write(F'Presented_and_recog_maxtpm={presented_and_recog_maxtpm}\n')
        extrafile.write(F'Presented_not_recog_medtpm={presented_not_recog_medtpm}\n')
        extrafile.write(F'Presented_and_recog_medtpm={presented_and_recog_medtpm}\n')
        #extrafile.write(F'Presented_not_recog_average_burden={presented_not_recog_average_burden}\n')
        #extrafile.write(F'Presented_and_recog_average_burden={presented_and_recog_average_burden}\n')
        #extrafile.write(F'Presented_not_recog_sub_avg_burden={presented_not_recog_sub_avg_burden}\n')
        #extrafile.write(F'Presented_and_recog_suv_avg_burden={presented_and_recog_sub_avg_burden}\n')
        #extrafile.write(F'Immuno_strength_lo={immuno_strength_lo}\n')
        #extrafile.write(F'Immuno_strength_hi={immuno_strength_hi}\n')
        #xtrafile.write(F'Immuno_strength={immuno_strength}\n')
        #extrafile.write(F'Presented_not_recog_medtpm_3={presented_not_recog_medtpm_3}\n')
        #extrafile.write(F'Presented_and_recog_medtpm_3={presented_and_recog_medtpm_3}\n')
        extrafile.write(F'StatisticalTest={mwutest}\n')
        extrafile.write(F'Med_immuno_strength={med_immuno_strength}\n')
        extrafile.write(F'Expected_immunogenic_peptide_num={sum(probs)}\n')
    return data, (n_presented_not_recognized, n_presented_and_recognized, med_immuno_strength)
    #def between(x, a, b): return min((max((x,a)),b))
    #immuno_strength_real = between(0, immuno_strength_hi, immuno_strength_lo)
    # return data, (n_presented_not_recognized, n_presented_and_recognized, presented_not_recog_sumtpm, presented_and_recog_sumtpm, immuno_strength)
    
def main():
    description = 'This script computes the probability that each neoantigen candidate is validated to be immunogenic (i.e., true positive). '
    epilog = '''
Hard thresholds should be much less strict than soft thresholds. 
If (output_directory, tesla_xls, tesla_patientID) are set, 
    then (input_folder, iedb_fasta, netmhc_cmd, alteration_type) are all irrelevant and therefore unused.
If the keyword rerank is in function,
    then (iedb_fasta, netmhc_cmd, alteration_type) are all irrelevant and therefore unused. '''.strip()
    
    parser = argparse.ArgumentParser(description = description, epilog = epilog, formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('-i', u2d('input_file'), help = 'Input file generated by bindstab_filter.py (binding-stability filter)', required = True)
    parser.add_argument('-I', u2d('iedb_fasta'), help = 'path to IEDB reference fasta file containing pathogen-derived immunogenic peptides', required = True)
    parser.add_argument('-o', u2d('output_file'), help = 'output file to store results of neoantigen prioritization', required = True)
    
    parser.add_argument('-D', u2d('dna_detail'), help = 'Optional input file providing SourceAlterationDetail for SNV and InDel variants from DNA-seq', default = '')
    parser.add_argument('-R', u2d('rna_detail'), help = 'Optional input file providing SourceAlterationDetail for SNV and InDel variants from RNA-seq', default = '')
    parser.add_argument('-F', u2d('fusion_detail'), help = 'Optional input file providing SourceAlterationDetail for fusion variants from RNA-seq', default = '')
    parser.add_argument('-S', u2d('splicing_detail'), help = 'Optional input file providing SourceAlterationDetail for splicing variants from RNA-seq', default = '')
    
    parser.add_argument('-t',  u2d('alteration_type'), default = 'snv,indel,fsv,fusion,splicing',
            help = 'type of alterations detected, can be a combination of (snv, indel, fsv, sv, and/or fusion separated by comma)')
    parser.add_argument(u2d('binding_affinity_hard_thres'), default = 231, type=float, # 34 is not used because newer version of netMHCpan generates lower values
            help = 'hard threshold of peptide-MHC binding affinity to predict peptide-MHC presentation to cell surface')
    parser.add_argument(u2d('binding_affinity_soft_thres'), default = 21, type=float,
            help = 'soft threshold of peptide-MHC binding affinity to predict peptide-MHC presentation to cell surface')
    parser.add_argument(u2d('binding_stability_hard_thres'), default = 0.13, type=float,
            help = 'hard threshold of peptide-MHC binding stability to predict peptide-MHC presentation to cell surface')
    parser.add_argument(u2d('binding_stability_soft_thres'), default = 1.4, type=float,
            help = 'soft threshold of peptide-MHC binding stability to predict peptide-MHC presentation to cell surface')
    parser.add_argument(u2d('tumor_abundance_hard_thres'), default = 1.0, type=float,
            help = 'hard threshold of peptide-MHC binding affinity to predict peptide-MHC recognition by T-cells')
    parser.add_argument(u2d('tumor_abundance_soft_thres'), default = 11, type=float,
            help = 'soft threshold of peptide-MHC binding affinity to predict peptide-MHC recognition by T-cells')
    parser.add_argument(u2d('agretopicity_thres'), default = 0.1, type=float,
            help = 'threshold of agretopicity to predict peptide-MHC recognition by T-cells')
    parser.add_argument(u2d('foreignness_thres'), default = 1e-16, type=float,
            help = 'threshold of foreignness to predict peptide-MHC recognition by T-cells')
    parser.add_argument(u2d('tumor_abundance_recognition_thres'), default = 33, type=float,
            help = 'threshold of tumor abundance to predict peptide-MHC recognition by T-cells')
    
    parser.add_argument(u2d('snvindel_location_param'), default = -1.5, type=float,
            help = 'location parameter of the logistic regression used to estimate the probability that a peptide-MHC is immunogenic '
            'if the peptide originate from SNVs and InDels. '
            'This parameter does not change the ranking of peptide-MHC immunogenities for peptides originating from SNVs and InDels. ')
    parser.add_argument(u2d('non_snvindel_location_param'), default = -1.5 - 3, type=float,
            help = 'location parameter of the logistic regression used to estimate the probability that a peptide-MHC is immunogenic '
            'if the peptide does not originate from SNVs and InDels. '
            'This parameter does not change the ranking of peptide-MHC immunogenities for peptides not originating from SNVs and InDels. ')
    
    parser.add_argument(u2d('immuno_strength_p_value'), default = 0.05, type=float,
            help = 'The p-value threshold for the mannwhitneyu test for recognized versus unrecognized neo-abundance. ')
    parser.add_argument(u2d('immuno_strength_effect_size'), default = 1.5, type=float,
            help = 'The median of recognized neo-abundance to the median of unrecognized one, '
            'above which/below the inverse of which the immuno-strength is high if p-value is also low. ')
    parser.add_argument(u2d('resue_by_bindstab_thres'), default = 1.4*3, type=float,
            help = 'A pMHC with WB (weak bind-level) or low RNA-seq variant signal is not penalized if the pMHC binding stability is above this threshold')
    
    
    parser.add_argument(u2d('dna_vcf'), default = '',
            help = 'VCF file (which can be block-gzipped) generated by calling small variants from DNA-seq data')
    parser.add_argument(u2d('rna_vcf'), default = '',
            help = 'VCF file (which can be block-gzipped) generated by calling small variants from RNA-seq data')
    #parser.add_argument('--rna-depth', default = '',
    #        help = 'A file containing summary information about RNA fragment depth')

    parser.add_argument(u2d('function'), default = '',
            help = 'The keyword rerank means using existing stats (affinity, stability, etc.) to re-rank the neoantigen candidates')
    parser.add_argument(u2d('tesla_xls'), default = '',
            help = 'Table S4 and S7 at https://doi.org/10.1016/j.cell.2020.09.015')
    parser.add_argument(u2d('tesla_patientID'), default = -1, type = int,
            help = 'the ID in the PATIENT_ID column to select the rows in --tesla-xls')
    
    args = parser.parse_args()
    paramset = args
    print(paramset)
    
    if not isna(args.tesla_xls):
        data = read_tesla_xls(args.tesla_xls, args.tesla_patientID)
        # read_tesla_xls filled the missing columns (such as RNA-seq stats) with the default values resulting in maximum probabilities
        # so we decrease the location param to decrease the probabilities accordingly.
        # args.snvindel_location_param -= 0.5
        data2, _ = datarank(data, args.output_file, paramset)
        exit(0)
    if args.function == 'rerank':
        keptdata = pd.read_csv(args.output_file + '.expansion', sep='\t')
        keptdata1 = keptdata[keptdata['ET_pep'] == keptdata['MT_pep']]
        data1, _ = datarank(keptdata1, args.output_file + '.reranked', paramset, drop_cols = ['ET_pep', 'ET_BindAff', 'BIT_DIST'])
        data, _ = datarank(keptdata, args.output_file + 'reranked.expansion', paramset)
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

    candidate_file = open(args.input_file) # open(F'{args.input_directory}/{prefix}_candidate_pmhc.tsv')
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
    data_exist = [] # save existing hla, mutant_type peptide
    agre_exist = []
    identity_idx = fields.index('Identity')
    for line1 in reader:
        line = copy.deepcopy(line1)
        blast_iedb_seqs = runblast(line[1], args.iedb_fasta, args.output_file)
        R = getR(line[1], blast_iedb_seqs)
        line.append(R)
        dna_varqual = 0
        dna_ref_depth = 0
        dna_alt_depth = 0
        rna_varqual = 0
        rna_ref_depth = 0
        rna_alt_depth = 0
        line_info_string = ""
        is_frameshift = False
        identity = line[identity_idx]
        if (identity.strip().split('_')[0] in ["SNV", 'INS', 'DEL', 'INDEL', 'FSV'] or identity.strip().split('_')[0].startswith("INDEL")):
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
                    #line_info_string += 'DNAseqVariantQuality${}'.format(max_varqual)
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
        elif (identity.strip().split('_')[0]=="FUS") and fusion:
            line_num = int(identity.strip().split('_')[1])
            fusion_line = fusion[line_num-1]
            ele = fusion_line.strip().split('\t')
            annotation_info = ["FusionName","JunctionReadCount","SpanningFragCount","est_J","est_S","SpliceType","LeftGene","LeftBreakpoint",
                                "RightGene","RightBreakpoint","LargeAnchorSupport","FFPM","LeftBreakDinuc","LeftBreakEntropy","RightBreakDinuc",
                                "RightBreakEntropy","annots","CDS_LEFT_ID","CDS_LEFT_RANGE","CDS_RIGHT_ID","CDS_RIGHT_RANGE","PROT_FUSION_TYPE",
                                "FUSION_MODEL","FUSION_CDS","FUSION_TRANSL","PFAM_LEFT","PFAM_RIGHT"]
            for i in range(0, len(ele),1):
                line_info_string+=annotation_info[i]+"$"+ele[i]+"#"
        elif (identity.strip().split('_')[0]=="SP") and splicing:
            line_num = int(identity.strip().split('_')[1])
            splicing_line = splicing[line_num-1]
            ele = splicing_line.strip().split('\t')
            annotation_info = ["chrom","txStart","txEnd","isoform","protein","strand","cdsStart","cdsEnd","gene","exonNum",
                                "exonLens","exonStarts","ensembl_transcript"]
            for i in range(0, len(ele),1):
                line_info_string+=annotation_info[i]+"$"+ele[i]+"#"
        else:
            continue
        #line[5] = identity.strip().split('_')[0]
        line.append(dna_varqual)
        line.append(dna_ref_depth)
        line.append(dna_alt_depth)
        line.append(rna_varqual)
        line.append(rna_ref_depth)
        line.append(rna_alt_depth)
        line.append(line_info_string if line_info_string else 'N/A')
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
    data.ET_BindAff = data.ET_BindAff.astype(float)
    data.MT_BindAff = data.MT_BindAff.astype(float)
    data.WT_BindAff = data.WT_BindAff.astype(str).replace('N/A', np.nan).astype(float)
    data.BindStab = data.BindStab.astype(float)
    data.Foreignness = data.Foreignness.astype(float)
    data.Agretopicity = data.Agretopicity.astype(str).replace('N/A', np.nan).astype(float)
    data.Quantification = data.Quantification.astype(float)
    data['RNA_normAD'] = data.Quantification # data.RNA_altDP.astype(float) / get_avg_depth_from_rna_depth_filename(args.rna_depth)
    
    # are_highly_abundant is not used because we have too little positive data
    # are_highly_abundant = ((data.MT_BindAff <= 34/10.0) & (data.BindStab >= 1.4*10.0) & (data.Quantification >= 1.0*10))
    # keptdata = data[(data.Quantification >= tumor_RNA_TPM_threshold) & ((~data.is_frameshift) | are_highly_abundant) & (data.Agretopicity > -1)]
    keptdata = data    
    #keptdata.to_csv(F'{args.output_file}.debug')
    keptdata1 = keptdata[keptdata['ET_pep'] == keptdata['MT_pep']]
    data1, _ = datarank(keptdata1, args.output_file, paramset, drop_cols = ['ET_pep', 'ET_BindAff', 'BIT_DIST'])
    data, _ = datarank(keptdata, args.output_file + '.expansion', paramset)
    
    if dnaseq_small_variants_file: dnaseq_small_variants_file.close()
    if rnaseq_small_variants_file: rnaseq_small_variants_file.close()
    
if __name__ == '__main__':
    main()

