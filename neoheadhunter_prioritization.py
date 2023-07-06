import copy,csv,getopt,logging,multiprocessing,os,sys,subprocess # ,os
#import csv
#import logging
#import multiprocessing
import pandas as pd
import numpy as np
from Bio import pairwise2
from Bio.SubsMat import MatrixInfo as matlist
from Bio.SeqIO.FastaIO import SimpleFastaParser
from math import log, exp

import pysam

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

# Result using this measure is dependent on RNA-seq read length, so it is not recommended to use it. 
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
            #if neo_seq == 'ILDTAGHEEY' and (seq in 'MLDHAGNMSACAGAL' or localds_core >= 20):
            #    print('NeoSeq={} WTseq={} aln_score={}'.format(neo_seq, seq, aln_score))
            align_score.append(localds_core)

    bindingEnergies = list(map(lambda x: -k * (a - x), align_score))
    ## This tweak ensure that we get similar results compared with antigen.garnish at
    ## https://github.com/andrewrech/antigen.garnish/blob/main/R/antigen.garnish_predict.R#L86
    # bindingEnergies = [max(bindingEnergies)] 
    sumExpEnergies = sum([exp(x) for x in bindingEnergies])
    Zk = (1 + sumExpEnergies)
    if neo_seq == 'YLYHRVDVI': print(F'__________>>>>>>>>>  YLYHRVDVI.ENERGY = {sumExpEnergies}')
    return float(sumExpEnergies) / Zk
    
    #lZk = logSum(bindingEnergies + [0])
    #lGb = logSum(bindingEnergies)
    #R=exp(lGb-lZk)
    #return R

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

def logSum(v):
    ma = max(v)
    return log(sum(map(lambda x: exp(x-ma),v))) + ma

def aligner(seq1,seq2):
    matrix = matlist.blosum62
    gap_open = -11
    gap_extend = -1
    aln = pairwise2.align.localds(seq1.upper(), seq2.strip().split('+')[0].upper(), matrix, gap_open, gap_extend)
    return aln
    
def write_file(a_list, name):
    textfile = open(name, "w")
    for element in a_list:
        textfile.write(element + "\n")
    textfile.close()
    return

##########Calculate wild type binding affinity###########
# def mutation_netmhc(prefix, netmhc_path, input_folder, output_folder, hla):
#     run_netmhc = netmhc_path+" -a "+hla+" -f "+input_folder+"/"+prefix+"_snv_indel_wt.fasta -l '8,9,10,11' -BA > "+output_folder+"/tmp_netmhc/"+hla+"_wt_tmp_hla_netmhc.txt"
#     print(run_netmhc)
#     subprocess.call(run_netmhc, shell=True, executable="/bin/bash")

# def mutation_netmhc_parallel(prefix, netmhc_path, input_folder, output_folder, hla_str):
#     os.system("mkdir "+output_folder+"/tmp_netmhc")
#     hla_list = list(hla_str.strip().split(","))
#     netmhc_hla_process=[]
#     for hla in hla_list:
#         run_netmhc = multiprocessing.Process(target=mutation_netmhc,args=(prefix, netmhc_path, input_folder, output_folder, hla))#"./netmhc_parallel.sh "+netmhc_path+" "+output_folder+" "+hla
#         netmhc_hla_process.append(run_netmhc)
#     for p in netmhc_hla_process:
#         p.daemon = True
#         p.start()
#     for p in netmhc_hla_process:
#         p.join()
#     os.system("cat "+output_folder+"/tmp_netmhc/* > "+output_folder+"/"+prefix+"_snv_indel_wt_netmhc.csv")

def get_wt_bindaff(wt_seq,hla,output_folder,netmhc_path):
    # write_file(wt_seq, "stage.pep")
    os.system("mkdir "+output_folder+"/tmp")
    with open(output_folder+"/tmp/wt.pep", "w") as pepfile:
        pepfile.write("%s"% wt_seq)
    pepfile.close()
    args = netmhc_path+" -p "+output_folder+"/tmp/wt.pep -a "+ hla+" -l "+str(len(wt_seq))+" -BA >> "+output_folder+"/tmp/wt.csv"
    subprocess.call(args, shell=True)  
    wt_bindaff = 1
    with open(output_folder+"/tmp/wt.csv") as f:
        data = f.read()
    nw_data = data.split('-----------------------------------------------------------------------------------\n')
    for i in range(len(nw_data)):
        if i%4 == 2:
            wt_bindaff = nw_data[i].strip().split()[15]
            break
    os.system("rm -rf "+output_folder+"/tmp")
    return wt_bindaff

def runblast(query_seq, target_fasta, output_folder):
    os.system(F"mkdir -p {output_folder}/tmp")
    query_fasta = F"{output_folder}/tmp/foreignness_query.{query_seq}.fasta"
    with open(query_fasta, 'w') as query_fasta_file:
        query_fasta_file.write(F'>{query_seq}\n{query_seq}\n')
        #for q in sorted(list(set(query_seqs))):
        #    query_fasta_file.write('>{q}\n{q}\n')
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

def max2(a, b): return np.maximum(a, b)
def max3(a, b, c): return max2(a, max2(b, c))
def max4(a, b, c, d): return max2(a, max3(b, c, d))

def min2(a, b): return np.minimum(a, b)
def min3(a, b, c): return min2(a, min2(b, c))
def min4(a, b, c, d): return min2(a, min3(b, c, d))

def or2prob(a): return a/(1+a)

def stepwise(a): return np.where(a >= 1.0, 1e20, 1e-20)

def indicator(x): return np.where(x, 1, 0)

def compute_immunogenic_probs(data, 
        t0Abundance = 33,    t0Agretopicity = 0.1,   t0Foreignness = 1e-16,
        t1Abundance = 11,    t1BindAff      = 34,    t1BindStab    = 1.4,
        t2Abundance = 11/11, t2BindAff      = 34*11, t2BindStab    = round(1.4/11,3),
        snvindel_location_param = 1.5, non_snvindel_location_param = 0.5, prior_strength = 1):
    are_snvs_or_indels_bool = (data.Identity.isin(['SNV', 'INS', 'DEL', 'INDEL']))
    are_snvs_or_indels = indicator(are_snvs_or_indels_bool)
    
    t0foreign_nfilters = indicator(np.logical_and((t0Foreignness > data.Foreignness), (t0Agretopicity < data.Agretopicity)))
    t0recognized_nfilters = (
        indicator(t1BindAff > data.BindAff) +
        indicator(t1BindStab < data.BindStab) +
        indicator(t0Abundance < data.Quantification) + 
        t0foreign_nfilters)
    t1presented_nfilters = (
        indicator(t1BindAff > data.BindAff) +
        indicator(t1BindStab < data.BindStab) +
        indicator(t1Abundance < data.Quantification))
    t2presented_nfilters = (
        indicator(t2BindAff > data.BindAff) +
        indicator(t2BindStab < data.BindStab) +
        indicator(t2Abundance < data.Quantification))
    
    t0_are_foreign = (t0foreign_nfilters == 0)
    t1_are_presented = (t1presented_nfilters == 0)
    
    presented_not_recog = t1_are_presented * are_snvs_or_indels * indicator(t0foreign_nfilters <  0)
    presented_and_recog = t1_are_presented * are_snvs_or_indels * indicator(t0foreign_nfilters == 0)
    presented_not_recog_burden = sum(data.RNA_normAD * presented_not_recog)
    presented_and_recog_burden = sum(data.RNA_normAD * presented_and_recog)
    prior_avg_burden = (presented_and_recog_burden + presented_not_recog_burden + 0.5) / (sum(presented_not_recog) + sum(presented_and_recog) + 1)
    presented_and_recog_avg_burden = (prior_avg_burden * prior_strength + presented_and_recog_burden) / (prior_strength + sum(presented_and_recog))
    presented_not_recog_avg_burden = (prior_avg_burden * prior_strength + presented_not_recog_burden) / (prior_strength + sum(presented_not_recog))
    # The variable immuno_strength should be positive/negative for patients with low/high immune strength
    immuno_strength = log(presented_not_recog_avg_burden / presented_and_recog_avg_burden) * 2 
    
    # Please be aware that t2presented_nfilters should be zero if the data were hard-filtered with t2 thresholds first. 
    log_odds_ratio = (t1BindAff / (t1BindAff + data.BindAff)
            + np.minimum(1 - t0recognized_nfilters + immuno_strength, 1)
            - t1presented_nfilters 
            - (t2presented_nfilters * 3) 
            - (are_snvs_or_indels * snvindel_location_param) + (1 - are_snvs_or_indels) * non_snvindel_location_param)
    p = 1 / (1 + np.exp(-log_odds_ratio))
    return (p, t2presented_nfilters, t1presented_nfilters, t0recognized_nfilters, 
        sum(presented_not_recog), sum(presented_and_recog), 
        presented_not_recog_burden, presented_and_recog_burden, 
        immuno_strength)

def compute_prob(data, thresRNAseqDPratio, thresBindAff, thresAgretopicity, thresForeignness, thresBindStab): # thresRNAseqVariantQuality
    # Reference: https://www.sciencedirect.com/science/article/pii/S0092867420311569 Figure 4H
    
    # cat /mnt/d/TESLA/results/TESLA_91_90_97/alteration_detection/TESLA_91_90_97_transcript_quantification/abundance.tsv | wc
    #  180254  901270 7613266
    # python -c 'print(33/(1e6 / 180000)/(2*3))' # == 0.99 is_approx 1.0 assuming diploid allele with 33% tumor purity
    # thresRNAseqDPratio = 2.0
    are_snvs_or_indels_bool = (data.Identity.isin(['SNV', 'INS', 'DEL', 'INDEL']))
    are_snvs_or_indels = indicator(are_snvs_or_indels_bool)
    
    are_tumor_not_present = indicator((data.Quantification < 11 - 1e-9))
    are_tumor_not_abundant = indicator((data.Quantification < 33 - 1e-9))
    
    are_not_recognized = (
        indicator(thresBindAff < data.BindAff) + 
        indicator(data.BindStab < thresBindStab) + 
        are_tumor_not_abundant +
        indicator(np.logical_and((data.Foreignness < thresForeignness), (thresAgretopicity < data.Agretopicity)))
    )
    are_not_presented = (
        indicator(thresBindAff < data.BindAff) +
        indicator(data.BindStab < thresBindStab) +
        are_tumor_not_present
    )
    are_recognized_only = indicator(np.logical_or((data.Foreignness >= thresForeignness), (thresAgretopicity >= data.Agretopicity)))
    
    are_presented = (1 - indicator(are_not_presented)) * are_snvs_or_indels
    are_recognized = (1 - indicator(are_not_recognized)) * are_snvs_or_indels
    
    presented_not_recog = (1 - indicator(are_not_presented)) * are_snvs_or_indels * (1 - are_recognized_only)
    presented_and_recog = (1 - indicator(are_not_presented)) * are_snvs_or_indels * (0 + are_recognized_only)
    presented_not_recog_burden = sum(data.RNA_normAD * presented_not_recog)
    presented_and_recog_burden = sum(data.RNA_normAD * presented_and_recog)
    
    prior_avg_burden = (presented_and_recog_burden + presented_not_recog_burden + 0.5) / (sum(presented_not_recog) + sum(presented_and_recog) + 1)
    presented_and_recog_avg_burden = (prior_avg_burden * 1 + presented_and_recog_burden) / (1 + sum(presented_and_recog))
    presented_not_recog_avg_burden = (prior_avg_burden * 1 + presented_not_recog_burden) / (1 + sum(presented_not_recog))
    offset = log(presented_and_recog_avg_burden / presented_not_recog_avg_burden) * 2
    log_odds_ratios = (thresBindAff / (thresBindAff + data.BindAff) - are_not_presented + np.minimum(-offset + are_recognized, 1)) - 1.5 
    p = 1 / (1 + np.exp(-log_odds_ratios + 1 - are_snvs_or_indels))
    return (p, 3 - are_not_presented, 4 - are_not_recognized, sum(presented_not_recog), sum(presented_and_recog), presented_not_recog_burden, presented_and_recog_burden, offset)

def datarank(data, outcsv):
    
    probs, t2presented_filters, t1presented_filters, t0recognized_filters, \
            n_presented, n_recognized, presented_and_not_recog_burden, presented_and_recognized_burden, immuno_strength \
            = compute_immunogenic_probs(data)
    # compute_prob(data, 0.15, 34, 0.1, 1e-16, 1.4) # rna_seq_variant_quality
    data['Probability'] = probs
    data['PresentationPreFilters'] = t2presented_filters
    data['PresentationFilters'] = t1presented_filters
    data['RecognitionFilters'] = t0recognized_filters
    data["Rank"]=data["Probability"].rank(method="first", ascending=False)

    data=data.sort_values("Rank")
    data=data.astype({"Rank":int})
    data.to_csv(outcsv, header=1, sep='\t', index=0, float_format='%6g')
    with open(outcsv + ".extrainfo", "w") as extrafile:
        extrafile.write(F'n_presented={n_presented}\n')
        extrafile.write(F'n_recognized={n_recognized}\n')
        extrafile.write(F'presented_not_recognized_burden={presented_and_not_recog_burden}\n')
        extrafile.write(F'presented_and_recognized_burden={presented_and_recognized_burden}\n')
        extrafile.write(F'immuno_strength={immuno_strength}\n')
    return data, (n_presented, n_recognized, presented_and_not_recog_burden, presented_and_recognized_burden, immuno_strength)
    
def main():
    opts,args=getopt.getopt(sys.argv[1:],"hi:I:o:n:f:a:t:p:T:",
        ["input_folder=","iedb_fasta=","output_folder=","netmhc_path=","foreignness_score=","agretopicity=","alteration_type=", "prefix=", 
         "dna_vcf=", "rna_vcf=", "rna_depth=", "function=", "tumor_RNA_tmp_threshold="])
    input_folder=""
    iedb_fasta=""
    output_folder=""
    netmhc_path=""
    bindaff = 33.0
    foreignness_score= 1e-16
    agretopicity=0.1
    rna_seq_variant_quality = 30.0
    alteration_type="snv,indel,fusion,splicing"
    prefix=""
    dna_vcf = ''
    rna_vcf = ''
    rna_depth = ''
    function = ''
    tumor_RNA_TPM_threshold = 1.0
    USAGE='''
        This script computes the probability that each neoantigen candidate is validated to be a true neoantigen. 
        usage: python bindaff_related_prioritization.py -i <input_folder> -o <output_folder> \
            -f <foreignness_score> -a <agretopicity> -t <alteration_type> -p <prefix>
            required argument:
                -i | --input_folder : input folder including result file from bindstab output
                -I | --iedb_fasta : path to iedb fasta reference file
                -o | --output_folder : output folder to store result
                -n | --netmhc_path : path to run netmhcpan
                -b | --binding_affinity : binding affinity threshold for neoantigen candidates
                -f | --foreignness_score : foreignness threshold for neoantigen candidates
                -a | --agretopicity : agretopicity threshold for neoantigen candidates
                -q | --rna_seq_variant_quality : RNA-seq variant quality of the neoantigen candidate
                -t | --alteration_type : neoantigen from alteration type to rank (default is "snv,indel,fusion,splicing")
                -p | --prefix : prefix of output file
                -T | --tumor_RNA_tmp_threshold : tumor RNA TPM threshold below which the neoepitope candidate is filtered out.
                --dna_vcf : VCF file (which can be block-gzipped) generated by calling small variants from DNA-seq data (optional)
                --rna_vcf : VCF file (which can be block-gzipped) generated by calling small variants from RNA-seq data (optional)
                --rna_depth : TSV flagstat file obtained by running (samtools flagstat -O tsv ...) on the RNA-seq BAM file (optional)
                --function : The keyword rerank means using existing stats (affinity, stability, etc.) to re-rank the neoantigen candidates.
    '''
    for opt,value in opts:
        if opt =="h":
            print (USAGE)
            sys.exit(2)
        elif opt in ("-i","--input_folder"):
            input_folder=value
        elif opt in ("-I","--iedb_fasta"):
            iedb_fasta=value
        elif opt in ("-o","--output_folder"):
            output_folder =value 
        elif opt in ("-n","--netmhc_path"):
            netmhc_path =value 
        elif opt in ("-b","--binding_affinity"):
            bindaff =float(value)
        elif opt in ("-f","--foreignness_score"):
            foreignness_score =float(value) 
        elif opt in ("-a","--agretopicity"):
            agretopicity =float(value)
        elif opt in ("-q","--rna_seq_variant_quality"):
            rna_seq_variant_quality =float(value)
        elif opt in ("-t","--alteration_type"):
            alteration_type =value 
        elif opt in ("-p","--prefix"):
            prefix =value
        elif opt in ("-T", "--tumor_RNA_TPM_threshold"):
            tumor_RNA_TPM_threshold = float(value)
        elif opt in ("--dna_vcf"):
            dna_vcf = value
        elif opt in ("--rna_vcf"):
            rna_vcf = value
        elif opt in ("--rna_depth"):
            rna_depth = value
        elif opt in ("--function"):
            function = value
    if function == 'rerank':
        data = pd.read_csv(output_folder+"/"+prefix+"_neoantigen_rank_neoheadhunter.tsv",sep='\t')
        data2, _ = datarank(data, output_folder+"/"+prefix+"_neoantigen_rank_neoheadhunter.rerank.tsv")
        exit(0)
        
    if (input_folder =="" or iedb_fasta=="" or output_folder=="" or netmhc_path==""):
        print (USAGE)
        sys.exit(2)

    #wt_bindaff_list = []
    #wt_list = []
    wt_pep_to_bindaff = {}
    tmp_wt_bindaff_file =csv.reader(open(input_folder+"/tmp_identity/"+prefix+"_bindaff_filtered.tsv"), delimiter="\t")
    for line in tmp_wt_bindaff_file:
        if line[7] != "":
            #wt_bindaff_list.append(line[7])
            #wt_list.append(line[2])
            wt_pep_to_bindaff[line[2]] = line[7]
    #wt_bindaff_list.pop(0)
    #wt_list.pop(0)

    snv_indel_file = open(output_folder+"/../info/"+prefix+"_snv_indel.annotation.tsv")
    if os.path.exists(output_folder+"/../info/"+prefix+"_fusion.tsv"):
        fusion_file = open(output_folder+"/../info/"+prefix+"_fusion.tsv")
    else:
        fusion_file = []
    if os.path.exists(output_folder+"/../info/"+prefix+"_splicing.csv"):
        splicing_file = open(output_folder+"/../info/"+prefix+"_splicing.csv")
    else:
        splicing_file = []
   
    if dna_vcf:
        dnaseq_small_variants_file = pysam.VariantFile(dna_vcf, 'r')
    else:
        dnaseq_small_variants_file = []
    #rnaseq_small_variants_filename = output_folder+"/../info/"+prefix+"_rnaseq_small_variants.vcf.gz"
    if rna_vcf:
        rnaseq_small_variants_file = pysam.VariantFile(rna_vcf, 'r')
    else:
        rnaseq_small_variants_file = []

    snv_indel = []
    fusion = []
    splicing = []

    for line in snv_indel_file:
        snv_indel.append(str(line))
    for line in fusion_file:
        fusion.append(str(line))
    for line in splicing_file:
        splicing.append(str(line))

    if snv_indel_file: snv_indel_file.close()
    if fusion_file: fusion_file.close()
    if splicing_file: splicing_file.close() 
    # if rnaseq_small_variants_file: rnaseq_small_variants_file.close()

    #iedb_seq = getiedbseq(iedb_fasta)
    #iedb_dict = iedb_fasta_to_dict(iedb_fasta)
    reader = csv.reader(open(input_folder+"/"+prefix+"_candidate_pmhc.csv"), delimiter=",")
    fields=next(reader)
    fields.append("Foreignness")
    fields.append("Agretopicity")
    fields.append("DNA_QUAL")
    fields.append("DNA_refDP")
    fields.append("DNA_altDP")
    fields.append("RNA_QUAL")
    fields.append("RNA_refDP")
    fields.append("RNA_altDP")
    fields.append("SourceAlterationDetail")
    fields.append('is_frameshift')
    data_raw = []
    data_exist = [] # save existing hla, mutant_type peptide
    agre_exist = []
    for line1 in reader:
        line = copy.deepcopy(line1)
        blast_iedb_seqs = runblast(line[1], iedb_fasta, output_folder)
        R = getR(line[1], blast_iedb_seqs)
        line.append(R)
        mt_bindaff = float(line[3])
        identity = line[5]
        if (("SNV" in identity) or ('INS' in identity) or ('DEL' in identity) or ("INDEL" in identity)) and line[2] in wt_pep_to_bindaff:
            wt_bindaff = wt_pep_to_bindaff[line[2]]
        else:
            wt_bindaff = get_wt_bindaff(line[2],line[0].replace('*',''),output_folder,netmhc_path)
        
        A = mt_bindaff/float(wt_bindaff) 
        if ([line[0],line[1]] in data_exist):
            indices = [i for i, x in enumerate(data_exist) if x == [line[0],line[1]] ]
            for index in indices:
                if (A > agre_exist[index]): # should get the biggest agre (agretopicity)
                    agre_exist[index] = -2 #A
                    data_raw[index][8] = -2 #A
                    logging.info(F'Invalidated previous {line[0]} {line[1]}')
                else:
                    A = -2
        else:
            data_exist.append([line[0],line[1]])
            agre_exist.append(A)
        line.append(A)
        
        dna_varqual = 0
        dna_ref_depth = 0
        dna_alt_depth = 0
        rna_varqual = 0
        rna_ref_depth = 0
        rna_alt_depth = 0
        line_info_string = ""
        is_frameshift = False
        if (identity.strip().split('_')[0] in ["SNV", 'INS', 'DEL', 'INDEL'] or identity.strip().split('_')[0].startswith("INDEL")):
            line_num = int(identity.strip().split('_')[1])
            snv_indel_line = snv_indel[line_num-1]
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
                        #print('var-equal-test {} == {}'.format(vepvar, ele[0]))
                        if vep_lenient_equal(vepvar, ele[0]):
                            dna_varqual = max((dna_varqual, varqual))
                            dna_ref_depth = max((dna_ref_depth, varRD))
                            dna_alt_depth = max((dna_alt_depth, varAD))
                    #line_info_string += 'DNAseqVariantQuality${}'.format(max_varqual)
                if rnaseq_small_variants_file:
                    for vcfrecord in rnaseq_small_variants_file.fetch(chrom, int(pos) - 6, int(pos) + 6):
                        vepvar, varqual, varRD, varAD = var_vcf2vep(vcfrecord)
                        #print('var-equal-test {} == {}'.format(vepvar, ele[0]))
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
        elif (identity.strip().split('_')[0]=="FUSION"):
            line_num = int(identity.strip().split('_')[1])
            fusion_line = fusion[line_num-1]
            ele = fusion_line.strip().split('\t')
            annotation_info = ["FusionName","JunctionReadCount","SpanningFragCount","est_J","est_S","SpliceType","LeftGene","LeftBreakpoint",
                                "RightGene","RightBreakpoint","LargeAnchorSupport","FFPM","LeftBreakDinuc","LeftBreakEntropy","RightBreakDinuc",
                                "RightBreakEntropy","annots","CDS_LEFT_ID","CDS_LEFT_RANGE","CDS_RIGHT_ID","CDS_RIGHT_RANGE","PROT_FUSION_TYPE",
                                "FUSION_MODEL","FUSION_CDS","FUSION_TRANSL","PFAM_LEFT","PFAM_RIGHT"]
            for i in range(0, len(ele),1):
                line_info_string+=annotation_info[i]+"$"+ele[i]+"#"
        elif (identity.strip().split('_')[0]=="SP"):
            line_num = int(identity.strip().split('_')[1])
            splicing_line = splicing[line_num-1]
            ele = splicing_line.strip().split('\t')
            annotation_info = ["chrom","txStart","txEnd","isoform","protein","strand","cdsStart","cdsEnd","gene","exonNum",
                                "exonLens","exonStarts","ensembl_transcript"]
            for i in range(0, len(ele),1):
                line_info_string+=annotation_info[i]+"$"+ele[i]+"#"
        else:
            continue
        line[5] = identity.strip().split('_')[0]
        line.append(dna_varqual) 
        line.append(dna_ref_depth)
        line.append(dna_alt_depth)
        line.append(rna_varqual)
        line.append(rna_ref_depth)
        line.append(rna_alt_depth)
        line.append(line_info_string)
        line.append(is_frameshift)
        data_raw.append(line)
        
    picked_rows = []
    alt_type = alteration_type.replace(" ", "").strip().split(",")
    for line in data_raw:
        type = line[5].strip().split("_")[0]
        if (type=="SP"):
            type="SPLICING"
        if type.lower() in alt_type:
            picked_rows.append(line)
    # data can be emtpy (https://stackoverflow.com/questions/44513738/pandas-create-empty-dataframe-with-only-column-names)
    data=pd.DataFrame(picked_rows, columns = fields)
    data.BindAff = data.BindAff.astype(float)
    data.BindStab = data.BindStab.astype(float)
    data.Foreignness = data.Foreignness.astype(float)
    data.Agretopicity = data.Agretopicity.astype(float)
    data.Quantification = data.Quantification.astype(float)
    data['RNA_normAD'] = data.RNA_altDP.astype(float) / get_avg_depth_from_rna_depth_filename(rna_depth) 
    
    are_highly_abundant = ((data.BindAff <= 34/10.0) & (data.BindStab >= 1.4*10.0) & (data.Quantification >= 1.0*10))
    keptdata = data[(data.Quantification >= tumor_RNA_TPM_threshold) & ((~data.is_frameshift) | are_highly_abundant) & (data.Agretopicity > -1)]
    
    keptdata.insert(len(keptdata.columns)-1, 'SourceAlterationDetail', keptdata.pop('SourceAlterationDetail'))
    keptdata.drop(['BindLevel'], axis=1)
    data2, _ = datarank(keptdata, output_folder+"/"+prefix+"_neoantigen_rank_neoheadhunter.tsv")
    # keptdata.to_csv(output_folder+"/"+prefix+"_neoantigen_rank_neoheadhunter.tsv",header=1,sep='\t',index=0, float_format='%6g')
    
    if dnaseq_small_variants_file: dnaseq_small_variants_file.close()
    if rnaseq_small_variants_file: rnaseq_small_variants_file.close()
    
if __name__ == '__main__':
    main()

