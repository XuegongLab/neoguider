#from tracemalloc import start
import pandas as pd
import sys,getopt,os
import re
from pyfaidx import Fasta
d = {'Cys': 'C', 'Asp': 'D', 'Ser': 'S', 'Gln': 'Q', 'Lys': 'K',
     'Ile': 'I', 'Pro': 'P', 'Thr': 'T', 'Phe': 'F', 'Asn': 'N', 
     'Gly': 'G', 'His': 'H', 'Leu': 'L', 'Arg': 'R', 'Trp': 'W', 
     'Ala': 'A', 'Val':'V', 'Glu': 'E', 'Tyr': 'Y', 'Met': 'M'}

def seq2str(s): return str(s)

def shorten(x):
    if len(x) % 3 != 0: 
        # print(x)
        raise ValueError('Input length should be a multiple of three')

    y = ''
    for i in range(len(x) // 3):
        try:
            y += d[x[3 * i : 3 * i + 3]]
        except:
            continue
    return y  


def translate(seq):
       
    table = {
        'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M',
        'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T',
        'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K',
        'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',                 
        'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L',
        'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P',
        'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q',
        'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R',
        'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V',
        'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A',
        'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E',
        'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G',
        'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S',
        'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L',
        'TAC':'Y', 'TAT':'Y', 'TAA':'_', 'TAG':'_',
        'TGC':'C', 'TGT':'C', 'TGA':'_', 'TGG':'W',
    }
    protein = ""
    # print(seq)
    for i in range(0, len(seq), 3):
        codon = seq[i:i + 3]
        if len(codon)<3:
            break
        if table[codon.upper()] == '_':
            break
        protein+= table[codon.upper()]
        if (i+4) >= len(seq):
            break
    return protein


#####prepare fasta format input file for netMHC######
opts,args=getopt.getopt(sys.argv[1:],"hi:o:p:r:s:e:t:P:",["input_snv_file=","out_dir=","human_peptide=","reference=","software=","expression_file=","tpm_threshold=","prefix="])
input_snv_file =""
out_dir=""
human_peptide=""
reference=""
software="VEP"
expression_file=""
tpm_threshold=1
prefix=""
USAGE='''
    This script convert annotation result to fasta format file for netMHC
    usage: python annoation2fasta.py -i <input_snv_file> -o <out_dir> -p <human_peptide> 
                -r <reference> -s <software> -e <expression_file> -t <tpm_threshold> -P <prefix>
        required argument:
            -i | --input_snv_file : input file,result from annotation software
            -o | --out_dir : output directory
            -p | --human_peptide : reference protein sequence of human
            -r | --reference : reference fasta file
            -s | --software : software for annotation
            -e | --expression_file : expression profile with fpkm value
            -t | --tpm_threshold : TPM value cutoff, default :1
            -P | --prefix : prefix of output file
'''
for opt,value in opts:
    if opt =="h":
        print (USAGE)
        sys.exit(2)
    elif opt in ("-i","--input_snv_file"):
        input_snv_file=value
    elif opt in ("-o","--out_dir"):
        out_dir =value
    elif opt in ("-p","--human_peptide"):
        human_peptide =value 
    elif opt in ("-r","--reference"):
        reference =value 
    elif opt in ("-s","--software"):
        software =value 
    elif opt in ("-e","--expression_file"):
        expression_file =value 
    elif opt in ("-t","--tpm_threshold"):
        tpm_threshold =value 
    elif opt in ("-P","--prefix"):
        prefix =value 

if (input_snv_file =="" or out_dir =="" or human_peptide=="" or reference=="" or expression_file=="" or tpm_threshold==""):
    print (USAGE)
    sys.exit(2) 

ref_fasta1 = Fasta(reference)
ref_fasta = { k.split()[0] : ref_fasta1[k] for k in ref_fasta1.keys() }

transcript_aa={}
variation=[]
protein_position=[]
cdna_position=[]
cds_position=[]
gene_symbol=[]
trans_name=[]
ref_amino_acid=[]
alt_amino_acid=[]
ref_nucleotide=[]
alt_nucleotide=[]
chrom_pos=[]
consequence=[]
output_line_num=[]
strand=[]
tpm_num=[]

# Set gene expression list to filter mutation annotations
exp = pd.read_csv(expression_file,header=0,sep='\t')
gene_exp = exp.loc[:,['target_id','tpm']]
gene_exp = gene_exp[gene_exp['tpm']>float(tpm_threshold)]
gene_exp_list_org = gene_exp['target_id'].to_list()
gene_exp_list = [x[:-2] for x in gene_exp_list_org]
gene_exp_tpm = gene_exp['tpm'].to_list()

for line in open(human_peptide,'r'):
    if line.startswith(">"):
        transcript_name = line.strip().split(' ')[4][11:26]
        transcript_aa[transcript_name] = '' 
    else:
        transcript_aa[transcript_name] += line.replace('\n','')

if software == "VEP" :
    line_num=0
    for line in open(input_snv_file):
        line_num+=1
        if line.startswith('#'):
            continue
        elif ("missense_variant" not in line) & ("frameshift_variant" not in line) & ("inframe_insertion" not in line) & ("inframe_deletion" not in line):
            continue
        else:
            record=line.strip().split('\t')
            if record[4] not in gene_exp_list:
                continue
            #print record
            variation.append(record[0])
            chr_p=record[1]
            tran_n=record[4]
            pro_pos=record[9].split('/')[0]
            extra=record[13].split(';')
            if len(record[10]) == 1:
                continue
            alt_aa=record[10].split('/')[1]
            ref_aa=record[10].split('/')[0]
            cdna_p=record[7]
            cds_p=record[8]
            consequence_str=record[6].split(",")[0]
            ref_n=record[0].split('_')[2].split('/')[0]
            alt_n=record[0].split('_')[2].split('/')[1]
            tpm=float(gene_exp_tpm[gene_exp_list.index(tran_n)])
            if (consequence_str=="frameshift_variant") or (consequence_str=="inframe_insertion") or (consequence_str=="inframe_deletion") or ("missense_variant" in consequence_str):
                alt_amino_acid.append(alt_aa)
                ref_amino_acid.append(ref_aa)
                trans_name.append(tran_n)
                protein_position.append(pro_pos)
                cdna_position.append(cdna_p)
                cds_position.append(cds_p)
                ref_nucleotide.append(ref_n)
                alt_nucleotide.append(alt_n)
                chrom_pos.append(chr_p)
                consequence.append(consequence_str)
                output_line_num.append(line_num)
                tpm_num.append(int(round(tpm,1)*10))

elif software == "SnpEff":
    line_num=0
    snpeff_feature =[]
    for line in open(input_snv_file):
        line_num+=1
        if line.startswith('#'):
            continue
        else:
            info = line.strip().split('\t')[7].split(';')
            for item in info:
                if item.startswith("ANN"):
                    for j in item.split(','):
                        record=j.split('|')
                        consequence_str=record[1]
                        if record[6].split('.')[0] not in gene_exp_list:
                            continue
                        col_info=line.strip().split('\t')
                        chr_p=col_info[0]+":"+col_info[1]
                        tran_n=record[6].split('.')[0]
                        if len(record[13])==0:
                            continue
                        pro_pos=record[13].split('/')[0]
                        if (pro_pos not in record[10].split('.')[1]):
                            continue
                        if (consequence_str=="frameshift_variant") or ("inframe_insertion" in consequence_str) or ("inframe_deletion" in consequence_str) or ("missense_variant" in consequence_str):
                            if len(record[10].split('.')[1].split(pro_pos)[0]) != 3:
                                continue
                            if len(record[10].split('.')[1].split(pro_pos)[1]) != 3:
                                continue
                            alt_aa=shorten(record[10].split('.')[1].split(pro_pos)[1])
                            ref_aa=shorten(record[10].split('.')[1].split(pro_pos)[0])
                            cdna_p=record[11].split('/')[0]
                            ref_n=col_info[3]
                            alt_n=col_info[4]
                            cds_p=record[12].split('/')[0]
                            snpeff_feature.append(j.split('|')[6].split('.')[0])
                            tpm=float(gene_exp_tpm[gene_exp_list.index(tran_n)])
                            variation.append(col_info[0]+"_"+col_info[1] +"_"+col_info[3]+"/"+col_info[4])
                            alt_amino_acid.append(alt_aa)
                            ref_amino_acid.append(ref_aa)
                            trans_name.append(tran_n)
                            protein_position.append(pro_pos)
                            cdna_position.append(cdna_p)
                            cds_position.append(cds_p)
                            ref_nucleotide.append(ref_n)
                            alt_nucleotide.append(alt_n)
                            chrom_pos.append(chr_p)
                            consequence.append(consequence_str)
                            output_line_num.append(line_num)
                            tpm_num.append(int(round(tpm,1)*10))
elif software == "Funcotator":
    line_num=0
    funcotator_feature = []
    for line in open(input_snv_file):
        line_num+=1
        if line.startswith('#'):
            continue
        else:
            for j in line.strip().split('\t')[7].split(';'):
                if ("MISSENSE" in j):
                    record=j.strip().split('[')[1].split('|')
                    variation.append(record[2]+"_"+record[3]+"_"+record[8]+"/"+record[10])
                    chr_p=record[2]+":"+record[3]
                    if (int(record[3])!=int(record[4])):
                        chr_p+="-"+record[4]
                    tran_n=record[12]
                    if (tran_n not in gene_exp_list_org):
                        continue
                    if len(record[18])==0:
                        continue
                    pro_pos_aa=record[18].split('.')[1]
                    pro_pos=int(re.search(r'\d+', pro_pos_aa).group())
                    alt_aa=pro_pos_aa.strip().split(str(pro_pos))[1]
                    ref_aa=pro_pos_aa.strip().split(str(pro_pos))[0]
                    cdna_p=""
                    if "del" in record[16]:
                        cdna_p=record[16].strip().split('.')[1].split('del')[0]
                    elif "ins" in record[16]:
                        cdna_p=record[16].strip().split('.')[1].split('ins')[0]
                    else:
                        cdna_p=record[16].strip().split('.')[1].split(record[8])[0]
                    cds_p=record[16]
                    consequence_str=record[5]
                    tpm=float(gene_exp_tpm[gene_exp_list_org.index(tran_n)])
                    if (consequence_str=="FRAME_SHIFT_INS") or (consequence_str=="FRAME_SHIFT_DEL") or (consequence_str=="IN_FRAME_INS") or \
                       (consequence_str=="IN_FRAME_DEL") or (consequence_str=="MISSENSE"):
                        alt_amino_acid.append(alt_aa)
                        ref_amino_acid.append(ref_aa)
                        trans_name.append(tran_n)
                        protein_position.append(pro_pos)
                        cdna_position.append(cdna_p)
                        cds_position.append(cds_p)
                        # ref_nucleotide.append(ref_n)
                        # alt_nucleotide.append(alt_n)
                        chrom_pos.append(chr_p)
                        consequence.append(consequence_str)
                        tpm_num.append(int(round(tpm,1)*10))
                        output_line_num.append(line_num)
                    if len(j.strip().split('|')[21])==0:
                        continue
                    for k in j.strip().split('|')[21].split('/'):
                        if k.strip().split('_')[1].split('.')[0] not in gene_exp_list:
                            continue
                        chr_p=line.strip().split('\t')[0]
                        tran_n=k.strip().split('_')[1].split('.')[0]
                        cons=line.strip().split('_')[2]
                        if len(k.strip().split('_')[2])!='MISSENSE':
                            continue
                        pro_pos_aa=k.strip().split('_')[3].split('.')[1]
                        pro_pos = int(re.search(r'\d+', pro_pos_aa).group())
                        g_s=k.strip().split('_')[0]
                        alt_aa=pro_pos_aa.strip().split(str(pro_pos))[1]
                        ref_aa=pro_pos_aa.strip().split(str(pro_pos))[0]
                        tpm=float(gene_exp_tpm[gene_exp_list.index(tran_n)])
                        alt_amino_acid.append(alt_aa)
                        ref_amino_acid.append(ref_aa)
                        trans_name.append(tran_n)
                        protein_position.append(pro_pos)
                        gene_symbol.append(g_s)
                        chrom_pos.append(chr_p)
                        output_line_num.append(line_num)
                        tpm_num.append(int(round(tpm,1)*10))
                        funcotator_feature.append(k.strip().split('_')[1].split('.')[0])

transcript_seq=[]
for name in trans_name:
    name=name.strip().split('.')[0]
    if name not in transcript_aa.keys():
        seq='NULL'
    else:
        seq=transcript_aa[name]
    transcript_seq.append(seq)

mut_peptide=[]
wt_peptide=[]
mt_header=[]
wt_header=[]
for i in range(len(trans_name)):
    if transcript_seq[i]=="NULL":
        continue
    else:
        pro_change_pos=0
        wt_head=""
        mt_head=""
        wt_pep=""
        mt_pep=""
        
        if software=="VEP":
            pro_change_pos=int(protein_position[i].strip().split("-")[0])
        else:
            pro_change_pos=int(protein_position[i])
        ref_amino_acid_seq=transcript_seq[i]
        if (consequence[i] == "missense_variant") or (consequence[i] == "MISSENSE"):
            wt_head='>SNV_'+ str(output_line_num[i])
            mt_head='>SNV_'+ str(output_line_num[i])+"_"+str(tpm_num[i])
            if pro_change_pos<=10:
                wt_pep=ref_amino_acid_seq[0:21]
                mt_pep=ref_amino_acid_seq[0:pro_change_pos-1]+alt_amino_acid[i]+ref_amino_acid_seq[pro_change_pos:pro_change_pos+21]
            elif pro_change_pos>10 and len(ref_amino_acid_seq)-pro_change_pos<=10:
                wt_pep=ref_amino_acid_seq[len(ref_amino_acid_seq)-21:len(ref_amino_acid_seq)]
                mt_pep=ref_amino_acid_seq[len(ref_amino_acid_seq)-21:pro_change_pos-1]+alt_amino_acid[i]+ref_amino_acid_seq[pro_change_pos:len(ref_amino_acid_seq)]
            else:
                wt_pep=ref_amino_acid_seq[pro_change_pos-11:pro_change_pos+10]
                mt_pep=ref_amino_acid_seq[pro_change_pos-11:pro_change_pos-1]+alt_amino_acid[i]+ref_amino_acid_seq[pro_change_pos:pro_change_pos+10]
        elif (consequence[i] == "frameshift_variant") or (consequence[i] == "FRAME_SHIFT_INS") or (consequence[i] == "FRAME_SHIFT_DEL") or \
             ("inframe_insertion" in consequence[i]) or (consequence[i] == "IN_FRAME_INS") or ("inframe_deletion" in consequence[i]) or \
             (consequence[i] == "IN_FRAME_DEL"):
            #if (consequence[i] == "frameshift_variant") or (consequence[i] == "FRAME_SHIFT_INS") or (consequence[i] == "FRAME_SHIFT_DEL"):
            #    indel_conseq = 'frameshift'
            #else:
            #    indel_conseq = 'inframe'
            indel_conseq=''
            if (consequence[i] in ["FRAME_SHIFT_INS", "IN_FRAME_INS", "inframe_insertion"]):
                head_id = 'INS'
            if (consequence[i] in ["FRAME_SHIFT_DEL", "IN_FRAME_DEL", "inframe_deletion"]):
                head_id = 'DEL'
            if (consequence[i] in ["frameshift_variant"]):
                head_id = 'NA'
            wt_head = F'>{head_id}_'+ str(output_line_num[i])
            mt_head = F'>{head_id}_'+ str(output_line_num[i])+"_"+str(tpm_num[i])
            chr = chrom_pos[i].split(":")[0]
            start_pos=int(chrom_pos[i].split(":")[1].split("-")[0])
            end_pos=start_pos
            if ("-" in chrom_pos[i]):
                end_pos=int(chrom_pos[i].split(":")[1].split("-")[1])
            if "-" in cds_position[i]:
                # cds_loc=int(cds_position[i].split('-')[0])
                cds_loc=int(cds_position[i].split('/')[0].split('-')[-1]) # Generate YQANVVWKV
            elif '_' in cds_position[i]:
                cds_loc=int(cds_position[i].split('_')[0].split('.')[1])
            else:
                cds_loc=int(cds_position[i].split('/')[0])
            if type(protein_position[i]) ==int:
                protein_start=int(protein_position[i])
            elif "-" in protein_position[i]:
                protein_start=int(protein_position[i].split('-')[0])
            else:
                protein_start=int(protein_position[i])
            frame_shift_num=(cds_loc-1)%3
            trans_n=trans_name[i].strip().split('.')[0]
            seq=transcript_aa[trans_n]
            from_base=variation[i].strip().split("_")[2].split("/")[0]
            to_base=variation[i].strip().split("_")[2].split("/")[1]
            cdna_start=cdna_position[i].split('/')[0].split('-')[0]
            cdna_end=cdna_start
            if '/' in cdna_position[i]:
                cdna_end=cdna_position[i].split('/')[1]
            if '_' in cdna_position[i]:
                cdna_end=cdna_position[i].split('_')[1]

            length=int(cdna_end)-int(cdna_start)
            first_ten_aa=""
            if protein_start<11:
                first_ten_aa=seq[0:protein_start-1]
            else:
                first_ten_aa=seq[protein_start-11:protein_start-1]
            if consequence[i] == "frameshift_variant":
                if (len(from_base)>len(to_base)) or (to_base=="-"): # the frameshift is due to deletion
                    dna_seq = seq2str(ref_fasta[chr][start_pos-frame_shift_num:start_pos]) + seq2str(ref_fasta[chr][end_pos:start_pos+length-frame_shift_num])
                    aa_seq = translate(dna_seq)
                elif (len(from_base)<len(to_base)) or (from_base=="-"): # the frameshift is due to insertion       
                    dna_seq = seq2str(ref_fasta[chr][start_pos-frame_shift_num:start_pos]) + to_base + seq2str(ref_fasta[chr][start_pos:start_pos+length-frame_shift_num])
                    aa_seq = translate(dna_seq)
                else:
                    print("[WARNING] Possibly wrong output consequence: {} {}:{} {}->{}".format(trans_name[i], chr, start_pos, from_base, to_base))
                    continue
                mt_pep=first_ten_aa+aa_seq
                wt_pep=seq[protein_start-11:protein_start-11+len(mt_pep)]
                if variation[i] == 'chr12_62775391_-/T': print('chr12_62775391_-/T ->  {} (frame_shift_num = {})'.format(aa_seq, frame_shift_num))
            elif consequence[i] == "FRAME_SHIFT_INS":
                dna_seq = seq2str(ref_fasta[chr][start_pos-frame_shift_num:start_pos]) + to_base + seq2str(ref_fasta[chr][start_pos:start_pos+length-frame_shift_num])
                aa_seq = translate(dna_seq)
                mt_pep=first_ten_aa+aa_seq
                wt_pep=seq[protein_start-11:protein_start-11+len(mt_pep)]
            elif consequence[i] == "FRAME_SHIFT_DEL":
                dna_seq = seq2str(ref_fasta[chr][start_pos-frame_shift_num:start_pos]) + to_base + seq2str(ref_fasta[chr][start_pos:start_pos+length-frame_shift_num])
                aa_seq = translate(dna_seq)
                mt_pep=first_ten_aa+aa_seq
                wt_pep=seq[protein_start-11:protein_start-11+len(mt_pep)]
            elif ("inframe_insertion" in consequence[i]) or (consequence[i] == "IN_FRAME_INS"):
                mt_pep=first_ten_aa+alt_amino_acid[i]+seq[protein_start:protein_start+10]
                wt_pep=seq[protein_start-10:protein_start-10+len(mt_pep)]
            elif ("inframe_deletion" in consequence[i]) or (consequence[i] == "IN_FRAME_DEL"):
                protein_end=protein_start+len(alt_amino_acid[i])
                if (alt_amino_acid[i]=="-"):
                    mt_pep=first_ten_aa+seq[protein_end:protein_end+10]
                else:
                    mt_pep=first_ten_aa+alt_amino_acid[i]+seq[protein_end:protein_end+10]
                wt_pep=seq[protein_start-10:protein_start-10+len(mt_pep)]
            
            else:
                print("[WARNING] Wrong Consequence!")
                break
        mt_header.append(mt_head)
        wt_header.append(wt_head)
        mut_peptide.append(mt_pep)
        wt_peptide.append(wt_pep)

mut_pep_len=[]
wt_pep_len=[]
for i in range(len(mut_peptide)):
    m_p_l=len(mut_peptide[i])
    w_p_l=len(wt_peptide[i])
    mut_pep_len.append(m_p_l)
    wt_pep_len.append(w_p_l)
#####drop duplicate###
fasta_out=pd.DataFrame()
fasta_out['mutation_header']=mt_header
fasta_out['mutation_peptide']=mut_peptide
fasta_out['wild_header']=wt_header
fasta_out['wild_peptide']=wt_peptide
fasta_out['mut_peptide_length']= mut_pep_len
fasta_out['wt_peptide_length']= wt_pep_len
fasta_dd=fasta_out.drop_duplicates(subset=['mutation_header','mutation_peptide','wild_header','wild_peptide','mut_peptide_length'])
data_filter=fasta_dd[(fasta_dd["mut_peptide_length"]>=8) & (fasta_dd["mut_peptide_length"]==fasta_dd["wt_peptide_length"])]
data_dd_reindex=data_filter.reset_index()
del data_dd_reindex['index']
#######write######
f_w=open(out_dir+"/"+prefix+"_snv_indel.fasta",'w')
for i in range(len(data_dd_reindex.mutation_header)):
    f_w.write('%s%s%s%s'%(data_dd_reindex.mutation_header[i],'\n',data_dd_reindex.mutation_peptide[i],'\n'))
f_w.close()

tmp_fasta_folder = os.path.join(out_dir,prefix+"_tmp_fasta")
if not os.path.exists(tmp_fasta_folder):
    os.mkdir(tmp_fasta_folder)
f_w=open(tmp_fasta_folder+"/"+prefix+"_snv_indel_wt.fasta",'w')
for i in range(len(data_dd_reindex.mutation_header)):
    f_w.write('%s%s%s%s'%(data_dd_reindex.wild_header[i],'\n',data_dd_reindex.wild_peptide[i],'\n'))
f_w.close()
