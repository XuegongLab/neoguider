import sys,getopt
import csv
import pandas as pd
from pyfaidx import Fasta

def write_file(a_list, name):
    textfile = open(name, "w")
    textfile.write(a_list + "\n")
    textfile.close()
    return


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
    for i in range(0, len(seq), 3):
        codon = seq[i:i + 3]
        if table[codon] == '_':
            break
        protein+= table[codon]
    return protein


def reverse(seq):
	reverse_list=[]
	seq_f_list=list(seq)
	for ele in seq_f_list:
		if ele=='A':
			reverse_ele='T'
		elif ele=='T':
			reverse_ele='A'
		elif ele=='C':
			reverse_ele='G'
		else:
			reverse_ele='C'
		reverse_list.append(reverse_ele)
	seq_str=''.join(reverse_list)
	reverse_seq=seq_str[::-1]
	return reverse_seq


#####prepare fasta format input file for netMHC######
opts,args=getopt.getopt(sys.argv[1:],"hi:e:o:p:t:",["star_fusion_prediction=","expression_file=","output_folder=","prefix=","tpm_threshold="])
star_fusion_prediction =""
expression_file=""
output_folder=""
prefix=""
tpm_threshold=1
USAGE='''
    This script convert fusion prediction result to fasta format file for netMHC
    usage: python parse_star_fusion.py -i <star_fusion_prediction> -e <expression_file> -o <output_folder> 
                                        -t <tpm_threshold> -p <prefix>
        required argument:
            -i | --star_fusion_prediction : input file,result from STAR-Fusion
            -e | --expression_file : transcript expression level file
            -o | --output_folder : output folder to store result
            -t | --tpm_threshold : threshold for tumor abundance
            -p | --prefix : prefix for output file
'''
for opt,value in opts:
    if opt =="h":
        print (USAGE)
        sys.exit(2)
    elif opt in ("-i","--star_fusion_prediction"):
        star_fusion_prediction=value
    elif opt in ("-e","--expression_file"):
        expression_file=value
    elif opt in ("-o","--output_folder"):
        output_folder =value 
    elif opt in ("-t","--tpm_threshold"):
        tpm_threshold =value 
    elif opt in ("-p","--prefix"):
        prefix =value 

if (star_fusion_prediction =="" or output_folder=="" or prefix==""):
	print (USAGE)
	sys.exit(2)		

exp = pd.read_csv(expression_file,header=0,sep='\t')
gene_exp = exp.loc[:,['target_id','tpm']]
gene_exp_list = gene_exp['target_id'].to_list()
gene_exp_tpm = gene_exp['tpm'].to_list()

reader = csv.reader(open(star_fusion_prediction), delimiter="\t")
next(reader, None)
mt_head_list=[]
mt_pep_seq_list=[]
output_line_num=[]
line_num=1
for line in reader:
    line_num+=1
    if len(line[23]) > 1:
        s = str(line[23])
        l1 = int(line[18].strip().split("-")[1])
        chr = line[7].strip().split(":")[0]
        break_point = int(line[7].strip().split(":")[1])
        consequence = line[21]
        exp_left = line[17]
        exp_right = line[19]
        if (consequence != "FRAMESHIFT") & (consequence != "INFRAME"):
            continue
        if ((exp_left not in gene_exp_list) or (exp_right not in gene_exp_list)):
            continue
        tpm = (float(gene_exp_tpm[gene_exp_list.index(exp_left)]) + float(gene_exp_tpm[gene_exp_list.index(exp_right)]))/2
        if (tpm < float(tpm_threshold)):
            continue
        mt_head='>FUSION_'+ str(line_num)+"_"+str(round(tpm,1)*10)
        mt_head_list.append(mt_head)
        cds=""
        if l1<=30 :
            if (consequence == "FRAMESHIFT"):
                cds = line[23][0:].strip().split('_')[0]
            elif (consequence == "INFRAME"):
                cds_end = l1+30
                cds = line[23][0:cds_end].strip().split('_')[0]
            else:
                continue
            mt_pep_seq = translate(cds.upper())
            mt_pep_seq_list.append(mt_pep_seq)
        else:
            cds_start = (l1//3 - 10+1)*3
            if (consequence == "FRAMESHIFT"):
                cds = line[23][cds_start:].strip().split('_')[0]
            elif (consequence == "INFRAME"):
                cds_end = cds_start+60
                cds = line[23][cds_start:cds_end].strip().split('_')[0]
            else:
                continue
            mt_pep_seq = translate(cds.upper())
            mt_pep_seq_list.append(mt_pep_seq)
        output_line_num.append(line_num)

mut_pep_len=[]
wt_pep_len=[]
for i in range(len(mt_pep_seq_list)):
    m_p_l=len(mt_pep_seq_list[i])
    mut_pep_len.append(m_p_l)
#####drop duplicate###
fasta_out=pd.DataFrame()
fasta_out['mutation_header']=mt_head_list
fasta_out['mutation_peptide']=mt_pep_seq_list
fasta_out['mut_peptide_length']= mut_pep_len
fasta_dd=fasta_out.drop_duplicates(subset=['mutation_header','mutation_peptide','mut_peptide_length'])
data_filter=fasta_dd[(fasta_dd["mut_peptide_length"]>=8)]
data_dd_reindex=data_filter.reset_index()
del data_dd_reindex['index']
#######write######

f_w=open(output_folder+"/"+prefix+"_fusion.fasta",'w')
for i in range(len(data_dd_reindex.mutation_header)):
    f_w.write('%s%s%s%s'%(data_dd_reindex.mutation_header[i],'\n',data_dd_reindex.mutation_peptide[i],'\n'))
f_w.close()
