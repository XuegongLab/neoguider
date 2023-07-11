#!/usr/bin/python
# -*- coding: UTF-8 -*-
###########netMHC result parsing and filter based on binding affinity and FPKM #########
import collections,getopt,math,os,sys
import pandas as pd 

def correct_RNA_quant(df1):
    df = df1
    mtpep_idx = df.columns.get_loc("MT_pep")
    quant_idx = df.columns.get_loc("Quantification")
    mtpep2quant = collections.defaultdict(int)
    for idx in range(len(df)):
        mtpep = df.iloc[idx,mtpep_idx]
        quant = df.iloc[idx,quant_idx]
        mtpep2quant[mtpep] += int(quant)
    for idx in range(len(df)):
        mtpep = df.iloc[idx,mtpep_idx]
        df.iloc[idx,quant_idx] = mtpep2quant[mtpep]
    return df

opts,args=getopt.getopt(sys.argv[1:],"i:g:o:b:l:p:",["input_dir","input_fasta","out_dir","binding_affinity_cutoff","hla_str","prefix"])
input_dir=""
input_fasta=""
out_dir=""
binding_affinity_cutoff=500
hla_str=""
prefix=""
USAGE='''usage: python parse_netMHC.py -i <input_dir> -g <input_fasta> -o <out_dir> 
                                        -b <binding_affinity_cutoff> -l <hla_str> -p <prefix>
        required argument:
            -i | --input_dir : input file,result from netMHC
            -g | --input_fasta : input fasta file for netMHC
            -o | --out_dir : output directory
            -b | --binding_affinity_cutoff : pipetide binding affinity cutoff , default: 500 nM
            -l | --hla_str : hla type string derived from opitype
            -p | --prefix : prefix of output file'''
    
for opt,value in opts:
    if opt =="h":
        print (USAGE)
        sys.exit(2)
    elif opt in ("-i","--input_dir"):
        input_dir=value
    elif opt in ("-g","--input_fasta"):
        input_fasta=value
    elif opt in ("-o","--out_dir"):
        out_dir =value  
    elif opt in ("-b","--binding_affinity_cutoff"):
        binding_affinity_cutoff =value
    elif opt in ("-l","hla_str"):
        hla_str=value
    elif opt in ("-p","prefix"):
        prefix=value
        
if (input_dir =="" or input_fasta =="" or out_dir =="" or hla_str==""):
    print (USAGE)
    sys.exit(2)        
#######extract full animo acid change##
Full_type_mutation=[]
Full_identity=[]
with open(input_fasta) as f:
    data=f.read()
mutID2tpm = {}
for rawline in data.strip().split('\n'):
    line = rawline.strip().split()[0]
    if not line.startswith('>'): continue
    full_type_mutation=line.split('_')[0].split('>')[1]
    full_identity=line.split('>')[1]
    Full_type_mutation.append(full_type_mutation)
    Full_identity.append(full_identity)
    
    mutID = '_'.join(line[1:].split('_')[0:2])
    for i, subline in enumerate(rawline.split()):
        if i > 0 and len(subline.split('=')) == 2:
            key, val = subline.split('=')
            if key == 'TPM': mutID2tpm[mutID] = float(val)
#print(mutID2tpm)
dup_type_mutation=[]
dup_full_identity=[]
hla_num=len(hla_str.split(','))
i=0
while i<hla_num:
    for j in range(len(Full_identity)):
        dup_type_mutation.append(Full_type_mutation[j])
        dup_full_identity.append(Full_identity[j])
    i=i+1


######## extract candidate neoantigens####
with open(input_dir+"/"+prefix+"_bindaff_raw.tsv") as f:
    data = f.read()
nw_data = data.split('-----------------------------------------------------------------------------------\n')
WT_header = []
MT_header = []
WT_neo = []
MT_neo = []
for i in range(len(nw_data)):
    if i%4 == 3:
        mt_pro_name = nw_data[i].strip('\n').split('.')[0]
        MT_header.append(mt_pro_name)
    elif i%4 == 2:
        mt_neo_data = nw_data[i].strip().split('\n')
        MT_neo.append(mt_neo_data)
WB_SB_MT_record = []
Identity = []


count=0
for i in range(len(MT_neo)):
    for j in range(len(MT_neo[i])):
        if MT_neo[i][j] == '----------------------------------------':
            continue
        is_SNV_or_INDEL =  (("SNV_" in MT_neo[i][j]) or ("INDEL" in MT_neo[i][j]) or ("INS_" in MT_neo[i][j]) or ("DEL_" in MT_neo[i][j]))
        if MT_neo[i][j].endswith('WB') or MT_neo[i][j].endswith('SB'):
            if is_SNV_or_INDEL:
                Identity.append(str(count))
            else:
                Identity.append("NA")
            WB_SB_MT_record.append(MT_neo[i][j])
        if is_SNV_or_INDEL:
            count+=1

with open(input_dir+"/"+prefix+"_snv_indel_bindaff_wt.tsv") as f:
    data = f.read()
    nw_data = data.split('-----------------------------------------------------------------------------------\n')
    WT_header = []
    WT_neo = []
    for i in range(len(nw_data)):
        if i%4 == 3:
            wt_pro_name = nw_data[i].strip('\n').split('.')[0]
            WT_header.append(wt_pro_name)
        elif i%4 == 2:
            wt_neo_data = nw_data[i].strip().split('\n')
            WT_neo.append(wt_neo_data)
    wt_bindaff_list=[]
    wt_list=[]
    for i in range(len(WT_neo)):
        for j in range(len(WT_neo[i])):
            if "----" in WT_neo[i][j]:
                continue
            wt_bindaff_list.append(WT_neo[i][j].strip().split()[15])
            wt_list.append(WT_neo[i][j].strip().split()[2])

    data_form = []
    data_form_tmp = []
    for i in range(len(WB_SB_MT_record)):
        line=[]
        mt_record = [line for line in WB_SB_MT_record[i].split(' ') if line!='']
        assert len(mt_record) > 10, F'The {i}-th mutation peptide {mt_record} is invalid!'
        HLA_tp = mt_record[1]
        mt_pep = mt_record[2]
        wt_pep = mt_record[3]
        mt_binding_aff= mt_record[15]
        mt_binding_level_des = mt_record[-1]
        iden = mt_record[10]
        wt_binding_aff = ""
        if "SNV" in mt_record[10] or "INDEL" in mt_record[10] or "INS" in mt_record[10] or "DEL" in mt_record[10]:
            wt_pos = int(Identity[i])
            wt_pep = wt_list[wt_pos]
            wt_binding_aff = wt_bindaff_list[wt_pos]
        if wt_pep == mt_pep:
            continue
        assert len(str(mt_record[10]).strip().split('_')) > 2, F'The {i}-th mutation peptide {mt_record} is invalid (error-2)!'
        #tpm = float(str(mt_record[10]).strip().split('_')[2])/10
        iden = iden.split("_")[0]+"_"+iden.split("_")[1]
        tpm = mutID2tpm[iden]
        line = [HLA_tp,mt_pep,wt_pep,float(mt_binding_aff),mt_binding_level_des,iden,tpm]
        line_tmp = [HLA_tp,mt_pep,wt_pep,float(mt_binding_aff),mt_binding_level_des,iden,tpm,wt_binding_aff]
        data_form.append(line)
        data_form_tmp.append(line_tmp)
        if (i % 200) == 0: print("finish append")

f=lambda x: x.split('.')[0]
fields = ['HLA_type','MT_pep','WT_pep','BindAff','BindLevel','Identity','Quantification']
######neoantigens filtering binding affinity#####
data= pd.DataFrame(data_form)
data.columns=fields
final_filter_data=data[(data.BindAff<float(binding_affinity_cutoff))] # filter binding affinity
final_filter_data = correct_RNA_quant(final_filter_data)
final_filter_data=final_filter_data.drop_duplicates(subset=['HLA_type','MT_pep','WT_pep','BindAff','BindLevel'])
final_filter_data.to_csv(out_dir+"/"+prefix+"_bindaff_filtered.tsv",header=1,sep='\t',index=0)

data_tmp= pd.DataFrame(data_form_tmp)
fields.append("WT_BindAff")
data_tmp.columns=fields
final_filter_data_tmp=data_tmp[(data_tmp.BindAff<float(binding_affinity_cutoff))] # filter binding affinity
final_filter_data_tmp = correct_RNA_quant(final_filter_data_tmp)
final_filter_data_tmp=final_filter_data_tmp.drop_duplicates(subset=['HLA_type','MT_pep','WT_pep','BindAff','BindLevel'])
if not os.path.exists(out_dir+"/tmp_identity"):
    os.mkdir(out_dir+"/tmp_identity")
final_filter_data_tmp.to_csv(out_dir+"/tmp_identity/"+prefix+"_bindaff_filtered.tsv",header=1,sep='\t',index=0)
