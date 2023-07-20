
import csv, logging, sys, os
import getopt
import pandas as pd

def write_file(a_list, name):
    textfile = open(name, "w")
    textfile.write(a_list + "\n")
    textfile.close()
    return

#####prepare csv format input file for neoantigen rank software######
opts,args=getopt.getopt(sys.argv[1:],"hi:n:o:t:p:",["input_rank_result=","neoantigen_input=","output_folder=","rank_software_type=","prefix="])
input_rank_result =""
neoantigen_input=""
output_folder=""
rank_software_type=""
prefix=""
USAGE='''
    This script analyze mixcr result and neoantigen output in single file for nettcr
    usage: python parse_rank_software.py -i <input_rank_result> -n <neoantigen_input> -o <output_folder> -t <rank_software_type> -p <prefix>
        required argument:
            -i | --input_rank_result : rank result from rank software
            -n | --neoantigen_input : neoantigen prediction file from parameter filters
            -o | --output_folder : output folder to store result
            -t | --rank_software_type : software for neoantigen ranking
            -p | --prefix : prefix of output file
'''
for opt,value in opts:
    if opt =="h":
        print (USAGE)
        sys.exit(2)
    elif opt in ("-i","--input_rank_result"):
        input_rank_result=value
    elif opt in ("-n","--neoantigen_input"):
        neoantigen_input =value
    elif opt in ("-o","--output_folder"):
        output_folder =value
    elif opt in ("-t","--rank_software_type"):
        rank_software_type =value
    elif opt in ("-p","--prefix"):
        prefix =value
if (input_rank_result =="" or neoantigen_input =="" or output_folder =="" or rank_software_type==""):
	print (USAGE)
	sys.exit(2)	

neoantigen_list=[]
reader_neo = csv.reader(open(neoantigen_input), delimiter="\t")
fields = next(reader_neo)

hla_index = fields.index('HLA_type')
neoantigen_index = fields.index('ET_pep')
wt_peptide_index = fields.index('WT_pep')
identity_index = fields.index('Identity')

for line in reader_neo:
    identity=line[identity_index]
    logging.debug(F'identity = {identity}')
    line[identity_index]=identity.split("_")[0]+"_"+identity.split("_")[1]
    line.append(0)
    neoantigen_list.append(line)

final_rank = []
if os.path.exists(input_rank_result):
    reader = csv.reader(open(input_rank_result), delimiter=",")
    next(reader, None)
else:
    logging.warning(F'The file input_rank_result={input_rank_result} does not exist!')
    reader = []
max_score = 0
tcr_max_score = []
neoantigen = ""
if rank_software_type == "ERGO":
    for line in reader:
        neoantigen = line[7]
        break
    for line in reader:
        if (line[7] == neoantigen) & (float(line[9])>float(max_score)):
            max_score = float(line[9])
        elif line[7]!=neoantigen:
            tcr_max_score.append([line[8],neoantigen,max_score])
            neoantigen = line[7]
            max_score = float(line[9])
        else:
            continue
    for neo in range(0,len(neoantigen_list),1):
        if len(tcr_max_score)!=0:
            neoantigen_list[neo][-1] = tcr_max_score[neo][2]
        final_rank.append(neoantigen_list[neo])
else:
    print("[WARNING] No output because of invalid software name. (ERGO)")
fields.append("TCRSpecificityScore")
data=pd.DataFrame(final_rank, columns = fields)
#data.columns=fields
data["Rank"]=data["TCRSpecificityScore"].rank(method='first',ascending=False)
data=data.sort_values("Rank")
data=data.astype({"Rank":int})
data=data.drop(columns=['BindLevel'], axis=1)
data.to_csv(output_folder+"/"+prefix+"_neoantigen_rank_tcr_specificity.tsv",header=1,sep='\t',index=0)
