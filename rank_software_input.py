import getopt,sys
import csv

def write_file(a_list, name):
    textfile = open(name, "w")
    # for element in a_list:
    textfile.write(a_list + "\n")
    textfile.close()
    return

#####prepare csv format input file for neoantigen rank software######
opts,args=getopt.getopt(sys.argv[1:],"hm:n:o:t:p:",["mixcr_prefix=","neoantigen_input=","output_folder=","rank_software_type=","prefix="])
mixcr_prefix =""
neoantigen_input=""
output_folder=""
rank_software_type=""
prefix=""
USAGE='''
    This script analyze mixcr result and neoantigen output in single file for nettcr
    usage: python rank_software_input.py -m <mixcr_prefix> -n <neoantigen_input> -o <output_folder> -t <rank_software_type> -p <prefix>
        required argument:
            -m | --mixcr_prefix : mixcr output file prefix
            -n | --neoantigen_input : neoantigen prediction file from parameter filters
            -o | --output_folder : output folder to store result
            -t | --rank_software_type : software for neoantigen ranking
            -p | --prefix : prefix of output file
'''
for opt,value in opts:
    if opt =="h":
        print (USAGE)
        sys.exit(2)
    elif opt in ("-m","--mixcr_prefix"):
        mixcr_prefix=value
    elif opt in ("-n","--neoantigen_input"):
        neoantigen_input =value
    elif opt in ("-o","--output_folder"):
        output_folder =value
    elif opt in ("-t","--rank_software_type"):
        rank_software_type =value
    elif opt in ("-p","--prefix"):
        prefix =value
if (mixcr_prefix =="" or neoantigen_input =="" or output_folder =="" or rank_software_type==""):
	print (USAGE)
	sys.exit(2)	

TRA_file = mixcr_prefix+".clonotypes.TRA.txt"
TRB_file = mixcr_prefix+".clonotypes.TRB.txt"
CDR3_b_list = []
CDR3_a_list = []
CDR3_b_list = []
CDR3_av_name = []
CDR3_bv_name = []
CDR3_aj_name = []
CDR3_bj_name = []
neoantigen_list = []
hla_list = []


reader_neo = csv.reader(open(neoantigen_input), delimiter=",")
next(reader_neo, None)
for line in reader_neo:
    if (line[1]==line[2]):
        continue
    neoantigen = line[1]
    hla = line[0]
    neoantigen_list.append(neoantigen)
    hla_list.append(hla)

if (rank_software_type == "ERGO"):
    
    reader_a = csv.reader(open(TRA_file), delimiter="\t")
    next(reader_a, None)
    for line in reader_a:
        CDR3_a = line[32]
        CDR3_a_list.append(CDR3_a)
        CDR3_av_name.append(line[5].strip().split("*")[0])
        CDR3_aj_name.append(line[7].strip().split("*")[0])

    reader_b = csv.reader(open(TRB_file), delimiter="\t")
    next(reader_b, None)
    for line in reader_b:
        CDR3_b = line[32]
        CDR3_b_list.append(CDR3_b)
        CDR3_bv_name.append(line[5].strip().split("*")[0])
        CDR3_bj_name.append(line[7].strip().split("*")[0])

        
    fields = ['TRA','TRB','TRAV','TRAJ','TRBV','TRBJ','T-Cell-Type','Peptide','MHC']
    rows = []
    for cdra in range(0,len(CDR3_a_list),1):
        r = [CDR3_a_list[cdra]]
        for cdrb in range(0,len(CDR3_b_list),1):
            r.append(CDR3_b_list[cdrb])
            r.append(CDR3_av_name[cdra])
            r.append(CDR3_aj_name[cdra])
            r.append(CDR3_bv_name[cdrb])
            r.append(CDR3_bj_name[cdrb])
            r.append("CD8")
            for i in range(0,len(neoantigen_list),1):
                r.append(neoantigen_list[i])
                r.append(hla_list[i])
                rows.append(r)
                r = [CDR3_a_list[cdra], CDR3_b_list[cdrb], CDR3_av_name[cdra], CDR3_aj_name[cdra], CDR3_bv_name[cdrb], CDR3_bj_name[cdrb], "CD8"]
            r = [CDR3_a_list[cdra]]

    with open(output_folder+"/"+prefix+"_cdr_ergo.csv","w") as f:
        write = csv.writer(f)
        write.writerow(fields)
        write.writerows(rows)

else:
    print("[WARNING] No output because of invalid software name. (ERGO)")
