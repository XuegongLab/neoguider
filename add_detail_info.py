import os, getopt, logging, sys
import csv
import pandas as pd

logging.basicConfig(level=logging.INFO)

def main():
    opts,args=getopt.getopt(sys.argv[1:],"hi:o:p:",["input_file=","output_folder=", "prefix="])
    input_file=""
    output_folder=""
    prefix=""
    USAGE='''
        This script convert fusion prediction result to fasta format file for netMHC
        usage: python bindaff_related_prioritization.py -i <input_folder> -o <output_folder> \
            -f <foreignness_score> -a <agretopicity> -t <alteration_type> -p <prefix>
            required argument:
                -i | --input_file : input folder including result file from bindstab output
                -o | --output_folder : output folder to store result
                -p | --prefix : prefix of output file
    '''
    for opt,value in opts:
        if opt =="h":
            print (USAGE)
            sys.exit(2)
        elif opt in ("-i","--input_file"):
            input_file=value
        elif opt in ("-o","--output_folder"):
            output_folder =value 
        elif opt in ("-p","--prefix"):
            prefix =value 

    if (input_file =="" or output_folder==""):
        print (USAGE)
        sys.exit(2)

    snv_indel_file = open(output_folder+"../info/"+prefix+"_DNA_snv_indel.annotation.tsv")
    if os.path.exists(output_folder+"../info/"+prefix+"_fusion.tsv"):
        fusion_file = open(output_folder+"../info/"+prefix+"_fusion.tsv")
    else:
        fusion_file = []
    if os.path.exists(output_folder+"../info/"+prefix+"_splicing.csv"):
        splicing_file = open(output_folder+"../info/"+prefix+"_splicing.csv")
    else:
        splicing_file = []
    snv_indel = []
    fusion = []
    splicing = []

    for line in snv_indel_file:
        snv_indel.append(str(line))
    for line in fusion_file:
        fusion.append(str(line))
    for line in splicing_file:
        splicing.append(str(line))
    
    reader = csv.reader(open(input_file), delimiter="\t")
    fields=next(reader)
    identity_index = fields.index('Identity')
    data_raw = []
    for line in reader:        
        identity = line[identity_index]
        line_info_string = ""
        if (identity.strip().split('_')[0] in ["SNV", "INS", "DEL", "INDEL"]):
            fastaID = identity.strip().split('_')[1]
            if fastaID[0] != 'D' : continue
            line_num = int(fastaID[1:])
            snv_indel_line = snv_indel[line_num-1]
            ele = snv_indel_line.strip().split('\t')
            if len(ele) == 14: # annotation software is vep
                annotation_info = ["Uploaded_variation","Location","Allele","Gene","Feature","Feature_type",
                                    "Consequence","cDNA_position","CDS_position","Protein_position","Amino_acids","Codons","Existing_variation","Extra"]
                for i in range(0,len(ele),1):
                    line_info_string+=annotation_info[i]+"$"+ele[i]+"#"
            elif len(ele)==11:
                annotation_info = ["CHROM","POS","ID","REF","ALT","QUAL","FILTER","INFO","FORMAT","normal","tumor"]
                for i in range(0,len(ele),1):
                    line_info_string+=annotation_info[i]+"$"+ele[i]+"#"
            else:
                continue
        elif (identity.strip().split('_')[0]=="FUS"):
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
        # line[4] = identity.strip().split('_')[0]
        line.insert(-1, line_info_string)
        data_raw.append(line)
        logging.debug(line)

    fields.append("SourceAlterationDetail")
    data=pd.DataFrame(data_raw)
    data.columns=fields
    data.to_csv(output_folder+"/"+prefix+"_neoantigen_rank_tcr_specificity_with_detail.tsv",header=1,sep='\t',index=0)


if __name__ == '__main__':
    main()
