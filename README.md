# NeoGuider

NeoGuider (NG, ng) is a neoantigen prioritization pipeline and machine-learning algorithm.
NG is able to sensitively detect and accurately rank neoantigen candidates.
Additionally, NG is able to estimate the probability that each neoantigen candidate is immunogenic
  (i.e., tested and thus validated to be positive by an immnoassay such as MHC-I multimer or IFN-gamma ELISpot).

## How to install

First, follow the instruction at https://bioconda.github.io/ to install bioconda if you haven't done so.

Then, run the following commands:
```
# If the working directory is not neoguider, then change it to neoguider
sh -evx install-step-1-by-conda.sh ng                              # install all dependencies by conda
conda run -n ng sh -evx install-step-2-for-neoepitope-detection.sh # download tools and database for detecting neoepitope candidates (in FASTA format) from sequencing data
conda run -n ng sh -evx install-step-2-for-feature-extraction.sh   # download tools and database for extracting features (in TSV format) from neoepitope candidates
```
If the "$conda install" command in install-step-1-by-conda.sh ran into any error, you can try replacing this command by each of the following commands:
```
conda create     --name ng --file env/requirements.list_e_no_pypi.txt    # if this command does not work then try the next one below
conda env create --name ng --file env/freeze.env_export.yml              # if this command does not work then try the next one below
conda env create --name ng --file env/freeze.env_export_no_builds.yml    # if this command does not work then try the next one below
conda env create --name ng --file env/freeze.env_export_from_history.yml
```

The tools netMHCpan, netMHCstabpan, and MixCR can only be manually downloaded and configured due to their licensing requirements.

(1) For using netMHCpan and netMHCstabpan (neoepitope feature extractors), please refer to https://services.healthtech.dtu.dk/services/NetMHC-4.0/ and https://services.healthtech.dtu.dk/services/NetMHCstabpan-1.0/ for how to manually download, install and activate netMHCpan and netMHCstabpan.
After doing so, please set the full paths of netMHCpan and netMHCstabpan in the config.ini file accordingly.

(2) For using MixCR (TCR-clonotype detector), please obtain a license from https://licensing.milaboratories.com/ and then run the command ```software/mixcr activate-license``` to activate MixCR.

Last but no the least, please be aware that UVC, netMHCpan, netMHCstabpan, PRIME, and MixCR are free for academic use but require commercial licensing for for-profit use.

## How to run

### How to use FASTQ files as input

Example shell command to run NeoGuider with FASTQ files as input:
```
# If the working directory is not neoguider, then change it to neoguider
snakemake --configfile config.yaml --config \
    res=${neoOutDir} \
    prefix=${patientID} \
    dna_tumor_fq1=${DNAtumorR1fastq} \
    dna_tumor_fq2=${DNAtumorR2fastq} \
    dna_normal_fq1=${DNAnormalR1fastq} \
    dna_normal_fq2=${DNAnormalR2fastq} \
    rna_tumor_fq1=${RNAtumorR1fastq} \
    rna_tumor_fq2=${RNAtumorR1fastq} \
    --resources mem_mb=960000 --cores 24
```
After a successful run, you should be able check neoantigen prioritization results at: ${NeoOutDir}/${patientID}_prioritization_from_reads.tsv

### How to use a peptide FASTA file as input

Example shell command to run NeoGuider on a peptide FASTA file and a string of HLA alleles as input:
```
snakemake --configfile config.yaml --config \
    res=${neoOutDir} \
    prefix=${patientID} \
    comma_sep_hla_list='HLA-A01:01,HLA-A02:01' \
    tumor_spec_peptide_fasta=${tumorSpecificPeptideFasta} \
    --cores 4
```
where the ${tumorSpecificPeptideFasta} file can contain the following content as an example:
```
>SNV_D0_B0 WT=ARDPHSGHFV MT=ALDPHSGHFV HLA=HLA-A01:01,HLA-A02:01 TPM=145.492026549908 SOURCE=...ARDPHSGHFV... MAX_BIT_DIST=0
ALDPHSGHFV
>SNV_D0_A  WT=ARDPHSGHFV MT=ALDPHSGHFV HLA=HLA-A01:01,HLA-A02:01 TPM=145.492026549908
ARDPHSGHFV
>SNV_D1_B0 WT=ALGPGVPHI  MT=ALSPVIPHI  HLA=HLA-A01:01,HLA-A02:01 TPM=6.306862689603 SOURCE=...ALGPGVPHI... MAX_BIT_DIST=0
ALSPVIPHI
>SNV_D1_A  WT=ALGPGVPHI  MT=ALSPVIPHI  HLA=HLA-A01:01,HLA-A02:01 TPM=6.306862689603
ALGPGVPHI
>SNV_D2_B0 WT=ALIHFLMIL  MT=AMVHYLMIL  HLA=HLA-A01:01,HLA-A02:01 TPM=0.259926390800257 SOURCE=...ALIHFLMIL... MAX_BIT_DIST=0
AMVHYLMIL
```
After a successful run, you should be able check neoantigen prioritization results at: ${NeoOutDir}/${patientID}_prioritization_from_pmhcs.tsv

The MT tag specifies the mutant-type peptide and is required for each peptide record.
The WT tag specifies the wild-type peptide and is optional (if WT is omitted, then agretopicity of zero is generated).
All other tags (such as TPM denoting transcript-per-million) are also optional.
Each peptide sequence specified by the MT or WT tag must occurs exactly once in the sequences of the input peptide FASTA file.

If comma_sep_hla_list (a string of comma-separated HLA alleles) is not specified when running snakemake,
then the HLA key=value pair must be specified in the ${tumorSpecificPeptideFasta} file.
Otherwise,
the HLA key=value pair can be omitted in the ${tumorSpecificPeptideFasta} file.

### How to use any feature (e.g., neoepitope feature) TSV file as input

Please run ```neopredictor.py --help``` to see how to run with an example-by-feature table containing numerical values as input. 
In brief, neopredictor.py perform training, testing, or both. 
The option --train specifies the TSV file containing training data. 
The option --test specifies the TSV file containing test data. 
The option --feature-sets specifies the set of columns of the training and/or test TSV files. 

The python script ```neopredictor.py``` can take any TSV file containing numerical values as input, so the output of tools such as pVAC-seq (which is tabular) can be the input to this script. 

### How to use this repository as a scikit-learn machine-learning library

The novel feature transformation is implemented in the IsotonicLogisitcRegression.py file. 
The IsotonicLogisitcRegression class in this file implements the fit, transform, fit_transform, predict, and predict_proba methods. 
Hence, this file conforms to the API specification of scikit-learn estimators and can be used as a general-purpose machine learning library. 

### Description of the columns generates by the pipeline by default

Here is a list of imprtant columns that are generated by the pipeline when taking FASTQ or FASTA files as input (```neopredictor.py``` and the relevant methods in IsotonicLogisitcRegression can take any table with any set of columns as input).

MT_pep: mutant peptide, peptide generated by a mutation (can also be an exogenous peptide such as a viral peptide). 

ST_pep: self peptide, peptide that is similar to the mutant peptide. This peptide is often (but not always) the same as the wild-type peptide. 

WT_pep: wild-type peptide, the non-mutated peptide that generated the mutant peptide. This peptide is often (but not always) the same as the self peptide.

HLA_type: MHC allotype that is paired with the peptide. 

MT_BindAff: mutant-peptide binding affiniy (IC50 in nanomolar, ranging from 0 to infinity) to the MHC molecules, estimated by NetMHCpan 4.1. 

%Rank_EL: the eluted-ligand likelihood rank (percentile ranging from 0 to 1, lower corresponds to more likelihood) of the mutant-peptide for being eluted with 
(i.e., bound to, acting as a ligand to) the MHC molecules, estimated by NetMHCpan 4.1.

BindStab: mutant-peptide binding stability (half life in hour, ranging from 0 to infinity) to the MHC molecules, estimated by NetMHCstabpan. 

Quantification: the abundance (transcript-per-million, TPM) of the gene that generated the mutant peptide, estimated by Kallisto from RNA-seq data.

Agretop: agretopicity (mutant-type binding affinity divided by the wild-type affinity, with both affinities estimated by NetMHCpan 4.1). 

PRIME_rank: the percentile of the PRIME-estimated immunogenicity of the mutant peptide in the MHC context (ranging from 0 to 1, lower means more immunogenic). 

PRIME_BArank: the percentile of the estimated binding affinity between the mutant peptide and the MHC molecules (ranging from 0 to 1, lower corresponds to higher affinity and lower IC50). 

mhcflurry_aff_percentile: the percentile of the affinity between the mutant peptide and the MHC molecules (ranging from 0 to 1, lower corresponds to higher affinity and lower IC50). 

mhcflurry_presentation_percentile: the percentile of the probability that the peptide is presented by the MHC molecules (ranging from 0 to 1, lower means more immunogenic). 

ln_NumTested: the number of all mutant peptides originated from the same patient (from which this peptide originated) that were or willb be tested for immunogenicity. 

ET_pep : heteroclitic peptide of the mutant peptide (a peptide that is further mutated for therapeutic purpose). This is experimental, so please ignore this for now. 

### Additional information

If the trained model in Python pickle format is not found in the model directory, then training is done automatically by snakemake.
Because training is automated, there is no need to manually train the model for a typical use case.

For advanced usage of the Snakefile, please refer to https://snakemake.readthedocs.io/en/stable/

## Trouble-shooting

If you have encountered an error with conda, please show us the full error log generated by conda.
If you have encountered an error with snakemake, please show us the full stdout and stderr generated by snakemake.
Most of the third-party tools that this pipeline uses were not developed by us,
so we are not guaranteed to find the root causes of the errors that are specific to these tools.

