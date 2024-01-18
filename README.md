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
sh -evx install-step-1-by-conda.sh ng # install all dependencies by conda
conda run -n hhh sh install-step-2.sh # download databases and build database indexes
```
If the "$conda install" command in install-step-1-by-conda.sh ran into any error, you can try replacing this command by each of the following commands:
```
conda create     --name ng --file env/requirements.list_e_no_pypi.txt    # if this command does not work then try the next one below
conda env create --name ng --file env/freeze.env_export.yml              # if this command does not work then try the next one below
conda env create --name ng --file env/freeze.env_export_no_builds.yml    # if this command does not work then try the next one below
conda env create --name ng --file env/freeze.env_export_from_history.yml
```

Next, you have to manually set up netMHCpan, netMHCstabpan and MixCR (due to their licensing requirements).
Please refer to https://services.healthtech.dtu.dk/services/NetMHC-4.0/ and https://services.healthtech.dtu.dk/services/NetMHCstabpan-1.0/ for how to manually download, install and activate netMHCpan and netMHCstabpan.
After doing so, please set the full paths of netMHCpan and netMHCstabpan in the config.ini file accordingly.
Please obtain a license for MixCR from https://licensing.milaboratories.com/ and then run the command ```software/mixcr activate-license``` to activate MixCR. 

Last but no the least, please be aware that UVC, netMHCpan, netMHCstabpan and MixCR are free for academic use but require commercial licensing for for-profit use.

## How to run

### With FASTQ files as input

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

### With peptide FASTA file as input

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

### With neoantigen-feature TSV file as input

Please run ```neopredictor.py --help``` to see how to run with neoantigen features as input

### Additional information

If the trained model in Python pickle format is not found in the model directory, then training is done automatically by snakemake.
Because training is automated, there is no need to manually train the model for a typical use case.

For advanced usage of the Snakefile, please refer to https://snakemake.readthedocs.io/en/stable/

## Trouble-shooting

If you have encountered an error with conda, please show us the full error log generated by conda.
If you have encountered an error with snakemake, please show us the full stdout and stderr generated by snakemake.
Most of the third-party tools that this pipeline uses were not developed by us,
  so we are not guaranteed to find the root causes of the errors that are specific to these tools.

