# NeoGuider

NeoGuider (NG, ng) is a neoantigen prioritization pipeline and machine-learning algorithm.
NG is able to sensitively detect and accurately rank neoantigen candidates.
Additionally, NG is able to estimate the probability that each neoantigen candidate is immunogenic
  (i.e., tested and thus validated to be positive by an immnoassay such as MHC-I multimer or IFN-gamma ELISpot).

The neoguider software is released with the APACHE-II license.

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
After doing so, please set the full paths of netMHCpan and netMHCstabpan in the config.yaml file accordingly.

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

The python script ```neopredictor.py``` can take any TSV file containing numerical values as input, so the tabular output of tools such as pVAC-seq and LENS (documented at https://pvactools.readthedocs.io/en/latest/pvacseq/output_files.html and https://uselens.io/en/lens-v1.6/lens_report.html) can be the input to this script. 
Most tools generate a table as output, so this script is very generic and can complement many other such tools. 

### How to use this repository as a scikit-learn machine-learning library

The novel feature transformation is implemented in the IsotonicLogisitcRegression.py file. 
The IsotonicLogisitcRegression class in this file implements the fit, transform, fit_transform, predict, and predict_proba methods. 
Hence, this file conforms to the API specification of scikit-learn estimators and can be used as a general-purpose machine learning library. 

### Description of the columns generates by the pipeline by default

The columns are classified into two groups:

#### Here is a list of important columns that are generated by the pipeline when taking FASTQ or FASTA files as input (```neopredictor.py``` and the relevant methods in IsotonicLogisitcRegression can take any table with any set of columns as input).

- MT_pep: mutant peptide, peptide generated by a mutation (can also be an exogenous peptide such as a viral peptide). 

- ST_pep: self peptide, peptide that is similar to the mutant peptide. This peptide is often (but not always) the same as the wild-type peptide because ST_pep is the best match from the entire proteome according to blastp. 

- WT_pep: wild-type peptide, the non-mutated peptide that generated the mutant peptide. This peptide is often (but not always) the same as the self peptide.

- HLA_type: MHC allotype that is paired with the peptide. 

- MT_BindAff: mutant-peptide binding affiniy (IC50 in nanomolar, ranging from 0 to infinity) to the MHC molecules, estimated by NetMHCpan 4.1. 

- %Rank_EL: the eluted-ligand likelihood rank (percentile ranging from 0 to 1, lower corresponds to more likelihood) of the mutant-peptide for being eluted with 
(i.e., bound to, acting as a ligand to) the MHC molecules, estimated by NetMHCpan 4.1.

- BindStab: mutant-peptide binding stability (half life in hour, ranging from 0 to infinity) to the MHC molecules, estimated by NetMHCstabpan. 

- Quantification: the abundance (transcript-per-million, TPM) of the gene that generated the mutant peptide, estimated by Kallisto from RNA-seq data.

- Agretop: agretopicity (mutant-type binding affinity divided by the wild-type affinity, with both affinities estimated by NetMHCpan 4.1). 

- PRIME_rank: the percentile of the PRIME-estimated immunogenicity of the mutant peptide in the MHC context (ranging from 0 to 1, lower means more immunogenic). 

- PRIME_BArank: the percentile of the estimated binding affinity between the mutant peptide and the MHC molecules (ranging from 0 to 1, lower corresponds to higher affinity and lower IC50). 

- mhcflurry_aff_percentile: the percentile of the affinity between the mutant peptide and the MHC molecules (ranging from 0 to 1, lower corresponds to higher affinity and lower IC50). 

- mhcflurry_presentation_percentile: the percentile of the probability that the peptide is presented by the MHC molecules (ranging from 0 to 1, lower means more immunogenic). 

- ln_NumTested: the number of all mutant peptides originated from the same patient (from which this peptide originated) that were or willb be tested for immunogenicity. 

- ET_pep : heteroclitic peptide of the mutant peptide (a peptide that is further mutated for therapeutic purpose). This is experimental, so please ignore this for now. 

- PredictedProbability: the predicted probability of immunogenicity.

- Rank: the rank of the pMHC combination that represents the neoepitope candidate, where a lower rank indicates a higher priority. 

#### The columns that are less important are as follows.

- ST_BindAff: the binding affinity of ST_pep estimated by NetMHCpan 4.1

- WT_BindAff: the binding affinity of WT_pep estimated by NetMHCpan 4.1

- MT_ST_pairAln: the sequece alignment of MT_pep with ST_pep

- MT_WT_pairAln: the sequece alignment of MT_pep with WT_pep

- MT_ST_bitDist: the bit-score difference between MT_pep and ST_pep according to the BLOSOM-62 substitution matrix

- MT_WT_bitDist: the bit-score difference between MT_pep and WT_pep according to the BLOSOM-62 substitution matrix

- MT_ST_hamDist: the edit distance between MT_pep and ST_pep

- MT_WT_hamDist: the edit distance between MT_pep and WT_pep

- Identity: the FASTA header (ID) that corresponds to this peptide-MHC pair

- Quantification: the TPM of the transcript(s) that generated the mutant peptide MT_pep

- BindLevel, Core, Of, Gp, Gl, Ip, Il, Icore, Score_EL, %Rank_EL, Score_BA, %Rank_BA: the output of netMHCpan 4.1

- ST_Agretopicity: agretopicity with respect to ST_pep, computed as the binding-affinity ratio of the mutant-peptide MT_pep with respect to the self-peptide ST_pep

- Agretopicity: agretopicity with respect to WT_pep

- Foreignness: foreigness which is computed in the same ways as antigen.garnish

- DNA_QUAL: the variant quality (VCF QUAL column) of the DNA variant that generated the mutant peptide MT_pep, estimated from DNA-seq data

- DNA_refDP: the ref (Wild-type) alelle sequencing depth of the DNA variant from DNA-seq data

- DNA_altDP: the alt (Mutant) allele sequencing depth of the DNA variant from DNA-seq data

- RNA_QUAL: the variant quality (VCF QUAL column) of the DNA variant that generated the mutant peptide MT_pep, estimated from RNA-seq data

- RNA_refDP: the ref (Wild-type) alelle sequencing depth of the DNA variant from RNA-seq data

- RNA_altDP: the alt (Mutant) allele sequencing depth of the DNA variant from RNA-seq data

- IsFrameshift: boolean value indicating whether the variant causes a frameshift

- WT_Foreignness: the foreigness of the wild-type peptide WT_pep

- ForeignDiff: the foreigness of MT_pep with respect to WT_pep

- XSinfo, MTinfo, STinfo, WTinfo: binding score of random peptide, MT_pep, ST_pep, and WT_pep according to the HLA allotype HLA_type

- ln_NumTested: natural logarithm of the number of peptides tested for immunogenicity (e.g., by pMHC multimer staining or INF-gamme ELISpot) for the corresponding patient. 
  By default, we assume that all peptides were tested for immunogenicity if this info is not provided. 

- VALID_N_TESTED: the actual number of peptides tested for immunogenicity. This information is not available if no peptides were tested. 
 
- VALID_CUMSUM: the culumative number of peptide-MHC pairs that were both tested positive for immunogenicity and ranked without higher priority than this pair

- PROBA_CUMSUM: the cumulative sum of probabilities of the peptide-MHC pairs that were both tested positive for immunogenicity and ranked without higher priority than this pair

- ML_pipeline: the name of the machine-learning pipeline used to generate PredictedProbability
    
- PredictedProbWithOtherFeatureSet_1: the probability predicted by using the alternative set of features. 
  By default, this set is generated by dropping the feature ln_NumTested and keeping all other features. 

- SourceAlterationDetail: variant annotation from VEP

- PepTrace: describes how the peptide is generated (used for debugging purpose)


### Additional information

By default, the pipeline also uses star-fusion and ASNEO to detect fusion and splicing variants (as specified by alteration_type='snv,indel,fsv,fusion,splicing' in the config.yaml). 
However, the pipeline does not prioritize the detected fusion and splicing variants. 
To enabled the prioritization of fusion and splicing variants, please set rna_only_prio_alteration_type=fusion,splicing from either the Snakemake command line or from the config.yaml file. 
To disable the generation of fusion and splicing variants, please pass the params --config alteration_type='snv,indel,fsv' from the snakemake command or modify alteration_type to 'snv,indel,fsv,fusion,splicing' in the config.yaml file. 
Here, snv, indel, fsv, fusion and splicing denote single-nucleotide variant, insertion-deletion variant, frameshift variant (can be caused by either snv or indel), fusion variant, and splicing variant.

If the trained model in Python pickle format is not found in the model directory, then training is done automatically by snakemake. 
Because training is automated, there is no need to manually train the model for a typical use case. 

For advanced usage of the Snakefile, please refer to https://snakemake.readthedocs.io/en/stable/ (e.g., how to override the default option values in the config.yaml, how to limit the computational resources used per task, etc.).

By default, this pipeline also generates the heteroclitic peptides (aka e-mimotopes) of the input peptides of interest. 
But the generation of heteroclitic peptides consumes a lot of computational resources (e.g., CPU and RAM) and is currently in experimental stage. 
To prevent the generation of heteroclitic peptides, please either pass the param hetero\_editdist=0.5 to the --config arg of snakemake or set hetero\_editdist to 0.5 in the config.yaml file. 

## Trouble-shooting

If you have encountered an error with conda, please show us the full error log generated by conda.
If you have encountered an error with snakemake, please show us the full stdout and stderr generated by snakemake.
Most of the third-party tools that this pipeline uses were not developed by us,
so we are not guaranteed to find the root causes of the errors that are specific to these tools.

