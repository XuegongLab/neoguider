#sudo apt install tcsh # is required by netMHCpan-4 /usr/bin/tcsh
# sshpass is required if we want to run netMHC command on a remote server

# note: you can use these mirror channels (i.e., with "-c $conda-forge -c $bioconda -c $pytorch") to speed up installation in China
condaforge="-c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/" # -c conda-forge
bioconda="-c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/bioconda/" # -c bioconda
pytorch="-c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/" # -c pytorch

conda=mamba
neoheadhunter=nhh # neoheadhunter

conda install -y mamba -n base
conda create -y -n $neoheadhunter

# common bin, common lib, machine-learning lib, bioinformatics bin, bioinformatics lib 
# note: 
#   pyfasta is replaced by pyfaidx
#   ASNEO requires 'biopython<=1.79' (ASNEO code can be refactored to upgrade biopython)
#   ERGO-II requires pytorch-lightning=0.8, but we will change a few lines of source code in ERGO-II 
#     in the next installation step to make it work with higher versions of pytorch-lightning
$conda install -y -n $neoheadhunter python=3.10 \
    gcc openjdk parallel perl sshpass tcsh \
    perl-carp-assert psutil pyyaml requests-cache zlib \
    pandas pytorch pytorch-lightning scikit-learn xgboost \
    bcftools blast bwa ensembl-vep gatk kallisto mosdepth optitype samtools snakemake star 'star-fusion>=1.11' \
    'biopython<=1.79' pybiomart pyfaidx pysam

conda run -n $neoheadhunter pip install sj2psi # for ASNEO.py

# The optitype environment provides a work-around for the issue at https://github.com/FRED-2/OptiType/issues/125
optitype=optitype_env
conda create -y -n $optitype
$conda install -y -n $optitype optitype=1.3.2 

# The following command can be run to generate the freeze and requirement files
# conda env export -n ${neoheadhunter} > ${neoheadhunter}.freeze.yml &&  conda list -e -n ${neoheadhunter} > ${neoheadhunter}.requirements.txt
# conda env export -n ${optitype} > ${optitype}.freeze.yml &&  conda list -e -n ${optitype} > ${optitype}.requirements.txt

