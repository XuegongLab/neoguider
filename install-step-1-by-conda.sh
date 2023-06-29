#sudo apt install tcsh # is required by netMHCpan-4 /usr/bin/tcsh
# sshpass is required if we want to run netMHC command on a remote server

# note: you can use these mirror channels (i.e., with "-c $conda-forge -c $bioconda -c $pytorch") to speed up installation in China
condaforge="-c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/" # -c conda-forge
bioconda="-c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/bioconda/" # -c bioconda
pytorch="-c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/" # -c pytorch

conda=mamba
neohunter=nhh # neohunter

conda install -y mamba -n base
conda create -y -n $neohunter

# common bin, common lib, machine-learning lib, bioinformatics bin, bioinformatics lib 
# note: pyfasta is replaced by pyfaidx, and ASNEO requires 'biopython<=1.79' (ASNEO code can be refactored to upgrade biopython)
$conda install -y -n $neohunter python=3.10 \
    gcc openjdk parallel perl sshpass tcsh \
    perl-carp-assert psutil pyyaml requests-cache zlib \
    pandas pytorch pytorch-lightning=0.8 scikit-learn xgboost \
    bcftools blast bwa ensembl-vep gatk kallisto mosdepth optitype samtools snakemake star 'star-fusion>=1.11' \
    'biopython<=1.79' pybiomart pyfaidx pysam

pip install sj2psi # for ASNEO.py

