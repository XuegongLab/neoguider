# note: the line below (which installs the tcsh shell) is part of the manual install of netMHCpan and netMHCstabpan, so the line below is commented out
#   sudo apt install tcsh # is required by netMHCpan-4 /usr/bin/tcsh
# note: sshpass is required if we want to run netMHCstabpan command on a remote server
# note: these mirror channels (for example, used with "-c $condaforge -c $bioconda" on the cmd-line) are used to speed up installation in China
#   you can set these mirror channels according to your geolocation
condaforge='https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/'
bioconda='https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/bioconda/'
main='https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/'
free='https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/'
fastai='https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/fastai/'
pytorch='https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/'

for channel in $pytorch $fastai $free $main $bioconda $condaforge ; do
    conda config --add channels "${channel}"
done

conda=mamba
neoguider="$1" # neoguider env name
if [ -z "$neoguider" ]; then neoguider=ng; fi

conda install -y mamba -n base
conda create -y -n $neoguider

# Order of packages: common bin, common lib, machine-learning lib, bioinformatics bin, bioinformatics lib 
# note:
#   pyfasta is replaced by pyfaidx
#   ASNEO requires 'biopython<=1.79' (ASNEO code can be refactored to upgrade biopython)
#   ERGO-II requires pytorch-lightning=0.8, but we will change a few lines of source code in ERGO-II
#     in the next installation step to make it work with higher versions of pytorch-lightning
#   podman can be used to provide a work-around for https://github.com/FRED-2/OptiType/issues/125
# Please feel free to modify other version sets as needed. 
$conda install -y -n $neoguider python=3.10 xlrd openpyxl \
    gcc openjdk parallel perl podman screen sshpass tcsh \
    perl-carp-assert psutil pyyaml requests-cache zlib \
    joblib numpy scipy pandas pytorch pytorch-lightning scikit-learn xgboost \
    bcftools blast 'bwa=0.7.17' 'ensembl-vep=109.3' 'kallisto=0.48.0' mosdepth 'optitype=1.3.5' samtools snakemake 'star=2.7.8' 'star-fusion=1.12.0' mhcflurry \
    'biopython<=1.79' pybiomart pyfaidx pysam
    # bcftools blast bwa ensembl-vep kallisto mosdepth optitype samtools snakemake star 'star-fusion>=1.11' \

# $conda run -n $neoguider mhcflurry-downloads fetch # We can manually download the mhcflurry data if this command fails

# Currently, edit-distance based on blosum alignment instead of edit-distance based on edlib is used
# note: if you have encountered the error: *** is not installable because it requires __cuda, which is missing on the system,
#   then you can refer to the work-around at
#   https://stackoverflow.com/questions/74836151/nothing-provides-cuda-needed-by-tensorflow-2-10-0-cuda112py310he87a039-0 
#   to solve this error (namely, export CONDA_OVERRIDE_CUDA="11.8" && export CONDA_CUDA_OVERRIDE="11.8").

$conda run -n $neoguider $conda update numexpr # fix warning from pandas

conda run -n $neoguider pip install sj2psi # for ASNEO.py

# > First work-around for https://github.com/FRED-2/OptiType/issues/125 : use podman to run OptiType
#   Please set IS_PODMAN_USED_TO_WORKAROUND_OPTITYPE_MEM_LEAK=True in the Snakefile to enable this work-around.
#   On some platforms, the error '''Error: command required for rootless mode with multiple IDs: exec: "newuidmap": executable file not found in $PATH''' pop up for this work-around. 
conda run -n $neoguider podman pull quay.io/biocontainers/optitype:1.3.2--py27_3 || true # Docker runs into error here

# > Second work-around for https://github.com/FRED-2/OptiType/issues/125 : use another conda env to run OptiType
#   Please set OTITYPE_CONDA_ENV=optitype_env in the Snakefile to enable this work-around.
#   On some platforms, conda and mamba cannot install the obsolete python versions that the previous versions of optitype depend on
optitype=optitype_env
conda create -y -n $optitype
$conda install -y -n $optitype optitype=1.3.2
conda run -n ${optitype} pip install --upgrade pyomo # pyomo=5.7.3

# > Third work-around for https://github.com/FRED-2/OptiType/issues/125 : manually set the comma_sep_hla_list config param to skip HLA typing. 

# The following commands can generate the requirements and freeze files
if false; then
    conda env export -n ${optitype} > env/${optitype}.freeze.env_export.yml &&  conda list -e -n ${optitype} > env/${optitype}.requirements.list_e.txt
    conda list       -n ${neoguider} -e | grep -v "^sj2psi=" > env/requirements.list_e_no_pypi.txt
    conda env export -n ${neoguider}                         > env/freeze.env_export.yml
    conda env export -n ${neoguider} --no-builds             > env/freeze.env_export_no_builds.yml
    conda env export -n ${neoguider} --from-history          > env/freeze.env_export_from_history.yml
fi

