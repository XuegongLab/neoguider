#!/usr/bin/env bash


rootdir="$(dirname -- "$0";)"
rootdir="$(cd "$rootdir"; pwd;)"

neoguider=$1 # neoguider env name
if [ -z "$neoguider" ]; then neoguider=ng; fi

###
### Tools and databases for extracting features from neoepitope candidates
###

###
### (1) software download
###

conda run -n $neoguider mhcflurry-downloads fetch || true # We can manually download the mhcflurry data if this command fails (e.g. due to problems with the connection to github)

# IMPORTNT-NOTE: netMHCpan and netMHCstabpan are free for non-commercial use only. For commercial use, please contact DTU Health Tech
# You have to go to the following three web-pages to manually download netMHCpan and netMHCstabpan and manually request for the licenses to use them
# The netMHCpan and netMHCstabpan software packages should be put under the software directory, as specified in the config.yml file.
# The download links are:
#   https://services.healthtech.dtu.dk/cgi-bin/sw_request?software=netMHCpan&version=4.1&packageversion=4.1b&platform=Linux # used alone
#   https://services.healthtech.dtu.dk/cgi-bin/sw_request?software=netMHCstabpan&version=1.0&packageversion=1.0a&platform=Linux
# BTW, the following lower version of netMHCpan is used with netMHCstabpan:
#   https://services.healthtech.dtu.dk/cgi-bin/sw_request?software=netMHCpan&version=2.8&packageversion=2.8a&platform=Linux

# IMPORTNT-NOTE: PRIME and MixMHCpred are free for non-commercial use only. For commercial use, please contact the GfellerLab
mkdir -p ${rootdir}/software/prime
pushd    ${rootdir}/software/prime
    git clone https://gitlab.com/cndfeifei/PRIME.git
    pushd PRIME && git checkout e798aad && popd
    g++ -O3 PRIME/lib/PRIME.cc -o PRIME/lib/PRIME.x
    git clone https://gitlab.com/cndfeifei/MixMHCpred.git
    pushd MixMHCpred && git checkout 0a7f9b9
        chmod +x MixMHCpred
        chmod +x install_packages && conda install mafft # ./install_packages # this command requires sudo, the corresponding non-sudo version is "$conda install mafft" (e.g., conda=mamba)
        # chmod +x setup_path && ./setup_path # recommended by the authors of MixMHCpred but not needed here
    popd
popd

###
### (2) database download
###

VEP_version=$(conda list --name $neoguider | grep "^ensembl-vep" | awk '{print $2}' | awk -F. '{print $1}') # 109 #"105"

cd ${rootdir}/database
wget -c http://ftp.ensembl.org/pub/grch37/release-${VEP_version}/fasta/homo_sapiens/pep/Homo_sapiens.GRCh37.pep.all.fa.gz
gunzip -fk Homo_sapiens.GRCh37.pep.all.fa.gz

wget http://mhcmotifatlas.org/data/classI/MS/Peptides/all_peptides.txt
python ../neomotif.py -i all_peptides.txt -o all_peptides -p Homo_sapiens.GRCh37.pep.all.fa # GRCh37_gencode_v19_CTAT_lib_Mar012021.plug-n-play/ctat_genome_lib_build_dir/ref_annot.pep

###
### (3) database index construction
###

for faa in Homo_sapiens.GRCh37.pep.all.fa iedb.fasta; do
    makeblastdb -in ${faa} -dbtype prot # protein database
    samtools faidx ${faa}
done

###
### (4) download of data that were pre-generated by this pipeline
###

# Zhao, Xiaofei (2025). MullerNCItrain_.train-data.min.tsv. figshare. Dataset. https://doi.org/10.6084/m9.figshare.28745798.v1
mkdir -p ${rootdir}/model && wget -c https://figshare.com/ndownloader/files/53476277 && cp 53476277 ${rootdir}/model/MullerNCItrain_.train-data.min.tsv

