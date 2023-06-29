#!/usr/bin/env bash

rootdir="$(dirname -- "$0";)"
rootdir="$(cd "$rootdir"; pwd;)"

mkdir -p ${rootdir}/software && pushd ${rootdir}/software 

### IMPORTNT-NOTE: UVC (along with UVC-delins) is free for non-commercial use only. For commercial use, please contact Genetron Health
export C_INCLUDE_PATH="${C_INCLUDE_PATH}:${CONDA_PREFIX}/include"

if [$1 != "*skip-uvc*" ]; then
        git clone https://gitlab.com/cndfeifei/uvc.git
        pushd uvc
        ./install-dependencies.sh && make -j 6 && make deploy
        cp bin/uvc* $CONDA_PREFIX/bin/ || true
        popd

        git clone https://gitlab.com/cndfeifei/uvc-delins.git
        pushd uvc-delins
        ./install-dependencies.sh && make -j 6 && make deploy
        cp bin/uvc* $CONDA_PREFIX/bin/ || true
        popd
fi

### IMPORTNT-NOTE: MixCR is free for non-commercial use only. For commercial use, please contact MiLaboratories Inc
# You have to activate MixCR with a license obtained from https://licensing.milaboratories.com/ in order to use it
wget -c https://github.com/milaboratory/mixcr/releases/download/v4.0.0/mixcr-4.0.0.zip
unzip mixcr-4.0.0.zip

# IMPORTNT-NOTE: netMHCpan and netMHCstabpan are free for non-commercial use only. For commercial use, please contact DTU Health Tech
## You have to go to the following three web-pages to manually download netMHCpan and netMHCstabpan and manually request for the licenses to use them
## https://services.healthtech.dtu.dk/cgi-bin/sw_request?software=netMHCpan&version=4.1&packageversion=4.1b&platform=Linux # used alone
## https://services.healthtech.dtu.dk/cgi-bin/sw_request?software=netMHCstabpan&version=1.0&packageversion=1.0a&platform=Linux
## https://services.healthtech.dtu.dk/cgi-bin/sw_request?software=netMHCpan&version=2.8&packageversion=2.8a&platform=Linux # used with netMHCstabpan

#wget -c https://snpeff.blob.core.windows.net/versions/snpEff_latest_core.zip
#unzip snpEff_latest_core.zip
#git clone https://github.com/XuegongLab/NeoHunter.git && cd NeoHunter

mkdir -p ${rootdir}/database && pushd ${rootdir}/database

VEP_version=$(conda list | grep "^ensembl-vep" | awk '{print $2}' | awk -F. '{print $1}') # 109 #"105"

wget -c http://ftp.ensembl.org/pub/grch37/release-${VEP_version}/variation/vep/homo_sapiens_vep_${VEP_version}_GRCh37.tar.gz
tar xvzf homo_sapiens_vep_${VEP_version}_GRCh37.tar.gz
wget -c http://ftp.ensembl.org/pub/grch37/release-${VEP_version}/fasta/homo_sapiens/cdna/Homo_sapiens.GRCh37.cdna.all.fa.gz
gunzip -fk Homo_sapiens.GRCh37.cdna.all.fa.gz
wget -c http://ftp.ensembl.org/pub/grch37/release-${VEP_version}/fasta/homo_sapiens/pep/Homo_sapiens.GRCh37.pep.all.fa.gz
gunzip -fk Homo_sapiens.GRCh37.pep.all.fa.gz

wget -c https://data.broadinstitute.org/Trinity/CTAT_RESOURCE_LIB/__genome_libs_StarFv1.10/GRCh37_gencode_v19_CTAT_lib_Mar012021.plug-n-play.tar.gz
tar xvzf GRCh37_gencode_v19_CTAT_lib_Mar012021.plug-n-play.tar.gz

########## index construction ##########

bwa index $(dirname $(which OptiTypePipeline.py))/data/hla_reference_rna.fasta
makeblastdb -in iedb.fasta # in database
samtools faidx iedb.fasta

bedtools sort \
    -faidx GRCh37_gencode_v19_CTAT_lib_Mar012021.plug-n-play/ctat_genome_lib_build_dir/ref_genome.fa.fai \
    -i     GRCh37_gencode_v19_CTAT_lib_Mar012021.plug-n-play/ctat_genome_lib_build_dir/ref_annot.gtf.mini.sortu | bedtools merge -i - \
    >      GRCh37_gencode_v19_CTAT_lib_Mar012021.plug-n-play/ctat_genome_lib_build_dir/ref_annot.gtf.mini.sortu.bed

bwa index GRCh37_gencode_v19_CTAT_lib_Mar012021.plug-n-play/ctat_genome_lib_build_dir/ref_genome.fa
kallisto index -i  GRCh37_gencode_v19_CTAT_lib_Mar012021.plug-n-play/ctat_genome_lib_build_dir//ref_annot.cdna.fa.kallisto-idx GRCh37_gencode_v19_CTAT_lib_Mar012021.plug-n-play/ctat_genome_lib_build_dir//ref_annot.cdna.fa

########## other databases for backward compatibility ##########

wget -c https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/hg19.fa.gz
gunzip -fk hg19.fa.gz

#bwa index hg19.fa
#samtools faidx hg19.fa && samtools dict hg19.fa > hg19.dict

#wget -c ftp://gsapubftp-anonymous@ftp.broadinstitute.org:21/bundle/hg19/1000G_phase1.indels.hg19.sites.vcf.gz
#wget -c ftp://gsapubftp-anonymous@ftp.broadinstitute.org:21/bundle/hg19/1000G_phase1.indels.hg19.sites.vcf.idx.gz
#wget -c ftp://gsapubftp-anonymous@ftp.broadinstitute.org:21/bundle/hg19/dbsnp_138.hg19.vcf.gz
#wget -c ftp://gsapubftp-anonymous@ftp.broadinstitute.org:21/bundle/hg19/dbsnp_138.hg19.vcf.idx.gz
#wget -c ftp://gsapubftp-anonymous@ftp.broadinstitute.org:21/bundle/hg19/Mills_and_1000G_gold_standard.indels.hg19.sites.vcf.gz
#wget -c ftp://gsapubftp-anonymous@ftp.broadinstitute.org:21/bundle/hg19/Mills_and_1000G_gold_standard.indels.hg19.sites.vcf.idx.gz

#mkdir -p vcfs
#for vcf in "1000G_phase1.indels.hg19.sites.vcf.gz" "dbsnp_138.hg19.vcf.gz" "Mills_and_1000G_gold_standard.indels.hg19.sites.vcf.gz" ; do
#        bcftools view -Oz -o vcfs/$vcf $vcf && bcftools index -f vcfs/$vcf
#        java -jar ../gatk-4.2.6.1/gatk-package-4.2.6.1-local.jar IndexFeatureFile -I vcfs/$vcf
#done

# download funcotator dataset
# wget -c https://console.cloud.google.com/storage/browser/_details/broad-public-datasets/funcotator/funcotator_dataSources.v1.6.20190124s.tar.gz
#wget -c ftp://gsapubftp-anonymous@ftp.broadinstitute.org/bundle/funcotator/funcotator_dataSources.v1.6.20190124s.tar.gz
#tar xzvf funcotator_dataSources.v1.6.20190124s.tar.gz

# download refseq annotation for hg19
#wget -c http://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/genes/hg19.refGene.gtf.gz
#gunzip -fk hg19.refGene.gtf.gz

