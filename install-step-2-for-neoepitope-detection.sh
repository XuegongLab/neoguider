#!/usr/bin/env bash

rootdir="$(dirname -- "$0";)"
rootdir="$(cd "$rootdir"; pwd;)"

neoguider=$2 # neoguider env name
if [ -z "$neoguider" ]; then neoguider=ng; fi

###
### Tools and database for detecting neoepitope candidates from sequencing data
###

###
### (1) software download
###

VEP_version=$(conda list --name $neoguider | grep "^ensembl-vep" | awk '{print $2}' | awk -F. '{print $1}') # 109 #"105"

mkdir -p ${rootdir}/software/asn
cd ${rootdir}/software/asn
git clone https://github.com/bm2-lab/ASNEO.git || true
pushd ASNEO && git checkout 9f43cff && popd
tar -xvf ASNEO/src/software.tar.gz
mv ASNEO/* ../ASNEO

cd       ${rootdir}/software 

# IMPORTNT-NOTE: UVC (along with UVC-delins) is free for non-commercial use only. For commercial use, please contact Genetron Health
export C_INCLUDE_PATH="${C_INCLUDE_PATH}:${CONDA_PREFIX}/include"

if [ $(echo "$1" | grep -cP "skip-uvc|skip-all-software") -eq 0 ]; then
    mv uvc uvc.bak || true
    git clone https://gitlab.com/cndfeifei/uvc.git || true
    pushd uvc && git checkout a01fb6f669b077f59d39199af48ebbbf88f8d5e4
    # rename libbz2.so into something else if an error pops up from the make command
    # older versions of C++ compilers do not support address sanitizors (ASAN) that are used for debugging
    #   in this case, please just skip ASAN by running make deploy manually.
    ./install-dependencies.sh && make -j 6 && make deploy 
    cp bin/uvc* "${CONDA_PREFIX}/bin/" || true
    popd

    mv uvc-delins uvc-delins.bak || true
    git clone https://gitlab.com/cndfeifei/uvc-delins.git || true
    pushd uvc-delins && git checkout 30af0939e3e9d4d492243ca38fe6cc14ac2c9ac7
    ./install-dependencies.sh && make -j 6 && make deploy
    cp bin/uvc* "${CONDA_PREFIX}/bin/" || true
    popd
fi

### IMPORTNT-NOTE: MixCR is free for non-commercial use only. For commercial use, please contact MiLaboratories Inc
# You have to activate MixCR with a license obtained from https://licensing.milaboratories.com/ in order to use it
if [ $(echo "$1" | grep -cP "skip-mixcr|skip-all-software") -eq 0 ]; then
    wget --no-check-certificate -c https://github.com/milaboratory/mixcr/releases/download/v4.0.0/mixcr-4.0.0.zip
    unzip mixcr-4.0.0.zip
fi
if [ $(echo "$1" | grep -cP "skip-ergo|skip-all-software") -eq 0 ]; then
    mv ERGO-II ERGO-II.bak || true
    git clone https://github.com/IdoSpringer/ERGO-II.git || true
    pushd ERGO-II && git checkout 85d320a && popd
    sed -i "s;ae_dir = 'TCR_Autoencoder';ae_dir = 'Models/AE' # CHANGED_FROM ae_dir = 'TCR_Autoencoder';g" ERGO-II/Models.py 
    sed -i "s;checkpoint = torch.load(ae_file);checkpoint = torch.load(ae_file, map_location='cuda:0') # CHANGED_FROM checkpoint = torch.load(ae_file);g" ERGO-II/Models.py 
    sed -i "s;from pytorch_lightning.logging import TensorBoardLogger;from pytorch_lightning.loggers import TensorBoardLogger # CHANGED_FROM from pytorch_lightning.logging import TensorBoardLogger;g" ERGO-II/Trainer.py
    sed -i "s;self.hparams = hparams;self.save_hyperparameters(hparams) # CHANGED_FROM self.hparams = hparams;g" ERGO-II/Trainer.py
    sed -i 's;@pl.data_loader;#@pl.data_loader # CHANGED_FROM @pl.data_loader;g' ERGO-II/Trainer.py
    sed -i "s;# df.to_csv('results.csv', index=False);df.to_csv(sys.argv[3], index=False) # df.to_csv('results.csv', index=False);g" ERGO-II/Predict.py
fi

if [ $(echo "$1" | grep -cP "skip-mutect2|skip-all-software") -eq 0 ]; then
    wget --no-check-certificate https://github.com/broadinstitute/gatk/releases/download/4.3.0.0/gatk-4.3.0.0.zip
    unzip gatk-4.3.0.0.zip
fi

#wget --no-check-certificate -c https://snpeff.blob.core.windows.net/versions/snpEff_latest_core.zip
#unzip snpEff_latest_core.zip
#git clone https://github.com/XuegongLab/NeoHunter.git && cd NeoHunter

###
### (2) database download
###

mkdir -p ${rootdir}/database
cd       ${rootdir}/database

wget --no-check-certificate -c http://ftp.ensembl.org/pub/grch37/release-${VEP_version}/variation/vep/homo_sapiens_vep_${VEP_version}_GRCh37.tar.gz
tar xvzf homo_sapiens_vep_${VEP_version}_GRCh37.tar.gz
wget --no-check-certificate -c http://ftp.ensembl.org/pub/grch37/release-${VEP_version}/fasta/homo_sapiens/cdna/Homo_sapiens.GRCh37.cdna.all.fa.gz
gunzip -cf Homo_sapiens.GRCh37.cdna.all.fa.gz > Homo_sapiens.GRCh37.pep.all.fa

wget --no-check-certificate -c https://data.broadinstitute.org/Trinity/CTAT_RESOURCE_LIB/__genome_libs_StarFv1.10/GRCh37_gencode_v19_CTAT_lib_Mar012021.plug-n-play.tar.gz
tar xvzf GRCh37_gencode_v19_CTAT_lib_Mar012021.plug-n-play.tar.gz

###
### (3) database index construction
###

bwa index $(dirname $(which OptiTypePipeline.py))/data/hla_reference_rna.fasta

bedtools sort \
    -faidx GRCh37_gencode_v19_CTAT_lib_Mar012021.plug-n-play/ctat_genome_lib_build_dir/ref_genome.fa.fai \
    -i     GRCh37_gencode_v19_CTAT_lib_Mar012021.plug-n-play/ctat_genome_lib_build_dir/ref_annot.gtf.mini.sortu | bedtools merge -i - \
    >      GRCh37_gencode_v19_CTAT_lib_Mar012021.plug-n-play/ctat_genome_lib_build_dir/ref_annot.gtf.mini.sortu.bed

bwa index GRCh37_gencode_v19_CTAT_lib_Mar012021.plug-n-play/ctat_genome_lib_build_dir/ref_genome.fa
# kallisto index -i  GRCh37_gencode_v19_CTAT_lib_Mar012021.plug-n-play/ctat_genome_lib_build_dir//ref_annot.cdna.fa.kallisto-idx GRCh37_gencode_v19_CTAT_lib_Mar012021.plug-n-play/ctat_genome_lib_build_dir//ref_annot.cdna.fa
kallisto index -i Homo_sapiens.GRCh37.cdna.all.fa.kallisto-idx Homo_sapiens.GRCh37.cdna.all.fa

###
### (4) other database download for backward compatibility
###

wget --no-check-certificate -c https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/hg19.fa.gz
gunzip -cf hg19.fa.gz > hg19.fa

wget --no-check-certificate -c http://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/genes/hg19.refGene.gtf.gz
gunzip -cf hg19.refGene.gtf.gz > hg19.refGene.gtf

#bwa index hg19.fa
#samtools faidx hg19.fa && samtools dict hg19.fa > hg19.dict

#wget --no-check-certificate -c ftp://gsapubftp-anonymous@ftp.broadinstitute.org:21/bundle/hg19/1000G_phase1.indels.hg19.sites.vcf.gz
#wget --no-check-certificate -c ftp://gsapubftp-anonymous@ftp.broadinstitute.org:21/bundle/hg19/1000G_phase1.indels.hg19.sites.vcf.idx.gz
#wget --no-check-certificate -c ftp://gsapubftp-anonymous@ftp.broadinstitute.org:21/bundle/hg19/dbsnp_138.hg19.vcf.gz
#wget --no-check-certificate -c ftp://gsapubftp-anonymous@ftp.broadinstitute.org:21/bundle/hg19/dbsnp_138.hg19.vcf.idx.gz
#wget --no-check-certificate -c ftp://gsapubftp-anonymous@ftp.broadinstitute.org:21/bundle/hg19/Mills_and_1000G_gold_standard.indels.hg19.sites.vcf.gz
#wget --no-check-certificate -c ftp://gsapubftp-anonymous@ftp.broadinstitute.org:21/bundle/hg19/Mills_and_1000G_gold_standard.indels.hg19.sites.vcf.idx.gz

#mkdir -p vcfs
#for vcf in "1000G_phase1.indels.hg19.sites.vcf.gz" "dbsnp_138.hg19.vcf.gz" "Mills_and_1000G_gold_standard.indels.hg19.sites.vcf.gz" ; do
#        bcftools view -Oz -o vcfs/$vcf $vcf && bcftools index -f vcfs/$vcf
#        java -jar ../gatk-4.2.6.1/gatk-package-4.2.6.1-local.jar IndexFeatureFile -I vcfs/$vcf
#done

# download funcotator dataset
# wget --no-check-certificate -c https://console.cloud.google.com/storage/browser/_details/broad-public-datasets/funcotator/funcotator_dataSources.v1.6.20190124s.tar.gz
#wget --no-check-certificate -c ftp://gsapubftp-anonymous@ftp.broadinstitute.org/bundle/funcotator/funcotator_dataSources.v1.6.20190124s.tar.gz
#tar xzvf funcotator_dataSources.v1.6.20190124s.tar.gz

# download refseq annotation for hg19

