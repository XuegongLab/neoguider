#!/usr/bin/env bash

rootdir="$(dirname -- "$0";)"
rootdir="$(cd "$rootdir"; pwd;)"

neoguider=$1 # neoguider env name
if [ -z "$neoguider" ]; then neoguider=ng; fi

#wget -c https://snpeff.blob.core.windows.net/versions/snpEff_latest_core.zip
#unzip snpEff_latest_core.zip
#git clone https://github.com/XuegongLab/NeoHunter.git && cd NeoHunter

mkdir -p ${rootdir}/database && pushd ${rootdir}/database

VEP_version=$(conda list --name $neoguider | grep "^ensembl-vep" | awk '{print $2}' | awk -F. '{print $1}') # 109 #"105"

wget -c http://ftp.ensembl.org/pub/release-109/variation/vep/mus_musculus_balbcj_vep_109_BALB_cJ_v1.tar.gz
tar xvzf mus_musculus_balbcj_vep_109_BALB_cJ_v1.tar.gz
wget -c http://ftp.ensembl.org/pub/release-109/fasta/mus_musculus_balbcj/cdna/Mus_musculus_balbcj.BALB_cJ_v1.cdna.all.fa.gz
gunzip -fk Mus_musculus_balbcj.BALB_cJ_v1.cdna.all.fa.gz
wget -c http://ftp.ensembl.org/pub/release-109/fasta/mus_musculus_balbcj/pep/Mus_musculus_balbcj.BALB_cJ_v1.pep.all.fa.gz
gunzip -fk Mus_musculus_balbcj.BALB_cJ_v1.pep.all.fa.gz

wget -c https://data.broadinstitute.org/Trinity/CTAT_RESOURCE_LIB/__genome_libs_StarFv1.10/Mouse_GRCm39_M31_CTAT_lib_Nov092022.plug-n-play.tar.gz
tar xvzf Mouse_GRCm39_M31_CTAT_lib_Nov092022.plug-n-play.tar.gz

########## index construction ##########

# Mouse MHC (H2) alleles are not typed because the H2 alleles of the lab mouse strains are well known from the literature.
#bwa index $(dirname $(which OptiTypePipeline.py))/data/hla_reference_rna.fasta

for faa in Mus_musculus_balbcj.BALB_cJ_v1.pep.all.fa ; do # iedb.fasta; do # iedb.fasta is not used in the end
    makeblastdb -in ${faa} -dbtype prot # in database
    samtools faidx ${faa}
done

bedtools sort \
    -faidx Mouse_GRCm39_M31_CTAT_lib_Nov092022.plug-n-play/ctat_genome_lib_build_dir/ref_genome.fa.fai \
    -i     Mouse_GRCm39_M31_CTAT_lib_Nov092022.plug-n-play/ctat_genome_lib_build_dir/ref_annot.gtf.mini.sortu | bedtools merge -i - \
    >      Mouse_GRCm39_M31_CTAT_lib_Nov092022.plug-n-play/ctat_genome_lib_build_dir/ref_annot.gtf.mini.sortu.bed

bwa index Mouse_GRCm39_M31_CTAT_lib_Nov092022.plug-n-play/ctat_genome_lib_build_dir/ref_genome.fa
# kallisto index -i  GRCh37_gencode_v19_CTAT_lib_Mar012021.plug-n-play/ctat_genome_lib_build_dir//ref_annot.cdna.fa.kallisto-idx GRCh37_gencode_v19_CTAT_lib_Mar012021.plug-n-play/ctat_genome_lib_build_dir//ref_annot.cdna.fa
kallisto index -i Mus_musculus_balbcj.BALB_cJ_v1.cdna.all.fa.kallisto-idx Mus_musculus_balbcj.BALB_cJ_v1.cdna.all.fa

########## other databases for backward compatibility ##########

wget -c https://hgdownload.soe.ucsc.edu/goldenPath/mm39/bigZips/mm39.fa.gz
gunzip -fk mm39.fa.gz

wget -c https://hgdownload.soe.ucsc.edu/goldenPath/mm39/bigZips/genes/refGene.gtf.gz
zcat refGene.gtf.gz > mm39.refGene.gtf

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

