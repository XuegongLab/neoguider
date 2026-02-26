rootdir=$PWD

conda create -y -n neo_cnv_env

# install cnv_facets

conda run -n neo_cnv_env mamba install -y cnv_facets

# install_ginkgo

bioconda_packages="bioconductor-ctc bioconductor-dnacopy"
condaforge_packages="r-devtools r-inline r-gplots r-scales r-plyr r-ggplot2 r-gridExtra r-fastcluster r-heatmap3"
conda run -n neo_cnv_env mamba install $bioconda_packages $condaforge_packages

#for p in $bioconda_packages $condaforge_packages ; do
#    conda run -n neo_cnv_env mamba install -y $p
#done

cd ${rootdir}/software
git clone https://github.com/robertaboukhalil/ginkgo.git && pushd ginkgo && git checkout d7c7790 && make && popd
sed 's;library(gridExtra);library(gridExtra)\n\n#NOTE: The workaround https://github.com/ChristophH/gplots described at https://github.com/robertaboukhalil/ginkgo no longer works with R version 4.X.X\n#NOTE: Therefore, we fall back to the heatmap function\nheatmap.2 = function(...) { return(heatmap(...)) };g' ginkgo/scripts/process.R -i
sed 's;tar -czf;#NOTE: the tar command results in runtime error and is therefore commented out\n#tar -czf;g' ginkgo/cli/ginkgo.sh -i

# download cnv_facets data

cd ${rootdir}/database
wget -c https://ftp.ncbi.nih.gov/snp/organisms/human_9606_b151_GRCh37p13/VCF/GATK/All_20180423.vcf.gz
wget -c https://ftp.ncbi.nih.gov/snp/organisms/human_9606_b151_GRCh37p13/VCF/GATK/All_20180423.vcf.gz.tbi

# download ginkgo data

mkdir ${rootdir}/database/screfs && pushd ${rootdir}/database/screfs
wget https://labshare.cshl.edu/shares/schatzlab/www-data/ginkgo/genomes/hg19.tgz
tar -xvf hg19.tgz
rm -r ${rootdir}/software/ginkgo/genomes/hg19 || true && cp -rs ${rootdir}/database/screfs/hg19 ${rootdir}/software/ginkgo/genomes/ || true
popd

