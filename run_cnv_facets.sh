#  echo "(source $(conda info --base)/etc/profile.d/conda.sh ; set -evx ; conda activate cnv_facets && "

SNP_VCF=database/All_20180423.vcf.gz 
#All_20180423.vcf.gzAll_20180423.vcf.gz
cnv_facets.R -t $TBAM -n $NBAM -vcf ${SNP_VCF} -o $OUTDIR/cnv_facets/${SID} # " #step=1"
bcftools query "$OUTDIR/cnv_facets/${SID}.vcf.gz" -HH -f '%CHROM\t%POS\t%END\t%TCN_EM' > "${OUTDIR}/cnv_facets/${SID}.cnv_facets.segs.tsv"

