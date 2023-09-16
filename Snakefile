import csv,logging,multiprocessing,os,subprocess,datetime
script_start_datetime = datetime.datetime.now()
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

script_basedir = F'{workflow.basedir}' # os.path.dirname(os.path.abspath(sys.argv[0]))
logging.debug(F'source_code_root_dir = {script_basedir}')

######### Setup #########

### Section 1: required input and output ###
DNA_TUMOR_FQ1 = config['dna_tumor_fq1']
DNA_TUMOR_FQ2 = config['dna_tumor_fq2']
DNA_NORMAL_FQ1 = config['dna_normal_fq1']
DNA_NORMAL_FQ2 = config['dna_normal_fq2']
RNA_TUMOR_FQ1 = config['rna_tumor_fq1']
RNA_TUMOR_FQ2 = config['rna_tumor_fq2']

RES = config['res'] # output result directory path
PREFIX = config['prefix']

### Section 2: parameters having no default values ###
netmhcpan_cmd = config['netmhcpan_cmd']
netmhcstabpan_cmd = config['netmhcstabpan_cmd']

### Section 3: parameters having some default values ###

tumor_depth = config['tumor_depth']
tumor_vaf = config['tumor_vaf']
normal_vaf = config['normal_vaf']
tumor_normal_var_qual = config['tumor_normal_var_qual']

binding_affinity_filt_thres = config['binding_affinity_filt_thres']
binding_affinity_hard_thres = config['binding_affinity_hard_thres']
binding_affinity_soft_thres = config['binding_affinity_soft_thres']

binding_stability_filt_thres = config['binding_stability_filt_thres']
binding_stability_hard_thres = config['binding_stability_hard_thres']
binding_stability_soft_thres = config['binding_stability_soft_thres']

tumor_abundance_filt_thres = config['tumor_abundance_filt_thres']
tumor_abundance_hard_thres = config['tumor_abundance_hard_thres']
tumor_abundance_soft_thres = config['tumor_abundance_soft_thres']

agretopicity_thres = config['agretopicity_thres']
foreignness_thres = config['foreignness_thres']
alteration_type = config['alteration_type']

#num_cores = config['num_cores']
netmhc_ncores = config['netmhc_ncores']
netmhc_nthreads = config['netmhc_nthreads']
ergo2_nthreads = config['ergo2_nthreads']

### Section 4: parameters having some default values of relative paths
HLA_REF = config.get('hla_ref', subprocess.check_output('printf $(dirname $(which OptiTypePipeline.py))/data/hla_reference_rna.fasta', shell=True).decode(sys.stdout.encoding))
CDNA_REF = config.get('cdna_ref', F'{script_basedir}/database/Homo_sapiens.GRCh37.cdna.all.fa') # .kallisto-idx
PEP_REF = config.get('pep_ref', F'{script_basedir}/database/Homo_sapiens.GRCh37.pep.all.fa')
VEP_CACHE = config.get('vep_cache', F'{script_basedir}/database')
CTAT = config.get('ctat', F'{script_basedir}/database/GRCh37_gencode_v19_CTAT_lib_Mar012021.plug-n-play/ctat_genome_lib_build_dir/')
MIXCR_PATH = config.get('mixcr_path',  F'{script_basedir}/software/mixcr.jar')
ERGO_EXE_DIR = config.get('ergo_exe_dir', F'{script_basedir}/software/ERGO-II')
ASNEO_REF = config.get('asneo_ref', F'{script_basedir}/database/hg19.fa')          # The {CTAT} REF does not work with ASNEO
ASNEO_GTF = config.get('asneo_gtf', F'{script_basedir}/database/hg19.refGene.gtf') # The {CTAT} gtf does not work with ASNEO

### Section 5: parameters having some default values depending on other values
REF = config.get('REF', F'{CTAT}/ref_genome.fa')
ERGO_PATH = config.get('ERGO_PATH', F'{ERGO_EXE_DIR}/Predict.py')
CDNA_KALLISTO_IDX = config.get('cdna_kallisto_idx', F'{CDNA_REF}.kallisto-idx')

tmpdirID = '.'.join([script_start_datetime.strftime('%Y-%m-%d_%H-%M-%S'), str(os.getpid()), PREFIX])
fifo_path_prefix = os.path.sep.join([config['fifo_dir'], tmpdirID])
logging.debug(F'fifo_path_prefix={fifo_path_prefix}')

variantcaller = config.get('variantcaller', 'uvc') # uvc or mutect2, please be aware that the performance of mutect2 is not evaluated. 

### Section 6: parameters that were empirically determined to take advantage of multi-threading efficiently

samtools_nthreads = 3
bcftools_nthreads = 3
bwa_nthreads = max((workflow.cores // 4, 1))
star_nthreads = max((workflow.cores // 4, 1))
uvc_nthreads = max((workflow.cores // 3, 1))
uvc_nthreads_on_cmdline = max((workflow.cores // 2, 1)) # over-allocate threads because each thread does not use 100% CPU and uses very little RAM
vep_nthreads = 4 # https://useast.ensembl.org/info/docs/tools/vep/script/vep_other.html : We recommend using 4 forks
mixcr_nthreads = max((workflow.cores // 2, 1))

# We used the bwa and STAR RAM usage from https://link.springer.com/article/10.1007/s00521-021-06188-z/figures/3
kallisto_mem_mb = 4000 # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7202009/
optitype_mem_mb = 40*1000 # bmcgenomics.biomedcentral.com/articles/10.1186/s12864-023-09351-z
samtools_sort_mem_mb = 768 * (samtools_nthreads + 1)
bwa_mem_mb = 9000
star_mem_mb = 32*1000
uvc_mem_mb = uvc_nthreads_on_cmdline * 1536
vep_mem_mb = vep_nthreads * 4000
mixcr_mem_mb = 32*1000
bwa_samtools_mem_mb = bwa_mem_mb + samtools_sort_mem_mb

### usually you should not modify the code below (please think twice before doing so) ###
IS_PODMAN_USED_TO_WORKAROUND_OPTITYPE_MEM_LEAK = True
OPTITYPE_CONFIG = F"{script_basedir}/software/optitype.config.ini"

def call_with_infolog(cmd, in_shell = True):
    logging.info(cmd)
    subprocess.call(cmd, shell = in_shell)
def make_dummy_files(files, origfile = F'{script_basedir}/placeholders/EmptyFile'):
    logging.info(F'Trying to create the dummy placeholder files {files}')
    for file in files:
        if config.get('refresh_placeholders', 0) == 1 or not os.path.exists(file):
            if os.path.exists(file): call_with_infolog(F'rm {file}')
            logging.info(F'Making directory of the file {file}')
            os.makedirs(os.path.dirname(file), exist_ok=True)
            call_with_infolog(F'cp {origfile} {file} && touch {file} && touch {file}.DummyPlaceholder')

DNA_TUMOR_ISPE  = (DNA_TUMOR_FQ2  not in [None, '', 'NA', 'Na', 'None', 'none', '.'])
DNA_NORMAL_ISPE = (DNA_NORMAL_FQ2 not in [None, '', 'NA', 'Na', 'None', 'none', '.'])
RNA_TUMOR_ISPE  = (RNA_TUMOR_FQ2  not in [None, '', 'NA', 'Na', 'None', 'none', '.'])

isRNAskipped = (RNA_TUMOR_FQ1 in [None, '', 'NA', 'Na', 'None', 'none', '.'])
if isRNAskipped and not ('comma_sep_hla_list' in config):
    logging.warning(F'Usually, either RNA FASTQ files (rna_tumor_fq1 and rna_tumor_fq2) or comma_sep_hla_list is specified in the config. ')

hla_typing_dir = F'{RES}/hla_typing'
info_dir = F'{RES}/info'
peptide_dir = F'{RES}/peptides'
alignment_dir = F'{RES}/alignments'
snvindel_dir = F'{RES}/snvindels'
pmhc_dir = F'{RES}/pmhcs'
prioritization_dir = F'{RES}/prioritization'

DNA_PREFIX = F'{PREFIX}_DNA'
RNA_PREFIX = F'{PREFIX}_RNA'
dna_snvindel_info_file = F'{info_dir}/{DNA_PREFIX}_snv_indel.annotation.tsv'
rna_snvindel_info_file = F'{info_dir}/{RNA_PREFIX}_snv_indel.annotation.tsv'

fusion_info_file = F'{info_dir}/{PREFIX}_fusion.tsv'
splicing_info_file = F'{info_dir}/{PREFIX}_splicing.tsv' # TODO: check if it makese sense to add_detail_info to TCR-unaware prioritization

neoheadhunter_prioritization_tsv = F'{prioritization_dir}/{PREFIX}_neoantigen_rank_neoheadhunter.tsv'
pmhc_as_input_prioritization_tsv = F'{prioritization_dir}/{PREFIX}_neoantigen_rank_pmhc_as_input.tsv' # prioritization from pMHC candidate summary info

tcr_specificity_result = F'{prioritization_dir}/{PREFIX}_neoantigen_rank_tcr_specificity_with_detail.tsv'

final_pipeline_out = F'{RES}/{PREFIX}_final_neoheadhunter_neoantigens.tsv'

rule all:
    input: neoheadhunter_prioritization_tsv, tcr_specificity_result, final_pipeline_out

# Note: the combination of pandas' reindex and optitype v1.3.5 results in memory leak: 
#   https://github.com/FRED-2/OptiType/issues/125
# Therefore, we used the podman container as a workaround for the memory leak.
# If the memory leak is fixed or if you have enough RAM, 
#   then you can ignore this workaround.
hla_fq_r1_fname = F'{PREFIX}.rna_hla_r1.fastq.gz'
hla_fq_r2_fname = F'{PREFIX}.rna_hla_r2.fastq.gz'
hla_fq_se_fname = F'{PREFIX}.rna_hla_se.fastq.gz'

hla_fq_r1 = F'{hla_typing_dir}/{hla_fq_r1_fname}'
hla_fq_r2 = F'{hla_typing_dir}/{hla_fq_r2_fname}'
hla_fq_se = F'{hla_typing_dir}/{hla_fq_se_fname}'

hla_bam   = F'{hla_typing_dir}/{PREFIX}.rna_hla_typing.bam'
hla_out   = F'{hla_typing_dir}/{PREFIX}_hlatype.tsv'

if 'comma_sep_hla_list' in config:
    os.makedirs(os.path.dirname(hla_out), exist_ok=True)
    with open(hla_out, 'w') as f: f.write('NotUsed')
elif os.path.exists(hla_out):
    with open(hla_out) as f: 
        lines = f.readlines()
        if lines and lines[0].strip() == 'NotUsed': os.system(F'rm {hla_out}')

logging.debug(F'HLA_REF = {HLA_REF}')

rule RNA_tumor_HLA_typing_preparation:
    output: hla_bam, hla_fq_r1, hla_fq_r2, hla_fq_se
    resources: mem_mb = bwa_mem_mb
    threads: bwa_nthreads
    # Note: razers3 is too memory intensive, so bwa mem is used instead of the command
    # (razers3 --percent-identity 90 --max-hits 1 --distance-range 0 --output {hla_bam} {HLA_REF} {RNA_TUMOR_FQ1} {RNA_TUMOR_FQ2})
    shell : '''
        bwa mem -t {bwa_nthreads} {HLA_REF} {RNA_TUMOR_FQ1} {RNA_TUMOR_FQ2} | samtools view -@ {samtools_nthreads} -bh -F4 -o {hla_bam}
        samtools fastq -@ {samtools_nthreads} {hla_bam} -1 {hla_fq_r1} -2 {hla_fq_r2} -s {hla_fq_se} '''
if 'comma_sep_hla_list' in config: make_dummy_files([hla_out])
rule HLA_typing:
    input: hla_fq_r1, hla_fq_r2, hla_fq_se
    output: out = hla_out
    resources: mem_mb = optitype_mem_mb
    threads: 1
    run:
        shell('rm -r {RES}/hla_typing/optitype_out/ || true && mkdir -p {RES}/hla_typing/optitype_out && cp {OPTITYPE_CONFIG} {RES}/hla_typing/config.ini')
        if RNA_TUMOR_ISPE:
            if IS_PODMAN_USED_TO_WORKAROUND_OPTITYPE_MEM_LEAK:
                shell('podman run -v {hla_typing_dir}:/data/ -t quay.io/biocontainers/optitype:1.3.2--py27_3 /usr/local/bin/OptiTypePipeline.py'
                      ' -c /data/config.ini -i /data/{hla_fq_r1_fname} /data/{hla_fq_r2_fname} --rna -o /data/optitype_out/ > {hla_out}.OptiType-container.stdout')
            else: shell('OptiTypePipeline.py -i {hla_fq_r1} {hla_fq_r2} --rna -o {RES}/hla_typing/optitype_out/ > {output.out}.OptiType.stdout')
        else:
            if IS_PODMAN_USED_TO_WORKAROUND_OPTITYPE_MEM_LEAK:
                shell('podman run -v {hla_typing_dir}:/data/ -t quay.io/biocontainers/optitype:1.3.2--py27_3 /usr/local/bin/OptiTypePipeline.py'
                      ' -c /data/config.ini -i /data/{hla_fq_se_fname}                         --rna -o /data/optitype_out/ > {hla_out}.OptiType-container.stdout')
            else: shell('OptiTypePipeline.py -i {hla_fq_se} --rna -o {RES}/hla_typing/optitype_out/ > {output.out}.OptiType.stdout')
        shell('cp {RES}/hla_typing/optitype_out/*/*_result.tsv {output.out}')
    
kallisto_out = F'{RES}/rna_quantification/{PREFIX}_kallisto_out'
outf_rna_quantification = F'{RES}/rna_quantification/abundance.tsv'
if 'outf_rna_quantification' in config:
    outf_rna_quantification = config['outf_rna_quantification']
elif isRNAskipped:
    make_dummy_files([outf_rna_quantification], F'{script_basedir}/placeholders/HighAbundance.tsv')
rule RNA_tumor_transcript_abundance_estimation:
    output: outf_rna_quantification
    resources: mem_mb = kallisto_mem_mb
    run:
        if RNA_TUMOR_ISPE:
            shell('kallisto quant -i {CDNA_KALLISTO_IDX} -b 100 -o {kallisto_out} {RNA_TUMOR_FQ1} {RNA_TUMOR_FQ2}')
        else:
            shell('kallisto quant -i {CDNA_KALLISTO_IDX} -b 100 -o {kallisto_out} --single -l 200 -s 30 {RNA_TUMOR_FQ1}')
        shell('cp {kallisto_out}/abundance.tsv {outf_rna_quantification}')
    
starfusion_out = F'{RES}/fusion/starfusion_out'
starfusion_bam = F'{starfusion_out}/Aligned.out.bam'
starfusion_res = F'{starfusion_out}/star-fusion.fusion_predictions.abridged.coding_effect.tsv'
starfusion_sjo = F'{starfusion_out}/SJ.out.tab'
starfusion_params = F' --genome_lib_dir {CTAT} --examine_coding_effect --output_dir {starfusion_out} --min_FFPM 0.1 '
rule RNA_tumor_fusion_detection:
    output:
        outbam = starfusion_bam,
        outres = starfusion_res,
        outsjo = starfusion_sjo,
    resources: mem_mb = star_mem_mb
    threads: star_nthreads
    run:
        if RNA_TUMOR_ISPE:
            shell('STAR-Fusion {starfusion_params} --CPU {star_nthreads} --left_fq {RNA_TUMOR_FQ1} --right_fq {RNA_TUMOR_FQ2} --outTmpDir {fifo_path_prefix}.starfusion.tmpdir')
        else:
            shell('STAR-Fusion {starfusion_params} --CPU {star_nthreads} --left_fq {RNA_TUMOR_FQ1} --outTmpDir {fifo_path_prefix}.starfusion.tmpdir')
        
fusion_neopeptide_faa = F'{peptide_dir}/{PREFIX}_fusion.fasta'
if isRNAskipped: make_dummy_files([fusion_neopeptide_faa, fusion_info_file])
rule RNA_fusion_peptide_generation:
    input:
        starfusion_res, outf_rna_quantification
    output:
        fusion_neopeptide_faa, fusion_info_file
    run:
        shell(
        'python {script_basedir}/parse_star_fusion.py -i {starfusion_res}'
        ' -e {outf_rna_quantification} -o {starfusion_out} -p {PREFIX} -t 1.0'
        ' && cp {starfusion_out}/{PREFIX}_fusion.fasta {peptide_dir}/{PREFIX}_fusion.fasta'
        ' && cp {starfusion_res} {fusion_info_file}'
        )
rna_tumor_bam = F'{alignment_dir}/{PREFIX}_RNA_tumor.bam'
rna_t_spl_bam = F'{alignment_dir}/{PREFIX}_RNA_t_spl.bam' # tumor splicing
dna_tumor_bam = F'{alignment_dir}/{PREFIX}_DNA_tumor.bam'
dna_normal_bam = F'{alignment_dir}/{PREFIX}_DNA_normal.bam'

rna_tumor_bai = F'{alignment_dir}/{PREFIX}_RNA_tumor.bam.bai'
rna_t_spl_bai = F'{alignment_dir}/{PREFIX}_RNA_t_spl.bam.bai'
dna_tumor_bai = F'{alignment_dir}/{PREFIX}_DNA_tumor.bam.bai'
dna_normal_bai = F'{alignment_dir}/{PREFIX}_DNA_normal.bam.bai'
rule RNA_tumor_preprocessing:
    input: starfusion_bam
    output:
        outbam = rna_tumor_bam,
        outbai = rna_tumor_bai,
    threads: 1 # samtools_nthreads
    run:
        shell(
        'rm {output.outbam}.tmp.*.bam || true '
        ' && samtools fixmate -@ {samtools_nthreads} -m {starfusion_bam} - '
        ' | samtools sort -@ {samtools_nthreads} -o - - '
        ' | samtools markdup -@ {samtools_nthreads} - {rna_tumor_bam}'
        ' && samtools index -@ {samtools_nthreads} {rna_tumor_bam}'
        )
# This rule is not-used in the end results (but may still be needed for QC), so it is still kept
HIGH_DP=1000*1000
rna_tumor_depth = F'{alignment_dir}/{PREFIX}_rna_tumor_F0xD04_depth.vcf.gz'
rna_tumor_depth_summary = F'{alignment_dir}/{PREFIX}_rna_tumor_F0xD04_depth_summary.tsv.gz'
rule RNA_postprocessing:
    input: rna_tumor_bam, rna_tumor_bai
    output: rna_tumor_depth 
    #, rna_tumor_depth_summary
    shell:
        ' samtools view -hu -@ {samtools_nthreads} -F 0xD04 {rna_tumor_bam} '
        ' | bcftools mpileup --threads {bcftools_nthreads} -a DP,AD -d {HIGH_DP} -f {REF} -q 0 -Q 0 -T {CTAT}/ref_annot.gtf.mini.sortu.bed - -o {rna_tumor_depth} '
        ' && bcftools index --threads {bcftools_nthreads} -ft {rna_tumor_depth} '
        ''' && cat {CTAT}/ref_annot.gtf.mini.sortu.bed | awk '{{ i += 1; s += $3-$2 }} END {{ print "exome_total_bases\t" s; }}' > {rna_tumor_depth_summary} '''
        ''' && bcftools query -f '%DP\n' {rna_tumor_depth} | awk '{{ i += 1 ; s += $1 }} END {{ print "exome_total_depth\t" s; }}' >> {rna_tumor_depth_summary} '''
# End of the not-used rule
   
asneo_out = F'{RES}/splicing/{PREFIX}_rna_tumor_splicing_asneo_out'
asneo_sjo = F'{asneo_out}/SJ.out.tab'
splicing_neopeptide_faa=F'{peptide_dir}/{PREFIX}_splicing.fasta'
if isRNAskipped: make_dummy_files([splicing_neopeptide_faa])
rule RNA_tumor_splicing_alignment:
    output: rna_t_spl_bam, rna_t_spl_bai, asneo_sjo 
    resources: mem_mb = star_mem_mb
    threads: star_nthreads
    run: # same as in PMC7425491 except for --sjdbOverhang 100
        shell(
        'STAR --genomeDir {REF}.star.idx --readFilesIn {RNA_TUMOR_FQ1} {RNA_TUMOR_FQ2} --runThreadN {star_nthreads} '
        ' –-outFilterMultimapScoreRange 1 --outFilterMultimapNmax 20 --outFilterMismatchNmax 10 --alignIntronMax 500000 –alignMatesGapMax 1000000 '
        ' --sjdbScore 2 --alignSJDBoverhangMin 1 --genomeLoad NoSharedMemory --outFilterMatchNminOverLread 0.33 --outFilterScoreMinOverLread 0.33 '
        ''' --sjdbOverhang 150 --outSAMstrandField intronMotif –sjdbGTFfile {ASNEO_GTF} --outFileNamePrefix {asneo_out}/ --readFilesCommand 'gunzip -c' ''' 
        ' --outSAMtype BAM Unsorted --outTmpDir {fifo_path_prefix}.star.tmpdir '
        ' && samtools fixmate -@ {samtools_nthreads} -m {asneo_out}/Aligned.out.bam - '
        ' | samtools sort -@ {samtools_nthreads} -o - -'
        ' | samtools markdup -@ {samtools_nthreads} - {rna_t_spl_bam}'
        ' && samtools index -@ {samtools_nthreads} {rna_t_spl_bam}'
        )

rule RNA_splicing_peptide_generation:
    input: rna_quant=outf_rna_quantification, sj=asneo_sjo # starfusion_sjo may also work (we didn't test this)
    output: splicing_neopeptide_faa 
    resources: mem_mb = 20000 # performance measured by Xiaofei Zhao
    threads: 1
    run:
        shell(
        'mkdir -p {info_dir} '
        ' && python {script_basedir}/software/ASNEO/neoheadhunter_ASNEO.py -j {input.sj} -g {ASNEO_REF} -o {asneo_out} -l 8,9,10,11 -p {PREFIX} -t 1.0 '
        ' -e {outf_rna_quantification}'
        ' && cat {asneo_out}/{PREFIX}_splicing_* > {peptide_dir}/{PREFIX}_splicing.fasta'
        )
rule DNA_tumor_alignment:
    output: dna_tumor_bam, dna_tumor_bai
    resources: mem_mb = bwa_samtools_mem_mb
    threads: bwa_nthreads
    shell:
        'rm {dna_tumor_bam}.tmp.*.bam || true'
        ' && bwa mem -t {bwa_nthreads} {REF} {DNA_TUMOR_FQ1} {DNA_TUMOR_FQ2} '
        ' | samtools fixmate -@ {samtools_nthreads} -m - -'
        ' | samtools sort -@ {samtools_nthreads} -o - -'
        ' | samtools markdup -@ {samtools_nthreads} - {dna_tumor_bam}'
        ' && samtools index -@ {samtools_nthreads} {dna_tumor_bam}'
    
rule DNA_normal_alignment:
    output: dna_normal_bam, dna_normal_bai
    resources: mem_mb = bwa_samtools_mem_mb
    threads: bwa_nthreads
    shell:
        'rm {dna_normal_bam}.tmp.*.bam || true'
        ' && bwa mem -t {bwa_nthreads} {REF} {DNA_NORMAL_FQ1} {DNA_NORMAL_FQ2}'
        ' | samtools fixmate -@ {samtools_nthreads} -m - -'
        ' | samtools sort -@ {samtools_nthreads} -o - -'
        ' | samtools markdup -@ {samtools_nthreads} - {dna_normal_bam}'
        ' && samtools index -@ {samtools_nthreads} {dna_normal_bam}'
    
gatk_jar=F'{script_basedir}/software/gatk-4.3.0.0/gatk-package-4.3.0.0-local.jar'

dna_vcf=F'{snvindel_dir}/{PREFIX}_DNA_tumor_DNA_normal.vcf'
dna_tonly_raw_vcf=F'{snvindel_dir}/{PREFIX}_DNA_tumor_DNA_normal.uvcTN.vcf.gz.byproduct/{PREFIX}_DNA_tumor_uvc1.vcf.gz'
if 'dna_vcf' in config: dna_vcf = config['dna_vcf']
    #os.makedirs(os.path.dirname(dna_vcf), exist_ok=True)
    #os.system(F'''cp {config['dna_vcf']} {dna_vcf}''')
rule DNA_SmallVariant_detection:
    input: tbam=dna_tumor_bam, tbai=dna_tumor_bai, nbam=dna_normal_bam, nbai=dna_normal_bai,
    output: dna_vcf, #dna_tonly_raw_vcf,
        vcf1 = F'{snvindel_dir}/{PREFIX}_DNA_tumor_DNA_normal.uvcTN.vcf.gz',
        vcf2 = F'{snvindel_dir}/{PREFIX}_DNA_tumor_DNA_normal.uvcTN-filter.vcf.gz',
        vcf3 = F'{snvindel_dir}/{PREFIX}_DNA_tumor_DNA_normal.uvcTN-delins.merged-simple-delins.vcf.gz',
    resources: mem_mb = uvc_mem_mb
    threads: uvc_nthreads
    run:
        if variantcaller == 'mutect2':
            shell('java -jar {gatk_jar} Mutect2 -R {REF} -I {dna_tumor_bam} -I {dna_normal_bam} -O {dna_vcf} 1> {dna_vcf}.stdout 2> {dna_vcf}.stderr ')
        else:
            shell(
        'uvcTN.sh {REF} {dna_tumor_bam} {dna_normal_bam} {output.vcf1} {PREFIX}_DNA_tumor,{PREFIX}_DNA_normal -t {uvc_nthreads_on_cmdline} '
            ' 1> {output.vcf1}.stdout.log 2> {output.vcf1}.stderr.log'
        ' && bcftools view {output.vcf1} -Oz -o {output.vcf2} '
            ' -i "(QUAL >= {tumor_normal_var_qual}) && (tAD[1] >= {tumor_depth}) && (tAD[1] >= (tAD[0] + tAD[1]) * {tumor_vaf}) && (nAD[1] <= (nAD[0] + nAD[1]) * {normal_vaf})"'
        ' && bash uvcvcf-raw2delins-all.sh {REF} {output.vcf2} {snvindel_dir}/{PREFIX}_DNA_tumor_DNA_normal.uvcTN-delins'
        ' && bcftools view {output.vcf3} -Ov -o {dna_vcf}'
            )

rna_vcf=F'{snvindel_dir}/{PREFIX}_RNA_tumor_DNA_normal.vcf'
rna_tonly_raw_vcf=F'{snvindel_dir}/{PREFIX}_RNA_tumor_DNA_normal.uvcTN.vcf.gz.byproduct/{PREFIX}_RNA_tumor_uvc1.vcf.gz'
rule RNA_SmallVariant_detection: # RNA filtering is more stringent
    input: tbam=rna_tumor_bam, tbai=rna_tumor_bai, nbam=dna_normal_bam, nbai=dna_normal_bai
    output: rna_vcf, #rna_tonly_raw_vcf,
        vcf1 = F'{snvindel_dir}/{PREFIX}_RNA_tumor_DNA_normal.uvcTN.vcf.gz',
        vcf2 = F'{snvindel_dir}/{PREFIX}_RNA_tumor_DNA_normal.uvcTN-filter.vcf.gz',
        vcf3 = F'{snvindel_dir}/{PREFIX}_RNA_tumor_DNA_normal.uvcTN-delins.merged-simple-delins.vcf.gz',
    resources: mem_mb = uvc_mem_mb
    threads: uvc_nthreads
    run:
         if variantcaller == 'mutect2':
             shell('java -jar {gatk_jar} Mutect2 -R {REF} -I {rna_tumor_bam} -I {dna_normal_bam} -O {rna_vcf} 1> {rna_vcf}.stdout 2> {rna_vcf}.stderr ')
         else:
             shell(
        'uvcTN.sh {REF} {rna_tumor_bam} {dna_normal_bam} {output.vcf1} {PREFIX}_RNA_tumor,{PREFIX}_DNA_normal -t {uvc_nthreads_on_cmdline} '
            ' 1> {output.vcf1}.stdout.log 2> {output.vcf1}.stderr.log'
        ' && bcftools view {output.vcf1} -Oz -o {output.vcf2} '
            ' -i "(QUAL >= 83) && (tAD[1] >= 7) && (tAD[1] >= (tAD[0] + tAD[1]) * 0.8) && (nAD[1] <= (nAD[0] + nAD[1]) * {normal_vaf})"'
        ' && bash uvcvcf-raw2delins-all.sh {REF} {output.vcf2} {snvindel_dir}/{PREFIX}_RNA_tumor_DNA_normal.uvcTN-delins'
        ' && bcftools view {output.vcf3} -Ov -o {rna_vcf}'
        )

# start-of-DNA-vep-mainline

dna_variant_effect = F'{snvindel_dir}/{PREFIX}_DNA_tumor_DNA_normal.variant_effect.tsv'
rule DNA_SmallVariant_effect_prediction:
    input: dna_vcf
    output: dna_variant_effect
    resources: mem_mb = vep_mem_mb
    threads: vep_nthreads
    shell: '''vep --no_stats --ccds --uniprot --hgvs --symbol --numbers --domains --gene_phenotype --canonical --protein --biotype --tsl --variant_class \
        --check_existing --total_length --allele_number --no_escape --xref_refseq --flag_pick_allele --offline --pubmed --af --af_1kg --af_gnomad \
        --regulatory --force_overwrite --assembly GRCh37 --buffer_size 5000 --failed 1 --format vcf --pick_order canonical,tsl,biotype,rank,ccds,length \
        --polyphen b --shift_hgvs 1 --sift b --species homo_sapiens \
        --dir {VEP_CACHE} --fasta {REF} --fork {vep_nthreads} --input_file {dna_vcf} --output_file {dna_variant_effect}'''

dna_snvindel_neopeptide_faa = F'{peptide_dir}/{DNA_PREFIX}_snv_indel.fasta'
dna_snvindel_wt_peptide_faa = F'{peptide_dir}/{DNA_PREFIX}_snv_indel_wt.fasta'
rule DNA_SmallVariant_peptide_generation:
    input: dna_variant_effect, outf_rna_quantification
    output: dna_snvindel_neopeptide_faa, dna_snvindel_wt_peptide_faa, dna_snvindel_info_file
    shell: '''python {script_basedir}/annotation2fasta.py -i {dna_variant_effect} -o {peptide_dir} -p {PEP_REF} \
        -r {REF} -s VEP -e {outf_rna_quantification} -P {DNA_PREFIX} --molecule_type=D -t -1
        cp {peptide_dir}/{DNA_PREFIX}_tmp_fasta/{DNA_PREFIX}_snv_indel_wt.fasta {dna_snvindel_wt_peptide_faa}
        cp {dna_variant_effect} {dna_snvindel_info_file}'''

# end-of-DNA-vep-mainline
# start-of-RNA-vep-sideline-for-rescue

rna_variant_effect = F'{snvindel_dir}/{PREFIX}_RNA_tumor_DNA_normal.variant_effect.tsv'
rule RNA_SmallVariant_effect_prediction:
    input: rna_vcf
    output: rna_variant_effect
    resources: mem_mb = vep_mem_mb
    threads: vep_nthreads
    shell: '''
        #bcftools view {dna_vcf} -Oz -o {dna_vcf}.gz && bcftools index -ft {dna_vcf}.gz
        #bcftools view {rna_vcf} -Oz -o {rna_vcf}.gz && bcftools index -ft {rna_vcf}.gz
        #bcftools isec {dna_vcf}.gz {rna_vcf}.gz -Oz -p {rna_vcf}.isecdir/
        vep --no_stats --ccds --uniprot --hgvs --symbol --numbers --domains --gene_phenotype --canonical --protein --biotype --tsl --variant_class \
        --check_existing --total_length --allele_number --no_escape --xref_refseq --flag_pick_allele --offline --pubmed --af --af_1kg --af_gnomad \
        --regulatory --force_overwrite --assembly GRCh37 --buffer_size 5000 --failed 1 --format vcf --pick_order canonical,tsl,biotype,rank,ccds,length \
        --polyphen b --shift_hgvs 1 --sift b --species homo_sapiens \
        --dir {VEP_CACHE} --fasta {REF} --fork {vep_nthreads} --input_file {rna_vcf} --output_file {rna_variant_effect}'''
        # --dir {VEP_CACHE} --fasta {REF} --fork {vep_nthreads} --input_file {rna_vcf}.isecdir/0001.vcf.gz --output_file {rna_variant_effect}
rna_snvindel_neopeptide_faa = F'{peptide_dir}/{RNA_PREFIX}_snv_indel.fasta'
rna_snvindel_wt_peptide_faa = F'{peptide_dir}/{RNA_PREFIX}_snv_indel_wt.fasta'
if isRNAskipped: make_dummy_files([rna_snvindel_neopeptide_faa, rna_snvindel_wt_peptide_faa, rna_snvindel_info_file])
rule RNA_SmallVariant_peptide_generation:
    input: rna_variant_effect, outf_rna_quantification
    output: rna_snvindel_neopeptide_faa, rna_snvindel_wt_peptide_faa, rna_snvindel_info_file
    shell: '''python {script_basedir}/annotation2fasta.py -i {rna_variant_effect} -o {peptide_dir} -p {PEP_REF} \
        -r {REF} -s VEP -e {outf_rna_quantification} -P {RNA_PREFIX} --molecule_type=R -t -1
        cp {peptide_dir}/{RNA_PREFIX}_tmp_fasta/{RNA_PREFIX}_snv_indel_wt.fasta {rna_snvindel_wt_peptide_faa}
        cp {rna_variant_effect} {rna_snvindel_info_file}'''

# end-of-RNA-vep-sideline-for-rescue

# example return value: ['HLA-A02:01']
def retrieve_hla_alleles():
    if 'comma_sep_hla_list' in config: return config['comma_sep_hla_list'].split(',')
    ret = []
    with open(F'{hla_out}') as file:
        reader = csv.reader(file, delimiter='\t')
        for line in reader:
            if line[0] == '': continue
            for i in range(1,7,1): 
                if line[i] != 'None': ret.append('HLA-' + line[i].replace('*',''))
    return ret

def run_netMHCpan(args):
    hla_str, infaa = args
    return call_with_infolog(F'{netmhcpan_cmd} -f {infaa} -a {hla_str} -l 8,9,10,11 -BA > {infaa}.netMHCpan-result')

def peptide_to_pmhc_binding_affinity(infaa, outtsv, hla_strs):
    call_with_infolog(F'rm -r {outtsv}.tmpdir/ || true')
    call_with_infolog(F'mkdir -p {outtsv}.tmpdir/')
    call_with_infolog(F'''cat {infaa} | awk '{{print $1}}' |  split -l 20 - {outtsv}.tmpdir/SPLITTED.''')
    cmds = [F'{netmhcpan_cmd} -f {outtsv}.tmpdir/{faafile} -a {hla_str} -l 8,9,10,11 -BA > {outtsv}.tmpdir/{faafile}.{hla_str}.netMHCpan-result'
            for hla_str in hla_strs for faafile in os.listdir(F'{outtsv}.tmpdir/') if faafile.startswith('SPLITTED.')]
    with open(F'{outtsv}.tmpdir/tmp.sh', 'w') as shfile:
        for cmd in cmds: shfile.write(cmd + '\n')
    # Each netmhcpan process uses much less than 100% CPU, so we can spawn many more processes
    call_with_infolog(F'cat {outtsv}.tmpdir/tmp.sh | parallel -j {netmhc_ncores}')
    call_with_infolog(F'find {outtsv}.tmpdir/ -iname "SPLITTED.*.netMHCpan-result" | xargs cat > {outtsv}')
    
all_vars_peptide_faa   = F'{pmhc_dir}/{PREFIX}_all_peps.fasta'
all_vars_netmhcpan_txt = F'{pmhc_dir}/{PREFIX}_all_peps.netmhcpan.txt'
if 'all_vars_peptide_faa' in config: all_vars_peptide_faa = config['all_vars_peptide_faa']

rule Peptide_preprocessing:
    input: dna_snvindel_neopeptide_faa, dna_snvindel_wt_peptide_faa,
           rna_snvindel_neopeptide_faa, rna_snvindel_wt_peptide_faa,
           fusion_neopeptide_faa, splicing_neopeptide_faa
    output: all_vars_peptide_faa
    threads: 1
    run:
        shell('cat {dna_snvindel_neopeptide_faa} | python {script_basedir}/fasta_filter.py --tpm {tumor_abundance_filt_thres} '
            ' | python {script_basedir}/neoexpansion.py --nbits 1.0 > {dna_snvindel_neopeptide_faa}.expansion')
        shell('cat'
            ' {dna_snvindel_neopeptide_faa}.expansion {dna_snvindel_wt_peptide_faa} '
            ' {rna_snvindel_neopeptide_faa}           {rna_snvindel_wt_peptide_faa} '
            ' {fusion_neopeptide_faa} {splicing_neopeptide_faa} '
            ' | python {script_basedir}/fasta_filter.py --tpm {tumor_abundance_filt_thres} > {all_vars_peptide_faa}')
rule PeptideMHC_binding_affinity_prediction:
    input: all_vars_peptide_faa, hla_out
    output: all_vars_netmhcpan_txt
    threads: netmhc_nthreads # 12
    run: 
        peptide_to_pmhc_binding_affinity(all_vars_peptide_faa, all_vars_netmhcpan_txt, retrieve_hla_alleles())

try:
    from urllib.parse import urlparse
except ImportError:
    from urlparse import urlparse
def uri_to_user_address_port_path(uri):
    if uri.startswith('ssh://'):
        parsed_url = urlparse(uri)
        addressport = parsed_url.netloc.split(':')
        assert len(addressport) <= 2
        if len(addressport) == 2:
            complete_address, port = addressport
        else:
            complete_address, port = (addressport[0], 22)
        useraddress = complete_address.split('@')
        assert len(useraddress) <= 2
        if len(useraddress) == 2:
            user, address = useraddress
        else:
            user, address = (parsed_url.username, useraddress[0])
        return (user, address, port, parsed_url.path)
    return (None, None, None, uri)

def run_netMHCstabpan(bindstab_filter_py, inputfile = F'{pmhc_dir}/{PREFIX}_bindaff_filtered.tsv', outdir = pmhc_dir):
    user, address, port, path = uri_to_user_address_port_path(netmhcstabpan_cmd)
    if netmhcstabpan_cmd == path:
        run_calculation = F'python {bindstab_filter_py} -i {inputfile} -o {outdir} -n {path} -b {binding_stability_filt_thres} -p {PREFIX}'
        call_with_infolog(run_calculation)
    else:
        outputfile1 = F'{outdir}/{PREFIX}_bindstab_raw.txt'
        outputfile2 = F'{outdir}/{PREFIX}_candidate_pmhc.tsv'
        remote_rmdir = F' sshpass -p "$NeohunterRemotePassword" ssh -p {port} {user}@{address} rm -r /tmp/{outdir}/ || true'
        remote_mkdir = F' sshpass -p "$NeohunterRemotePassword" ssh -p {port} {user}@{address} mkdir -p /tmp/{outdir}/'
        remote_send = F' sshpass -p "$NeohunterRemotePassword" scp -P {port} {bindstab_filter_py} {inputfile} {user}@{address}:/tmp/{outdir}/'
        remote_main_cmd = F'python /tmp/{outdir}/bindstab_filter.py -i /tmp/{inputfile} -o /tmp/{outdir} -n {path} -b {binding_stability_filt_thres} -p {PREFIX}'
        remote_exe = F' sshpass -p "$NeohunterRemotePassword" ssh -p {port} {user}@{address} {remote_main_cmd}'
        remote_receive1 = F' sshpass -p "$NeohunterRemotePassword" scp -P {port} {user}@{address}:/tmp/{outputfile1} {outdir}'
        remote_receive2 = F' sshpass -p "$NeohunterRemotePassword" scp -P {port} {user}@{address}:/tmp/{outputfile2} {outdir}'
        call_with_infolog(remote_rmdir)
        call_with_infolog(remote_mkdir)
        call_with_infolog(remote_send)
        call_with_infolog(remote_exe)
        call_with_infolog(remote_receive1)
        call_with_infolog(remote_receive2)

all_vars_netmhc_filtered_tsv = F'{pmhc_dir}/{PREFIX}_all_peps.netmhcpan_filtered.tsv'
#rule PeptideMHC_binding_affinity_filter:
#    input: all_vars_peptide_faa, all_vars_netmhcpan_txt
#    output: all_vars_netmhc_filtered_tsv
#    shell: 
#        'python {script_basedir}/parse_netmhcpan.py -f {all_vars_peptide_faa} -n {all_vars_netmhcpan_txt} -o {all_vars_netmhc_filtered_tsv} '
#        '-a {binding_affinity_filt_thres} -l SB,WB'

all_vars_bindstab_raw_txt = F'{pmhc_dir}/{PREFIX}_bindstab_raw.txt'
all_vars_bindstab_filtered_tsv = F'{pmhc_dir}/{PREFIX}_candidate_pmhc.tsv'
rule PeptideMHC_binding_stability_prediction:
    input: all_vars_peptide_faa, all_vars_netmhcpan_txt
    # all_vars_netmhc_filtered_tsv
    output: all_vars_bindstab_raw_txt, all_vars_bindstab_filtered_tsv
    run: 
        bindstab_filter_py = F'{script_basedir}/bindstab_filter.py'
        shell('python {script_basedir}/parse_netmhcpan.py -f {all_vars_peptide_faa} -n {all_vars_netmhcpan_txt} -o {all_vars_netmhc_filtered_tsv} '
              '-a {binding_affinity_filt_thres} -l SB,WB,NB')
        run_netMHCstabpan(bindstab_filter_py, F'{all_vars_netmhc_filtered_tsv}', F'{pmhc_dir}')

iedb_path = F'{script_basedir}/database/iedb.fasta' # '/mnt/d/code/neohunter/NeoHunter/database/iedb.fasta'
prioritization_thres_params = ' '.join([x.strip() for x in F'''
--binding-affinity-hard-thres {binding_affinity_hard_thres}
--binding-affinity-soft-thres {binding_affinity_soft_thres}
--binding-stability-hard-thres {binding_stability_hard_thres}
--binding-stability-soft-thres {binding_stability_soft_thres}
--tumor-abundance-hard-thres {tumor_abundance_hard_thres}
--tumor-abundance-soft-thres {tumor_abundance_soft_thres}
--agretopicity-thres {agretopicity_thres}
--foreignness-thres {foreignness_thres}
--alteration-type {alteration_type}
'''.strip().split()])
prioritization_function_params = ''

logging.debug(F'neoheadhunter_prioritization_tsv = {neoheadhunter_prioritization_tsv} (from {prioritization_dir})')

ruleorder: Prioritization_with_all_TCRs > Prioritization_with_all_TCRs_with_minimal_varinfo

rule Prioritization_with_all_TCRs:
    input: iedb_path, all_vars_bindstab_filtered_tsv, # dna_tonly_raw_vcf, rna_tonly_raw_vcf, # dna_vcf, rna_vcf,
        dna_snvindel_info_file, rna_snvindel_info_file, fusion_info_file
        # splicing_info_file   
    output: neoheadhunter_prioritization_tsv, final_pipeline_out
    run:
        if variantcaller == 'mutect2':
            call_with_infolog(F'python {script_basedir}/neoheadhunter_prioritization.py -i {all_vars_bindstab_filtered_tsv} -I {iedb_path} '
            F' -D {dna_snvindel_info_file} -R {rna_snvindel_info_file} -F {fusion_info_file} '
            F' -o {neoheadhunter_prioritization_tsv} -t {alteration_type} --passflag 0x0 '
            F''' {prioritization_thres_params} {prioritization_function_params.replace('_', '-')}''')
        else:
            call_with_infolog(F'python {script_basedir}/neoheadhunter_prioritization.py -i {all_vars_bindstab_filtered_tsv} -I {iedb_path} '
            F' -D {dna_snvindel_info_file} -R {rna_snvindel_info_file} -F {fusion_info_file} ' # ' -S {splicing_info_file} '
            F' -o {neoheadhunter_prioritization_tsv} -t {alteration_type} --passflag 0x0 '
            F' --dna-vcf {dna_tonly_raw_vcf} --rna-vcf {rna_tonly_raw_vcf} ' # ' --rna-depth {rna_tumor_depth_summary} '
            F''' {prioritization_thres_params} {prioritization_function_params.replace('_', '-')}''')
        call_with_infolog(F'cp {neoheadhunter_prioritization_tsv} {final_pipeline_out}')

### rules that were not run by default

rule Prioritization_with_all_TCRs_with_minimal_varinfo:
    input: iedb_path, all_vars_bindstab_filtered_tsv 
    output: pmhc_as_input_prioritization_tsv
    run: 
        call_with_infolog(F'python {script_basedir}/neoheadhunter_prioritization.py -i {all_vars_bindstab_filtered_tsv} -I {iedb_path} '
            F' -o {pmhc_as_input_prioritization_tsv} -t {alteration_type} --passflag 0x3 '
            F''' {prioritization_thres_params} {prioritization_function_params.replace('_', '-')}''')

neoheadhunter_validation_out=F'{neoheadhunter_prioritization_tsv}.validation'
pmhc_as_input_validation_out=F'{pmhc_as_input_prioritization_tsv}.validation'

###   validation rules

rule Prioritization_with_all_TCRs_validation:
    input: neoheadhunter_prioritization_tsv
    output: neoheadhunter_validation_out
    run:
        truth_file = config.get('truth_file')
        truth_patientID = config.get('truth_patientID')
        call_with_infolog(F'python {script_basedir}/neoheadhunter_prioritization.py -i - -I - --truth-file {truth_file} --truth-patientID {truth_patientID} '
            F' -o {neoheadhunter_prioritization_tsv} -t {alteration_type} --passflag 0x0 '
            F''' {prioritization_thres_params} {prioritization_function_params.replace('_', '-')}''')

rule Prioritization_with_all_TCRs_with_minimal_varinfo_validation:
    input: pmhc_as_input_prioritization_tsv
    output: pmhc_as_input_validation_out
    run:
        truth_file = config.get('truth_file')
        truth_patientID = config.get('truth_patientID')
        call_with_infolog(F'python {script_basedir}/neoheadhunter_prioritization.py -i - -I - --truth-file {truth_file} --truth-patientID {truth_patientID} '
            F' -o {pmhc_as_input_prioritization_tsv} -t {alteration_type} --passflag 0x3 '
            F''' {prioritization_thres_params} {prioritization_function_params.replace('_', '-')}''')

### auxiliary prioritization steps

mixcr_output_dir = F'{prioritization_dir}/{PREFIX}_mixcr_output'
mixcr_output_pref = F'{mixcr_output_dir}/{PREFIX}'
mixcr_output_done_flag = F'{mixcr_output_dir}.DONE'
mixcr_cmdline_params = ' analyze shotgun -s hs --starting-material rna --only-productive --receptor-type tcr '
mixcr_mem_gb = max((mixcr_mem_mb // 1000 - 4, 1))
logging.debug(F'MIXCR_PATH = {MIXCR_PATH}')
rule TCR_clonotype_detection:
    output: mixcr_output_done_flag
    resources: mem_mb = mixcr_mem_mb
    threads: mixcr_nthreads
    shell: '''
        mkdir -p {mixcr_output_dir}
        java -Xmx{mixcr_mem_gb}g -jar {MIXCR_PATH} {mixcr_cmdline_params} {RNA_TUMOR_FQ1} {RNA_TUMOR_FQ2} {mixcr_output_pref}
        touch {mixcr_output_done_flag}
    '''
tcr_specificity_software = 'ERGO'
ergo2_score = F'{prioritization_dir}/{PREFIX}_tcr_specificity_score.csv'
rule PeptideMHC_TCR_interaction_prediction:
    input: all_vars_bindstab_filtered_tsv, mixcr_output_done_flag
    output: ergo2_score
    threads: ergo2_nthreads # 12
    shell: '''
        python {script_basedir}/rank_software_input.py -m {mixcr_output_pref} -n {all_vars_bindstab_filtered_tsv} -o {prioritization_dir} -t {tcr_specificity_software} -p {PREFIX}
        cd {ERGO_EXE_DIR}
        python {ERGO_PATH} mcpas {prioritization_dir}/{PREFIX}_cdr_ergo.csv {ergo2_score}
        '''
rule Prioritization_with_each_TCR:
    input: all_vars_bindstab_filtered_tsv, ergo2_score
    output: tcr_specificity_result
    threads: 1   
    shell: '''
        cd {script_basedir}
        python {script_basedir}/parse_rank_software.py -i {ergo2_score} -n {all_vars_bindstab_filtered_tsv} -o {prioritization_dir}/ -t {tcr_specificity_software} -p {PREFIX}
        python {script_basedir}/add_detail_info.py -i {prioritization_dir}/{PREFIX}_neoantigen_rank_tcr_specificity.tsv -o {prioritization_dir}/ -p {PREFIX}
    '''

