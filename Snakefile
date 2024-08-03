import csv, datetime, json, logging, multiprocessing, os, shutil, subprocess

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

prep_peplens = config['prep_peplens']
#peplen_list = sorted(int(x) for x in prep_peplens.split(','))
#assert peplen_list
#peplen_list = list(range(peplen_list[0] - 1, peplen_list[-1] + 1 + 1)) # [peplen_list[0] - 1] + peplen_list + [peplen_list[-1] + 1]
#prep_peplens = ','.join(F'{peplen}' for peplen in peplen_list)

binding_affinity_filt_thres = config['binding_affinity_filt_thres']
#binding_affinity_hard_thres = config['binding_affinity_hard_thres']
#binding_affinity_soft_thres = config['binding_affinity_soft_thres']

binding_stability_filt_thres = config['binding_stability_filt_thres']
#binding_stability_hard_thres = config['binding_stability_hard_thres']
#binding_stability_soft_thres = config['binding_stability_soft_thres']

tumor_abundance_filt_thres = config['tumor_abundance_filt_thres']
#tumor_abundance_hard_thres = config['tumor_abundance_hard_thres']
#tumor_abundance_soft_thres = config['tumor_abundance_soft_thres']
#tumor_abundance_recognition_thres = config['tumor_abundance_recognition_thres']

agretopicity_thres = config['agretopicity_thres']
foreignness_thres = config['foreignness_thres']
alteration_type = config['alteration_type']

netmhc_ncores = config['netmhc_ncores']
netmhc_nthreads = config['netmhc_nthreads']
ergo2_nthreads = config['ergo2_nthreads']

### Section 4: parameters having some default values of relative paths

MIXCR_PATH = config.get('mixcr_path',  F'{script_basedir}/software/mixcr.jar')
ERGO_EXE_DIR = config.get('ergo_exe_dir', F'{script_basedir}/software/ERGO-II')

HLA_REF = config.get('hla_ref', subprocess.check_output('printf $(dirname $(which OptiTypePipeline.py))/data/hla_reference_rna.fasta', shell=True).decode(sys.stdout.encoding))
CDNA_REF = config.get('cdna_ref', F'{script_basedir}/database/Homo_sapiens.GRCh37.cdna.all.fa') # .kallisto-idx
PEP_REF = config.get('pep_ref', F'{script_basedir}/database/Homo_sapiens.GRCh37.pep.all.fa')
VEP_CACHE = config.get('vep_cache', F'{script_basedir}/database')
CTAT = config.get('ctat', F'{script_basedir}/database/GRCh37_gencode_v19_CTAT_lib_Mar012021.plug-n-play/ctat_genome_lib_build_dir/')
ASNEO_REF = config.get('asneo_ref', F'{script_basedir}/database/hg19.fa')          # The {CTAT} REF does not work with ASNEO
ASNEO_GTF = config.get('asneo_gtf', F'{script_basedir}/database/hg19.refGene.gtf') # The {CTAT} gtf does not work with ASNEO

if config.get('species') == 'Mus_musculus':
    HLA_REF = config.get('hla_ref', subprocess.check_output('printf $(dirname $(which OptiTypePipeline.py))/data/hla_reference_rna.fasta', shell=True).decode(sys.stdout.encoding))
    CDNA_REF = config.get('cdna_ref', F'{script_basedir}/database/Mus_musculus_balbcj.BALB_cJ_v1.cdna.all.fa') # .kallisto-idx
    PEP_REF = config.get('pep_ref', F'{script_basedir}/database/Mus_musculus_balbcj.BALB_cJ_v1.pep.all.fa')
    VEP_CACHE = config.get('vep_cache', F'{script_basedir}/database')
    CTAT = config.get('ctat', F'{script_basedir}/database/Mouse_GRCm39_M31_CTAT_lib_Nov092022.plug-n-play/ctat_genome_lib_build_dir/')
    ASNEO_REF = config.get('asneo_ref', F'{script_basedir}/database/mm39.fa')          # The {CTAT} REF does not work with ASNEO
    ASNEO_GTF = config.get('asneo_gtf', F'{script_basedir}/database/mm39.refGene.gtf') # The {CTAT} gtf does not work with ASNEO

### Section 5: parameters having some default values depending on other values
REF = config.get('REF', F'{CTAT}/ref_genome.fa')
ERGO_PATH = config.get('ERGO_PATH', F'{ERGO_EXE_DIR}/Predict.py')
CDNA_KALLISTO_IDX = config.get('cdna_kallisto_idx', F'{CDNA_REF}.kallisto-idx')

tmpdirID = '.'.join([script_start_datetime.strftime('%Y-%m-%d_%H-%M-%S'), str(os.getpid()), PREFIX])
fifo_path_prefix = os.path.sep.join([config['fifo_dir'], tmpdirID])
logging.debug(F'fifo_path_prefix={fifo_path_prefix}')

variantcaller = config.get('variantcaller', 'uvc') # uvc or mutect2, please be aware that the performance of mutect2 is not evaluated. 
stab_tmp = config.get('netmhcstabpan_tmp', '/tmp')

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
IS_PODMAN_USED_TO_WORKAROUND_OPTITYPE_MEM_LEAK = False
OPTITYPE_CONDA_ENV = 'optitype_env'
OPTITYPE_CONFIG = f"{script_basedir}/software/optitype.config.ini"
OPTITYPE_NOPATH_CONFIG = f"{script_basedir}/software/optitype_nopath.config.ini"

NA_REP = ''
def isna(arg): return arg in [None, '', 'NA', 'Na', 'None', 'none', '.']
def call_with_infolog(cmd, in_shell = True):
    logging.info(cmd)
    return subprocess.call(cmd, shell = in_shell)
def make_dummy_files(files, origfile = F'{script_basedir}/placeholders/EmptyFile', extension='.DummyPlaceholder', is_refresh_allowed=False):
    logging.info(F'Trying to file-copy from {origfile} to {files} with extension="{extension}" and is_refresh_allowed={is_refresh_allowed}')
    for file in files:
        if (config.get('refresh_files', 0) == 1 and is_refresh_allowed) or not os.path.exists(file):
            if os.path.exists(file): call_with_infolog(F'rm {file}')
            logging.info(F'Making directory of the file {file}')
            os.makedirs(os.path.dirname(file), exist_ok=True)
            try:
                shutil.copy2(origfile, file)
                if extension: call_with_infolog(F'touch {file}{extension}')
            except shutil.SameFileError as err:
                logging.warning(err)
            if config.get('refresh_files', 0) == 1 and is_refresh_allowed:
                call_with_infolog(F'touch {file}')

IS_ANY_TUMOR_SEQ_DATA_AS_INPUT = ((not isna(DNA_TUMOR_FQ1))  or (not isna(RNA_TUMOR_FQ1)))
DNA_TUMOR_ISPE  = (not isna(DNA_TUMOR_FQ2))
DNA_NORMAL_ISPE = (not isna(DNA_NORMAL_FQ2))
RNA_TUMOR_ISPE  = (not isna(RNA_TUMOR_FQ2))

isRNAskipped = isna(RNA_TUMOR_FQ1)
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

features_extracted_from_reads_tsv = F'{prioritization_dir}/{PREFIX}_features_from_reads.tsv' # prioritization from the reads of sequencing data
features_extracted_from_pmhcs_tsv = F'{prioritization_dir}/{PREFIX}_features_from_pmhcs.tsv' # prioritization from the pMHC summary info

tcr_specificity_result = F'{prioritization_dir}/{PREFIX}_neoantigen_rank_tcr_specificity_with_detail.tsv'

#final_pipeline_out = F'{RES}/{PREFIX}_final_neoepitope_candidates.tsv'
pipeline_out_from_reads = F'{RES}/{PREFIX}_prioritization_from_reads.tsv'
pipeline_out_from_pmhcs = F'{RES}/{PREFIX}_prioritization_from_pmhcs.tsv'

if   IS_ANY_TUMOR_SEQ_DATA_AS_INPUT      : pipeline_out_final = pipeline_out_from_reads
elif 'tumor_spec_peptide_fasta' in config: pipeline_out_final = pipeline_out_from_pmhcs
else:
    logging.critical('Either dna_tumor_fq1, rna_tumor_fq1 or tumor_spec_peptide_fasta must be specified in the config. ')
    exit(-1)
rule all:
    input: pipeline_out_final

#rule all:
#    input: final_pipeline_out #, tcr_specificity_result

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
        shell('rm -r {RES}/hla_typing/optitype_out/ || true && mkdir -p {RES}/hla_typing/optitype_out')
        if IS_PODMAN_USED_TO_WORKAROUND_OPTITYPE_MEM_LEAK:
            shell('cp {OPTITYPE_CONFIG} {RES}/hla_typing/config.ini')
        if RNA_TUMOR_ISPE:
            if IS_PODMAN_USED_TO_WORKAROUND_OPTITYPE_MEM_LEAK:
                shell('podman run -v {hla_typing_dir}:/data/ -t quay.io/biocontainers/optitype:1.3.2--py27_3 /usr/local/bin/OptiTypePipeline.py'
                      ' -c /data/config.ini -i /data/{hla_fq_r1_fname} /data/{hla_fq_r2_fname} --rna -o /data/optitype_out/ > {hla_out}.OptiType-container.stdout')
            elif OPTITYPE_CONDA_ENV != '':
                shell('$CONDA_EXE run -n {OPTITYPE_CONDA_ENV} OptiTypePipeline.py -i {hla_fq_r1} {hla_fq_r2} --rna -o {RES}/hla_typing/optitype_out/'
                      ' -c {OPTITYPE_NOPATH_CONFIG} > {output.out}.OptiType.stdout')
            else:
                shell('OptiTypePipeline.py -i {hla_fq_r1} {hla_fq_r2} --rna -o {RES}/hla_typing/optitype_out/ > {output.out}.OptiType.stdout')
        else:
            if IS_PODMAN_USED_TO_WORKAROUND_OPTITYPE_MEM_LEAK:
                shell('podman run -v {hla_typing_dir}:/data/ -t quay.io/biocontainers/optitype:1.3.2--py27_3 /usr/local/bin/OptiTypePipeline.py'
                      ' -c /data/config.ini -i /data/{hla_fq_se_fname}                         --rna -o /data/optitype_out/ > {hla_out}.OptiType-container.stdout')
            elif OPTITYPE_CONDA_ENV != '':
                shell('$$CONDA_EXE run -n {OPTITYPE_CONDA_ENV} OptiTypePipeline.py -i {hla_fq_se} --rna -o {RES}/hla_typing/optitype_out/'
                      ' -c {OPTITYPE_NOPATH_CONFIG} > {output.out}.OptiType.stdout')
            else:
                shell('OptiTypePipeline.py -i {hla_fq_se} --rna -o {RES}/hla_typing/optitype_out/ > {output.out}.OptiType.stdout')
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
        
fusion_neopeptide_fasta = F'{peptide_dir}/{PREFIX}_fusion.fasta'
if isRNAskipped: make_dummy_files([fusion_neopeptide_fasta, fusion_info_file])
rule RNA_fusion_peptide_generation:
    input:
        starfusion_res, outf_rna_quantification
    output:
        fusion_neopeptide_fasta, fusion_info_file
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
splicing_neopeptide_fasta=F'{peptide_dir}/{PREFIX}_splicing.fasta'
if isRNAskipped: make_dummy_files([splicing_neopeptide_fasta])
rule RNA_tumor_splicing_alignment:
    output: rna_t_spl_bam, rna_t_spl_bai, asneo_sjo 
    resources: mem_mb = star_mem_mb
    threads: star_nthreads
    run: # same as in PMC7425491 except for --sjdbOverhang 100
        if RNA_TUMOR_ISPE:
            readFilesInParam=F"{RNA_TUMOR_FQ1} {RNA_TUMOR_FQ2}"
        else:
            readFilesInParam=F"{RNA_TUMOR_FQ1}"
        shell(
        'STAR --genomeDir {REF}.star.idx --readFilesIn {readFilesInParam} --runThreadN {star_nthreads} '
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
    output: splicing_neopeptide_fasta 
    resources: mem_mb = 20000 # performance measured by Xiaofei Zhao
    threads: 1
    run:
        shell(
        'mkdir -p {info_dir} '
        ' && python {script_basedir}/software/ASNEO/neoASNEO.py -j {input.sj} -g {ASNEO_REF} -o {asneo_out} -l 8,9,10,11 -p {PREFIX} -t 1.0 '
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
if config.get('species', '') == 'Homo_sapiens':
    vep_species = 'homo_sapiens'
    vep_assembly = 'GRCh37'
    vep_params = '--polyphen b --shift_hgvs 1 --sift b'
elif config.get('species', '') == 'Mus_musculus':
    vep_species = 'mus_musculus_balbcj'
    vep_assembly = 'BALB_cJ_v1'
    vep_params = ''
else:
    exit(-1)

rule DNA_SmallVariant_effect_prediction:
    input: dna_vcf
    output: dna_variant_effect
    resources: mem_mb = vep_mem_mb
    threads: vep_nthreads
    shell: '''vep --no_stats --ccds --uniprot --hgvs --symbol --numbers --domains --gene_phenotype --canonical --protein --biotype --tsl --variant_class \
        --check_existing --total_length --allele_number --no_escape --xref_refseq --flag_pick_allele --offline --pubmed --af --af_1kg --af_gnomad \
        --regulatory --force_overwrite --assembly {vep_assembly} --buffer_size 5000 --failed 1 --format vcf --pick_order canonical,tsl,biotype,rank,ccds,length \
        {vep_params} --species {vep_species} \
        --dir {VEP_CACHE} --fasta {REF} --fork {vep_nthreads} --input_file {dna_vcf} --output_file {dna_variant_effect}'''

dna_snvindel_neopeptide_fasta = F'{peptide_dir}/{DNA_PREFIX}_snv_indel.fasta'
# dna_snvindel_wt_peptide_fasta = F'{peptide_dir}/{DNA_PREFIX}_snv_indel_wt.fasta'
rule DNA_SmallVariant_peptide_generation:
    input: dna_variant_effect, outf_rna_quantification
    output: dna_snvindel_neopeptide_fasta, # dna_snvindel_wt_peptide_fasta, 
        dna_snvindel_info_file
    shell: '''python {script_basedir}/annotation2fasta.py -i {dna_variant_effect} -o {peptide_dir} -p {PEP_REF} \
        -r {REF} -s VEP -e {outf_rna_quantification} -P {DNA_PREFIX} --molecule_type=D -t -1        
        cp {dna_variant_effect} {dna_snvindel_info_file}'''
# cp {peptide_dir}/{DNA_PREFIX}_tmp_fasta/{DNA_PREFIX}_snv_indel_wt.fasta {dna_snvindel_wt_peptide_fasta}

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
        --regulatory --force_overwrite --assembly {vep_assembly} --buffer_size 5000 --failed 1 --format vcf --pick_order canonical,tsl,biotype,rank,ccds,length \
        {vep_params} --species {vep_species} \
        --dir {VEP_CACHE} --fasta {REF} --fork {vep_nthreads} --input_file {rna_vcf} --output_file {rna_variant_effect}'''
        # --dir {VEP_CACHE} --fasta {REF} --fork {vep_nthreads} --input_file {rna_vcf}.isecdir/0001.vcf.gz --output_file {rna_variant_effect}
rna_snvindel_neopeptide_fasta = F'{peptide_dir}/{RNA_PREFIX}_snv_indel.fasta'
# rna_snvindel_wt_peptide_fasta = F'{peptide_dir}/{RNA_PREFIX}_snv_indel_wt.fasta'
if isRNAskipped: make_dummy_files([rna_snvindel_neopeptide_fasta, 
        # rna_snvindel_wt_peptide_fasta, 
        rna_snvindel_info_file])
rule RNA_SmallVariant_peptide_generation:
    input: rna_variant_effect, outf_rna_quantification
    output: rna_snvindel_neopeptide_fasta, # rna_snvindel_wt_peptide_fasta, 
        rna_snvindel_info_file
    shell: '''python {script_basedir}/annotation2fasta.py -i {rna_variant_effect} -o {peptide_dir} -p {PEP_REF} \
        -r {REF} -s VEP -e {outf_rna_quantification} -P {RNA_PREFIX} --molecule_type=R -t -1        
        cp {rna_variant_effect} {rna_snvindel_info_file}'''
# cp {peptide_dir}/{RNA_PREFIX}_tmp_fasta/{RNA_PREFIX}_snv_indel_wt.fasta {rna_snvindel_wt_peptide_fasta}

# end-of-RNA-vep-sideline-for-rescue

# example return value: ['HLA-A02:01']
def retrieve_hla_alleles():
    if 'comma_sep_hla_list' in config: return config['comma_sep_hla_list'].split(',')
    if not os.path.exists(hla_out): return []
    ret = []
    with open(F'{hla_out}') as file:
        reader = csv.reader(file, delimiter='\t')
        for lineno, line in enumerate(reader):
            if line[0] == '': continue
            if len(ret) == 0: assert len(line) > 7
            for i in range(1, min((len(line)) ,7), 1):
                if line[i] != 'None': ret.append('HLA-' + line[i].replace('*',''))
    return sorted(list(set(ret)))

#def run_netMHCpan(args):
#    hla_str, infaa = args
#    return call_with_infolog(F'{netmhcpan_cmd} -f {infaa} -a {hla_str} -l {prep_peplens} -BA > {infaa}.netMHCpan-result')

def peptide_to_pmhc_binding_affinity(infaa, outtsv, hla_string):
    call_with_infolog(F'rm -r {outtsv}.tmpdir/ || true')
    call_with_infolog(F'mkdir -p {outtsv}.tmpdir/')
    call_with_infolog(F'''cat {infaa} | awk '{{print $1}}' |  split -l 20 - {outtsv}.tmpdir/SPLITTED.''')
    cmds = [F'{netmhcpan_cmd} -f {outtsv}.tmpdir/{faafile} -a {hla_string} -l {prep_peplens} -BA > {outtsv}.tmpdir/{faafile}.{hla_string}.netMHCpan-result'
            # for hla_str in hla_strs
            for faafile in os.listdir(F'{outtsv}.tmpdir/') if faafile.startswith('SPLITTED.')]
    with open(F'{outtsv}.tmpdir/tmp.sh', 'w') as shfile:
        for cmd in cmds: shfile.write(cmd + '\n')
    # Each netmhcpan process uses much less than 100% CPU, so we can spawn many more processes
    call_with_infolog(F'cat {outtsv}.tmpdir/tmp.sh | parallel -j {netmhc_ncores}'
        F' && find {outtsv}.tmpdir/ -iname "SPLITTED.*.netMHCpan-result" | xargs cat > {outtsv}')

def peptide_to_pmhc_binding_stability(infaa, outtsv, hla_strs):
    #call_with_infolog(F'rm -r {outtsv}.tmpdir/ || true')
    #call_with_infolog(F'mkdir -p {outtsv}.tmpdir/')
    #call_with_infolog(F'''cat {infaa} | awk '{{print $1}}' |  split -l 20 - {outtsv}.tmpdir/SPLITTED.''')
    #cmds = [F'{netmhcpan_cmd} -f {outtsv}.tmpdir/{faafile} -a {hla_str} -l {prep_peplens} -BA > {outtsv}.tmpdir/{faafile}.{hla_str}.netMHCpan-result'
    #        for hla_str in hla_strs for faafile in os.listdir(F'{outtsv}.tmpdir/') if faafile.startswith('SPLITTED.')]
    #with open(F'{outtsv}.tmpdir/tmp.sh', 'w') as shfile:
    #    for cmd in cmds: shfile.write(cmd + '\n')
    # Each netmhcpan process uses much less than 100% CPU, so we can spawn many more processes
    #call_with_infolog(F'cat {outtsv}.tmpdir/tmp.sh | parallel -j {netmhc_ncores}'
    #    F' && find {outtsv}.tmpdir/ -iname "SPLITTED.*.netMHCpan-result" | xargs cat > {outtsv}')
    bindstab_filter_py = F'{script_basedir}/bindstab_filter.py'
    hla_string = ','.join(hla_strs)
    user, address, port, path = uri_to_user_address_port_path(netmhcstabpan_cmd)
    
    inputfile = infaa
    outdir = os.path.dirname(outtsv)
    logging.info(F'>>>>> outdir={outdir}')
    outputfile1 = outtsv
    
    if netmhcstabpan_cmd == path:
        run_calculation = F'python {bindstab_filter_py}             -i {inputfile}      -o {outputfile1}      -n {path} -c {netmhc_ncores} --peplens {prep_peplens}'
        call_with_infolog(run_calculation)
    else:
        remote_main_cmd = F'python {stab_tmp}/{outdir}/bindstab_filter.py -i {stab_tmp}/{inputfile} -o {stab_tmp}/{outputfile1} -n {path} -c {netmhc_ncores} --peplens {prep_peplens}'
        remote_rmdir = F' sshpass -p "$StabPanRemotePassword" ssh -p {port} {user}@{address} rm -r {stab_tmp}/{outdir}/ || true'
        remote_mkdir = F' sshpass -p "$StabPanRemotePassword" ssh -p {port} {user}@{address} mkdir -p {stab_tmp}/{outdir}/'
        remote_send = F' sshpass -p "$StabPanRemotePassword" scp -P {port} {bindstab_filter_py} {script_basedir}/fasta_partition.py {inputfile} {user}@{address}:{stab_tmp}/{outdir}/'
        # remote_main_cmd = F'python {stab_tmp}/{outdir}/bindstab_filter.py -i {stab_tmp}/{inputfile} -o {stab_tmp}/{outdir} -n {path} -b {binding_stability_filt_thres} -H {hla_string}
        remote_exe = F' sshpass -p "$StabPanRemotePassword" ssh -p {port} {user}@{address} {remote_main_cmd}'        
        remote_gzip = F' sshpass -p "$StabPanRemotePassword" ssh -p {port} {user}@{address} gzip --fast {stab_tmp}/{outputfile1} || true'
        remote_receive1 = F' sshpass -p "$StabPanRemotePassword" scp -P {port} {user}@{address}:{stab_tmp}/{outputfile1}.gz {outdir}' 
        call_with_infolog(remote_rmdir)
        call_with_infolog(remote_mkdir)
        call_with_infolog(remote_send)
        remote_exe_retcode = call_with_infolog(remote_exe)        
        call_with_infolog(remote_gzip)
        call_with_infolog(remote_receive1)
        call_with_infolog(F'gzip -d {outputfile1}.gz')

tumor_spec_peptide_fasta = F'{RES}/{PREFIX}_neo_peps.fasta'
if not isna(config.get('tumor_spec_peptide_fasta', NA_REP)): #tumor_spec_peptide_fasta = config['tumor_spec_peptide_fasta'] 
    make_dummy_files([tumor_spec_peptide_fasta], origfile=config['tumor_spec_peptide_fasta'], extension='.copied', is_refresh_allowed=True)
    
homologous_peptide_fasta = F'{pmhc_dir}/{PREFIX}_all_peps.fasta'
#if not isna(config.get('homologous_peptide_fasta', NA_REP)): homologous_peptide_fasta = config['homologous_peptide_fasta']

hetero_nbits = config.get('hetero_nbits', 0.75)
rule Peptide_preprocessing:
    input: dna_snvindel_neopeptide_fasta, # dna_snvindel_wt_peptide_fasta,
           rna_snvindel_neopeptide_fasta, # rna_snvindel_wt_peptide_fasta,
           fusion_neopeptide_fasta, splicing_neopeptide_fasta,
           hla_out # required for Peptide_processing but still listed here because tumor_spec_peptide_fasta that has pre-filled HLA key-value pairs in its FASTA header can be used as input to Peptide_processing
    output: tumor_spec_peptide_fasta
    threads: 1
    run:
        comma_sep_hla_str = ','.join(retrieve_hla_alleles())
        shell('cat {dna_snvindel_neopeptide_fasta} | python {script_basedir}/fasta_filter.py --tpm {tumor_abundance_filt_thres} '
            ' | python {script_basedir}/neoexpansion.py --nbits {hetero_nbits} > {dna_snvindel_neopeptide_fasta}.expansion')
        shell('cat'
            ' {dna_snvindel_neopeptide_fasta}.expansion ' # '{dna_snvindel_wt_peptide_fasta} '
            ' {rna_snvindel_neopeptide_fasta}           ' # '{rna_snvindel_wt_peptide_fasta} '
            ' {fusion_neopeptide_fasta} {splicing_neopeptide_fasta} '
            ' | python {script_basedir}/fasta_filter.py --tpm {tumor_abundance_filt_thres} '
            ' | python {script_basedir}/fasta_addkey.py --key HLA --val "{comma_sep_hla_str}" '
            ' > {tumor_spec_peptide_fasta}')

rule Peptide_processing:
    input: tumor_spec_peptide_fasta
    output: homologous_peptide_fasta
    threads: workflow.cores
    run:
        comma_sep_hla_str = ','.join(retrieve_hla_alleles())
        shell('cat {tumor_spec_peptide_fasta} '
            ' | python {script_basedir}/fasta_addkey.py --key HLA --val "{comma_sep_hla_str}" '
            ' | python {script_basedir}/neoexpansion.py --reference {PEP_REF} --tmp {homologous_peptide_fasta}.tmp > {homologous_peptide_fasta}')

homologous_netmhcpan_txt = F'{pmhc_dir}/{PREFIX}_all_peps.netmhcpan.txt'
rule PeptideMHC_binding_affinity_prediction:
    input: homologous_peptide_fasta #, hla_out
    output: homologous_netmhcpan_txt
    threads: netmhc_nthreads # 12
    run:
        shell(F'cat {homologous_peptide_fasta} | python {script_basedir}/fasta_partition.py --key HLA --out {homologous_peptide_fasta}')
        homologous_peptide_fasta_summary = F'{homologous_peptide_fasta}.partition/mapping.json'
        with open(homologous_peptide_fasta_summary) as jsonfile: hla2faa = json.load(jsonfile)
        homologous_peptide_fasta_partdir = os.path.dirname(os.path.realpath(homologous_peptide_fasta_summary))
        for hla_str, faa in sorted(hla2faa.items()):
            peptide_to_pmhc_binding_affinity(F'{homologous_peptide_fasta_partdir}/{faa}', F'{homologous_peptide_fasta_partdir}/{faa}.netmhcpan.txt', hla_str)
        shell(F'cat {homologous_peptide_fasta_partdir}/*.netmhcpan.txt > {homologous_netmhcpan_txt}')

homologous_netmhcstabpan_txt = F'{pmhc_dir}/{PREFIX}_all_peps.netmhcstabpan.txt'
rule PeptideMHC_binding_stability_prediction:
    input: homologous_peptide_fasta #, hla_out
    output: homologous_netmhcstabpan_txt
    threads: netmhc_nthreads # 12
    run:
        peptide_to_pmhc_binding_stability(homologous_peptide_fasta, homologous_netmhcstabpan_txt, retrieve_hla_alleles())

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
        remote_rmdir = F' sshpass -p "$StabPanRemotePassword" ssh -p {port} {user}@{address} rm -r {stab_tmp}/{outdir}/ || true'
        remote_mkdir = F' sshpass -p "$StabPanRemotePassword" ssh -p {port} {user}@{address} mkdir -p {stab_tmp}/{outdir}/'
        remote_send = F' sshpass -p "$StabPanRemotePassword" scp -P {port} {bindstab_filter_py} {inputfile} {user}@{address}:{stab_tmp}/{outdir}/'
        remote_main_cmd = F'python {stab_tmp}/{outdir}/bindstab_filter.py -i {stab_tmp}/{inputfile} -o {stab_tmp}/{outdir} -n {path} -b {binding_stability_filt_thres} -p {PREFIX}'
        remote_exe = F' sshpass -p "$StabPanRemotePassword" ssh -p {port} {user}@{address} {remote_main_cmd}'
        remote_receive1 = F' sshpass -p "$StabPanRemotePassword" scp -P {port} {user}@{address}:{stab_tmp}/{outputfile1} {outdir}'
        remote_receive2 = F' sshpass -p "$StabPanRemotePassword" scp -P {port} {user}@{address}:{stab_tmp}/{outputfile2} {outdir}'
        remote_gzip = F' sshpass -p "$StabPanRemotePassword" ssh -p {port} {user}@{address} gzip --fast {stab_tmp}/{outputfile1} {stab_tmp}/{outputfile2} || true'
        call_with_infolog(remote_rmdir)
        call_with_infolog(remote_mkdir)
        call_with_infolog(remote_send)
        remote_exe_retcode = call_with_infolog(remote_exe)
        call_with_infolog(remote_receive1)
        call_with_infolog(remote_receive2)
        call_with_infolog(remote_gzip)

homologous_netmhcpan_filtered_tsv = F'{prioritization_dir}/{PREFIX}_all_peps.netmhcpan_filtered.tsv'
#rule PeptideMHC_binding_affinity_filter:
#    input: homologous_peptide_fasta, homologous_netmhcpan_txt
#    output: homologous_netmhc_filtered_tsv
#    shell: 
#        'python {script_basedir}/parse_netmhcpan.py -f {homologous_peptide_fasta} -n {homologous_netmhcpan_txt} -o {homologous_netmhc_filtered_tsv} '
#        '-a {binding_affinity_filt_thres} -l SB,WB'

homologous_bindstab_raw_txt = F'{pmhc_dir}/{PREFIX}_bindstab_raw.txt'
homologous_bindstab_filtered_tsv = F'{pmhc_dir}/{PREFIX}_candidate_pmhc.tsv'

#rule disabled_PeptideMHC_binding_stability_prediction:
#    input: homologous_peptide_fasta, homologous_netmhcpan_txt
#    # homologous_netmhc_filtered_tsv
#    output: homologous_bindstab_raw_txt, homologous_bindstab_filtered_tsv
#    run: 
#        bindstab_filter_py = F'{script_basedir}/bindstab_filter.py'
#        shell('python {script_basedir}/parse_netmhcpan.py -f {homologous_peptide_fasta} -n {homologous_netmhcpan_txt} -o {homologous_netmhc_filtered_tsv} '
#              '-a {binding_affinity_filt_thres} -l {prep_peplens}')
#        run_netMHCstabpan(bindstab_filter_py, F'{homologous_netmhc_filtered_tsv}', F'{pmhc_dir}')

iedb_path = F'{script_basedir}/database/iedb.fasta' # '/mnt/d/code/neohunter/NeoHunter/database/iedb.fasta'

"""prioritization_thres_params = ' '.join([x.strip() for x in F'''
--binding-affinity-hard-thres {binding_affinity_hard_thres}
--binding-affinity-soft-thres {binding_affinity_soft_thres}
--binding-stability-hard-thres {binding_stability_hard_thres}
--binding-stability-soft-thres {binding_stability_soft_thres}
--tumor-abundance-hard-thres {tumor_abundance_hard_thres}
--tumor-abundance-soft-thres {tumor_abundance_soft_thres}
--tumor-abundance-recognition-thres {tumor_abundance_recognition_thres}
--agretopicity-thres {agretopicity_thres}
--foreignness-thres {foreignness_thres}
--alteration-type {alteration_type}
'''.strip().split()])"""

prioritization_function_params = ''

logging.debug(F'features_extracted_from_reads_tsv = {features_extracted_from_reads_tsv} (from {prioritization_dir})')

#ruleorder: PrioPrep_with_all_TCRs_from_reads > PrioPrep_with_all_TCRs_from_pMHCs

rule PrioPrep_with_all_TCRs_from_reads:
    input: iedb_path, homologous_netmhcpan_txt, homologous_netmhcstabpan_txt, # homologous_bindstab_filtered_tsv, # dna_tonly_raw_vcf, rna_tonly_raw_vcf, # dna_vcf, rna_vcf,
        dna_snvindel_info_file, rna_snvindel_info_file, fusion_info_file
        # splicing_info_file   
    output: features_extracted_from_reads_tsv #, final_pipeline_out
    run:
        shell('python {script_basedir}/parse_netmhcpan.py -f {homologous_peptide_fasta} -n {homologous_netmhcpan_txt} -o {homologous_netmhcpan_filtered_tsv} '
              '-a {binding_affinity_filt_thres} -l {prep_peplens}')
        if variantcaller == 'mutect2':
            call_with_infolog(F'python {script_basedir}/gather_results.py --netmhcstabpan-file {homologous_netmhcstabpan_txt} -i {homologous_netmhcpan_filtered_tsv} -I {iedb_path} '
            F' -D {dna_snvindel_info_file} -R {rna_snvindel_info_file} -F {fusion_info_file} '
            F' -o {features_extracted_from_reads_tsv} -t {alteration_type} ' #' --passflag 0x0 '
            F''' {prioritization_function_params.replace('_', '-')}''')
        else:
            call_with_infolog(F'python {script_basedir}/gather_results.py --netmhcstabpan-file {homologous_netmhcstabpan_txt} -i {homologous_netmhcpan_filtered_tsv} -I {iedb_path} '
            F' -D {dna_snvindel_info_file} -R {rna_snvindel_info_file} -F {fusion_info_file} ' # ' -S {splicing_info_file} '
            F' -o {features_extracted_from_reads_tsv} -t {alteration_type} ' # ' --passflag 0x0 '
            F' --dna-vcf {dna_tonly_raw_vcf} --rna-vcf {rna_tonly_raw_vcf} ' # ' --rna-depth {rna_tumor_depth_summary} '
            F''' {prioritization_function_params.replace('_', '-')}''')

rule PrioPrep_with_all_TCRs_from_pMHCs:
    input: iedb_path, homologous_netmhcpan_txt, homologous_netmhcstabpan_txt # homologous_bindstab_filtered_tsv 
    output: features_extracted_from_pmhcs_tsv
    run:
        shell('python {script_basedir}/parse_netmhcpan.py -f {homologous_peptide_fasta} -n {homologous_netmhcpan_txt} -o {homologous_netmhcpan_filtered_tsv} '
              '-a {binding_affinity_filt_thres} -l {prep_peplens}')
        call_with_infolog(F'python {script_basedir}/gather_results.py --netmhcstabpan-file {homologous_netmhcstabpan_txt} -i {homologous_netmhcpan_filtered_tsv} -I {iedb_path} '
            F' -o {features_extracted_from_pmhcs_tsv} -t {alteration_type} '
            F''' {prioritization_function_params.replace('_', '-')}''')

# This directory contains files generated by Validation_with_all_TCRs_from_pMHCs on data posted at https://figshare.com/s/147e67dde683fb769908 
# (cited by https://doi.org/10.1016/j.immuni.2023.09.002)
traindata_tsv = F'{script_basedir}/model/MullerNCItrain_.train-data.min.tsv'
trained_model = F'{script_basedir}/model/MullerNCItrain_.peplen_train_8to12.default.pickle'
kept_peplens = config.get('kept_peplens', prep_peplens)

rule Model_training:
    input: traindata_tsv
    output: trained_model
    run:
        shell('python {script_basedir}/neopredictor.py --model {trained_model} --peplens 8,9,10,11,12 --train {traindata_tsv}')

#features_tsv = ''
#if   IS_ANY_TUMOR_SEQ_DATA_AS_INPUT   : features_tsv = features_extracted_from_reads_tsv
#elif 'tumor_spec_peptide_fasta' in config: features_tsv = features_extracted_from_pmhcs_tsv
#else:
#    logging.critical('Either dna_tumor_fq1, rna_tumor_fq1 or tumor_spec_peptide_fasta must be specified in the config. ')
#    exit(-1)

rule Prioritization_with_all_TCRs_from_reads:
    input: features_extracted_from_reads_tsv, trained_model
    output: pipeline_out_from_reads
    run:
        shell('python {script_basedir}/neopredictor.py --model {trained_model} --peplens {kept_peplens} --test {features_extracted_from_reads_tsv} --suffix prioritized')
        shell('cp {features_extracted_from_reads_tsv}.prioritized {pipeline_out_from_reads}')

rule Prioritization_with_all_TCRs_from_pMHCs:
    input: features_extracted_from_pmhcs_tsv, trained_model
    output: pipeline_out_from_pmhcs
    run:
        shell('python {script_basedir}/neopredictor.py --model {trained_model} --peplens {kept_peplens} --test {features_extracted_from_pmhcs_tsv} --suffix prioritized')
        shell('cp {features_extracted_from_pmhcs_tsv}.prioritized {pipeline_out_from_pmhcs}')

### validation rules to construct ground truth that can be subsequently used for machine learning

reads_as_input_validation_out=F'{features_extracted_from_reads_tsv}.validation'
pmhcs_as_input_validation_out=F'{features_extracted_from_pmhcs_tsv}.validation'
reads_as_input_pep_valida_out=F'{features_extracted_from_reads_tsv}.peptide-validation'
pmhcs_as_input_pep_valida_out=F'{features_extracted_from_pmhcs_tsv}.peptide-validation'

rule Validation_with_all_TCRs_from_reads:
    input: features_extracted_from_reads_tsv
    output: reads_as_input_validation_out, reads_as_input_pep_valida_out
    run:
        truth_file = config.get('truth_file')
        truth_patientID = config.get('truth_patientID')
        call_with_infolog(F'python {script_basedir}/gather_results.py --netmhcstabpan-file - -i - -I - --truth-file {truth_file} --truth-patientID {truth_patientID} '
            F' -o {features_extracted_from_reads_tsv} -t {alteration_type}'
            F''' {prioritization_function_params.replace('_', '-')}''')

rule Validation_with_all_TCRs_from_pMHCs:
    input: features_extracted_from_pmhcs_tsv
    output: pmhcs_as_input_validation_out, pmhcs_as_input_pep_valida_out
    run:
        truth_file = config.get('truth_file')
        truth_patientID = config.get('truth_patientID')
        call_with_infolog(F'python {script_basedir}/gather_results.py --netmhcstabpan-file - -i - -I - --truth-file {truth_file} --truth-patientID {truth_patientID} '
            F' -o {features_extracted_from_pmhcs_tsv} -t {alteration_type}'
            F''' {prioritization_function_params.replace('_', '-')}''')

### auxiliary prioritization rules

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
    input: homologous_bindstab_filtered_tsv, mixcr_output_done_flag
    output: ergo2_score
    threads: ergo2_nthreads # 12
    shell: '''
        python {script_basedir}/rank_software_input.py -m {mixcr_output_pref} -n {homologous_bindstab_filtered_tsv} -o {prioritization_dir} -t {tcr_specificity_software} -p {PREFIX}
        cd {ERGO_EXE_DIR}
        python {ERGO_PATH} mcpas {prioritization_dir}/{PREFIX}_cdr_ergo.csv {ergo2_score}
        '''
rule Prioritization_with_each_TCR:
    input: homologous_bindstab_filtered_tsv, ergo2_score
    output: tcr_specificity_result
    threads: 1   
    shell: '''
        cd {script_basedir}
        python {script_basedir}/parse_rank_software.py -i {ergo2_score} -n {homologous_bindstab_filtered_tsv} -o {prioritization_dir}/ -t {tcr_specificity_software} -p {PREFIX}
        python {script_basedir}/add_detail_info.py -i {prioritization_dir}/{PREFIX}_neoantigen_rank_tcr_specificity.tsv -o {prioritization_dir}/ -p {PREFIX}
    '''

