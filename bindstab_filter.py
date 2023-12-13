import argparse,json,logging,os,sys

def call_with_infolog(cmd):
    logging.info(cmd)
    os.system(cmd)

def main(args_input = sys.argv[1:]):
    logging.basicConfig(format=' bindstab_filter2.py %(asctime)s - %(message)s', level=logging.INFO)
    
    parser = argparse.ArgumentParser('This script runs netMHCstabpan to compute binding stability. ')
    parser.add_argument('-i', '--infaa',              help = 'Input peptide fasta file. ')
    parser.add_argument('-o', '--outtsv',             help = 'Output TSV file. ')
    parser.add_argument('-n', '--netMHCstabpan_path', help = 'The netMHCstabpan file path. ')
    # parser.add_argument('-H', '--hla_strs',           help = 'String of the comma-separated HLA aleles. ')
    parser.add_argument('-p', '--peplens',            help = 'Peptide lengths. ', default = '8,9,10,11,12')
    parser.add_argument('-c', '--netmhcstab_ncores',  help = 'Number of cores to use. ', default = 24)
    
    args = parser.parse_args()
    
    infaa = args.infaa
    netMHCstabpan_path = args.netMHCstabpan_path
    # hla_strs = args.hla_strs.split(',')
    peplens = args.peplens
    netmhcstab_ncores = args.netmhcstab_ncores
    
    script_basedir = os.path.dirname(os.path.realpath(__file__))
    call_with_infolog('cat {} | python {}/fasta_partition.py --key HLA --out {}'.format(infaa, script_basedir, infaa))
    all_vars_peptide_faa_summary = '{}.partition/mapping.json'.format(infaa)
    with open(all_vars_peptide_faa_summary) as jsonfile: hla2faa = json.load(jsonfile)
    all_vars_peptide_faa_partdir = os.path.dirname(os.path.realpath(all_vars_peptide_faa_summary))
    for hla_str, faa in sorted(hla2faa.items()):
        # peptide_to_pmhc_binding_affinity(F'{all_vars_peptide_faa_partdir}/{faa}', F'{all_vars_peptide_faa_partdir}/{faa}.netmhcpan.txt', hla_str)
        outtsv = args.outtsv + '.subdir/' + faa + '.netmhcstabpan.txt'
        call_with_infolog('rm -r {}.tmpdir/ || true'.format(outtsv))
        call_with_infolog('mkdir -p {}.tmpdir/'.format(outtsv))
        
        call_with_infolog('''cat {} | awk '{{print $1}}' |  split -l 20 - {}.tmpdir/SPLITTED.'''.format(infaa, outtsv))
        cmds = ['{} -f {}.tmpdir/{} -a {} -l {} -ia > {}.tmpdir/{}.{}.netMHCstabpan-result'
                .format(netMHCstabpan_path, outtsv, faafile, hla_str, peplens, outtsv, faafile, hla_str)
                # for hla_str in hla_strs 
                for faafile in os.listdir('{}.tmpdir/'.format(outtsv)) if faafile.startswith('SPLITTED.')]
        with open('{}.tmpdir/tmp.sh'.format(outtsv), 'w') as shfile:
            for cmd in cmds: shfile.write(cmd + '\n')
        # Each netmhcpan process uses much less than 100% CPU, so we can spawn many more processes
        call_with_infolog('cat {}.tmpdir/tmp.sh | parallel -j {}'.format(outtsv, netmhcstab_ncores) + 
            ' && find {}.tmpdir/ -iname "SPLITTED.*.netMHCstabpan-result" | xargs cat > {}'.format(outtsv, outtsv))
    call_with_infolog('cat {}.subdir/*.netmhcstabpan.txt > {}'.format(args.outtsv, args.outtsv))
    # shell(F'cat {all_vars_peptide_faa_partdir}/*.netmhcpan.txt > {all_vars_netmhcpan_txt}')

if __name__ == '__main__': main()

