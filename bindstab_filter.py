import argparse,csv,getopt,logging,multiprocessing,os,subprocess,sys
from time import sleep
try:
    from urllib.parse import urlparse
except ImportError:
    from urlparse import urlparse
    
def call_with_infolog(cmd):
    logging.info(cmd)
    os.system(cmd)
def main(args_input = sys.argv[1:]):
    logging.basicConfig(format=' bindstab_filter2.py %(asctime)s - %(message)s', level=logging.INFO)
    
    parser = argparse.ArgumentParser('This script runs netMHCstabpan to compute binding stability. ')
    parser.add_argument('-i', '--infaa',              help = 'Input peptide fasta file. ')
    parser.add_argument('-o', '--outtsv',             help = 'Output TSV file. ')
    parser.add_argument('-n', '--netMHCstabpan_path', help = 'The netMHCstabpan file path. ')
    parser.add_argument('-H', '--hla_strs',           help = 'String of the comma-separated HLA aleles. ')
    parser.add_argument('-p', '--peplens',            help = 'Peptide lengths. ', default = '8,9,10,11,12')
    parser.add_argument('-c', '--netmhcstab_ncores',  help = 'Number of cores to use. ', default = 24)
    
    args = parser.parse_args()
    
    infaa = args.infaa
    outtsv = args.outtsv
    netMHCstabpan_path = args.netMHCstabpan_path
    hla_strs = args.hla_strs.split(',')
    peplens = args.peplens
    netmhcstab_ncores = args.netmhcstab_ncores
    
    call_with_infolog('rm -r {}.tmpdir/ || true'.format(outtsv))
    call_with_infolog('mkdir -p {}.tmpdir/'.format(outtsv))
    call_with_infolog('''cat {} | awk '{{print $1}}' |  split -l 20 - {}.tmpdir/SPLITTED.'''.format(infaa, outtsv))
    cmds = ['{} -f {}.tmpdir/{} -a {} -l {} -ia > {}.tmpdir/{}.{}.netMHCstabpan-result'
            .format(netMHCstabpan_path, outtsv, faafile, hla_str, peplens, outtsv, faafile, hla_str)
            for hla_str in hla_strs for faafile in os.listdir('{}.tmpdir/'.format(outtsv)) if faafile.startswith('SPLITTED.')]
    with open('{}.tmpdir/tmp.sh'.format(outtsv), 'w') as shfile:
        for cmd in cmds: shfile.write(cmd + '\n')
    # Each netmhcpan process uses much less than 100% CPU, so we can spawn many more processes
    call_with_infolog('cat {}.tmpdir/tmp.sh | parallel -j {}'.format(outtsv, netmhcstab_ncores) + 
        ' && find {}.tmpdir/ -iname "SPLITTED.*.netMHCstabpan-result" | xargs cat > {}'.format(outtsv, outtsv))
    
if __name__ == '__main__': main()

