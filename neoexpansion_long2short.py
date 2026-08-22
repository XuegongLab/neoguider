import argparse, sys

def main():
    parser = argparse.ArgumentParser(description = 'Experimental work (please do not use output sequences with MAX_BIT_DIST>0 for now). ',
            formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-l', '--minlen', type=int, help='Min length of the input  peptides below which cleavage is not performed on. ', default=12*2+1)
    parser.add_argument('-L', '--length', type=int, help='Max length of the output peptides to be cleaved into. ', default=12)
    args = parser.parse_args()
    hdr = None
    hdr_pep_list = []
    for line in sys.stdin:
        if line.startswith('>'):
            if hdr: hdr_pep_list.append((hdr, pep0))
            hdr = line.strip()
            pep0 = ''
        else:
            assert hdr, F'Encountered sequence before header in FASTA input!'
            pep0 += line.strip()
    if hdr: hdr_pep_list.append((hdr, pep0))
    #print(hdr_pep_list)
    for hdr, pep1 in hdr_pep_list:
        #pep2 = '*' * args.length + pep1 + '*' * args.length
        if len(pep1) <= args.minlen:
            print(F'{hdr}\n{pep1}')
            continue
        beg_min = 0         # args.length
        end_max = len(pep1) # len(pep2) - args.length
        span = args.length * 2 + 1
        for beg in range(-args.length, len(pep1)-args.length):
            end = beg + span
            subpep = pep1[max((beg,beg_min)):min((end,end_max))]
            subbeg = beg - max((beg,beg_min))
            new_ID = hdr.split()[0] + str(beg+args.length)
            pep_comment = ' '.join([tok for (j, tok) in enumerate(hdr.split()) if j > 0])
            skipped_positions = (list(range(subbeg, subbeg+args.length)) + list(range(subbeg+args.length+1, subbeg+args.length*2+1)))
            skipped_positions = ','.join([str(x) for x in skipped_positions])
            pep_comment += F' skipped_positions={skipped_positions}'
            print(F'{new_ID} {pep_comment} MT={subpep} WT={subpep}\n{subpep}')

if __name__ == '__main__': main()
 
