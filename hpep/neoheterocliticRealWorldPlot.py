import argparse, collections, os, sys
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt

def process_tsv(file_path):
    # Initialize counters and sets
    count_rank_under_05 = 0
    unique_mt_pep_under_05 = set()
    count_mt_equals_et = 0
    unique_mt_pep_equals = set()
    
    # Process in chunks to handle large file
    chunk_size = 100000
    
    print("Processing file...")
    
    for chunk in pd.read_csv(file_path, sep='\t', chunksize=chunk_size):
        # Filter rows with PRIME_BArank < 0.5
        mask_rank = chunk['PRIME_BArank'] < 0.5
        filtered_chunk = chunk[mask_rank]
        
        # Update first set of counters
        count_rank_under_05 += len(filtered_chunk)
        unique_mt_pep_under_05.update(filtered_chunk['MT_pep'].dropna().unique())
        
        # Filter rows where MT_pep == ET_pep (within the already filtered rows)
        mask_equals = filtered_chunk['MT_pep'] == filtered_chunk['ET_pep']
        equals_chunk = filtered_chunk[mask_equals]
        
        # Update second set of counters
        count_mt_equals_et += len(equals_chunk)
        unique_mt_pep_equals.update(equals_chunk['MT_pep'].dropna().unique())
    
    return {
        'rows_rank_under_05': count_rank_under_05,
        'unique_mt_rank_under_05': len(unique_mt_pep_under_05),
        'rows_mt_equals_et': count_mt_equals_et,
        'unique_mt_equals': len(unique_mt_pep_equals)
    }

def preproess_2(file_path):
    rank_et_mt_list, rank_mt_list = [], []
    for chunk in pd.read_csv(file_path, sep='\t', chunksize=10*1000):
        rank_et_mt_list.extend(zip(chunk['PRIME_BArank'], chunk['ET_pep'], chunk['MT_pep']))
        chunk = chunk.loc[(chunk['ET_pep'] == chunk['MT_pep']), :]
        rank_mt_list.extend(zip(chunk['PRIME_BArank'], chunk['ET_pep'], chunk['MT_pep'])) 
    return rank_et_mt_list, rank_mt_list

def process_2(rank_et_mt_list):
    rank_et_mt_list = sorted(rank_et_mt_list)
    rank2ets = collections.defaultdict(set)
    rank2mts = collections.defaultdict(set)
    for rank, et, mt in rank_et_mt_list:
        rank2ets[rank].add(et)
        rank2mts[rank].add(mt)
    visited_ets = set()
    rank_ets_num_list, rank_mts_num_list = ([], [])
    for rank, ets in sorted(rank2ets.items()):
        visited_ets |= ets
        rank_ets_num_list.append((rank, len(visited_ets)))
    visited_mts = set()
    for rank, mts in sorted(rank2mts.items()):
        visited_mts |= mts
        rank_mts_num_list.append((rank, len(visited_mts)))
    return rank_ets_num_list, rank_mts_num_list

# Run the analysis
file_path = f"/mnt/d/code/neoguider/testdata/hpeps_from_fastq/prioritization/SRR7890830-SRR7890845-SRR9134697_features_from_reads.tsv.expansion.untraced"
file_outp = f'/mnt/d/code/neoguider/testdata/hpeps_from_fastq/prioritization/SRR7890830-SRR7890845-SRR9134697_features_from_reads.tsv.expansion.untraced.pdf'

parser = argparse.ArgumentParser(description=
        'This program takes as input an .tsv.expansion.untraced file and outputs a PDF file. '
        'The PDF file shows the unique peptide count (HT, MT, and HT-associated MT) passing a threshold as a function of the peptide-filtering threshold. ', 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-i', '--inputs',   help='Input files',   default=[file_path], nargs='+')
parser.add_argument('-o', '--output',   help='Output file',   default=file_outp)
parser.add_argument('-p', '--patients', help='Patient names', default=['N/A'], nargs='+')

args = parser.parse_args()

assert len(args.patients) == len(args.inputs), f'len(args.patients) == len(args.inputs) failed!'

#file_path = args.input
file_outp = args.output

fig, axes = plt.subplots(1, len(args.inputs), constrained_layout=True, figsize=(8, 5))

for i, file_path in enumerate(args.inputs):
    rank_et_mt_list, rank_mt_list = preproess_2(file_path)
    rank_et_ets_num_list, rank_et_mts_num_list = process_2(rank_et_mt_list)
    rank_mt_ets_num_list, rank_mt_mts_num_list = process_2(rank_mt_list)
    assert rank_mt_ets_num_list == rank_mt_mts_num_list
    ax = axes[i]
    ax.plot(*list(zip(*rank_et_ets_num_list)), label='Het')
    ax.plot(*list(zip(*rank_et_mts_num_list)), label='HetMut')
    ax.plot(*list(zip(*rank_mt_ets_num_list)), label='Mut')
    #ax.set_xlabel('PRIME_BArank threshold to filter peptides')
    #ax.set_ylabel('Number of unique peptides kept')
    ax.set_title(f'Patient {args.patients[i]}')
    ax.grid(True, alpha=0.25)
    ax.legend()
    ax.set_xscale('log')
    ax.set_yscale('log')
fig.supxlabel('PRIME_BArank threshold to filter peptides')
fig.supylabel('Number of unique peptides kept')
#plt.tight_layout()
plt.savefig(file_outp)
plt.savefig(file_outp + '.png', dpi=600)

#results = process_tsv(file_path)
#print("\nResults:")
#print(f"1. Rows with PRIME_BArank < 0.5: {results['rows_rank_under_05']}")
#print(f"2. Unique MT_pep from above: {results['unique_mt_rank_under_05']}")
#print(f"3. Rows with MT_pep == ET_pep AND PRIME_BArank < 0.5: {results['rows_mt_equals_et']}")
#print(f"4. Unique MT_pep from above: {results['unique_mt_equals']}")

