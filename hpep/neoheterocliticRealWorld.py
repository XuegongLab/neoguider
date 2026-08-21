import pandas as pd

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

# Run the analysis
file_path = "/mnt/d/code/neoguider/testdata/hpeps_from_fastq/prioritization/SRR7890830-SRR7890845-SRR9134697_features_from_reads.tsv.expansion.untraced"

results = process_tsv(file_path)

print("\nResults:")
print(f"1. Rows with PRIME_BArank < 0.5: {results['rows_rank_under_05']}")
print(f"2. Unique MT_pep from above: {results['unique_mt_rank_under_05']}")
print(f"3. Rows with MT_pep == ET_pep AND PRIME_BArank < 0.5: {results['rows_mt_equals_et']}")
print(f"4. Unique MT_pep from above: {results['unique_mt_equals']}")

