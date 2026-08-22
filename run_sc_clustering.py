#!/usr/bin/env python

import pandas as pd
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster
import numpy as np

import argparse
import json

# https://www.doubao.com/chat/37016436512096514
# https://sorryios.ai/c/69803e4b-06f4-8326-9e75-a15db9a5619e

def cluster_samples(input_file, percentile_cutoff=30):
    # Load the data
    data = pd.read_csv(input_file, sep='\t')  # Assuming tab-separated values

    # Remove non-numeric columns (CHR, START, END)
    data_numeric = data.iloc[:, 3:].apply(pd.to_numeric, errors='coerce')  # Skip the first three columns

    # Check if any column is empty after cleaning
    data_clean = data_numeric.dropna(axis=1, how='any')  # Drop columns with missing values
    
    # Remove columns with zero variance (causing issues in correlation calculation)
    data_clean = data_clean.loc[:, data_clean.var() > 0]  # Keep only columns with non-zero variance

    print('finished data_clean')
    # Compute the pairwise Pearson correlation coefficient and transform to distance (1 - PCC)
    dist_matrix = 1 - np.corrcoef(data_clean.T)

    print(dist_matrix)
    # Perform hierarchical clustering using the Ward method
    # Z = linkage(dist_matrix, method='ward')

    # Calculate the cutoff at the specified percentile of pairwise distances
    dist_values = pdist(data_clean.T,  'correlation')
    # dist_values = [y for x in dist_matrix for y in x]
    Z = linkage(dist_values, method='average') # https://chat.deepseek.com/a/chat/s/754fb359-6b8f-4540-96aa-bcc76e10a9aa
    cutoff = np.percentile(dist_values, percentile_cutoff)
    print(dist_values)
    print(f'cutoff={cutoff}')

    # Assign clusters based on the cutoff
    clusters = fcluster(Z, t=cutoff, criterion='distance')

    # Create the output dictionary with cluster names as keys
    cluster_dict = {}
    for idx, cluster_id in enumerate(clusters):
        cluster_id = str(cluster_id)
        sample_name = data_clean.columns[idx]
        if cluster_id not in cluster_dict:
            cluster_dict[cluster_id] = []
        cluster_dict[cluster_id].append(sample_name)

    # Convert the dictionary into the requested JSON format
    output_json = json.dumps(cluster_dict, indent=4)

    return output_json

parser = argparse.ArgumentParser(description='Read the Ginkgo output file SegCopy, and write to a cluster json file',
        formatter_class = argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-i', '--input',  required=True, type=str, help='The SegCopy input file')
parser.add_argument('-o', '--output', required=True, type=str, help='The cluster json output file')
parser.add_argument('-p', '--percentile_cutoff', required=False, type=float, default=25, help='The percentile cutoff of pairwise PCCs')

args = parser.parse_args()

output_json = cluster_samples(args.input, args.percentile_cutoff)

# Optionally save the JSON output to a file
with open(args.output, 'w') as f:
    f.write(output_json)

