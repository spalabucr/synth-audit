import pandas as pd
import numpy as np
from tqdm import tqdm

# get one hot encoding
def onehot_encode(df, full_uniq_vals):
    new_cols = []
    for col in full_uniq_vals.keys():
        for uniq_val in full_uniq_vals[col]:
            new_cols.append((df[col] == uniq_val).rename(f'{col}_{uniq_val}'))
    
    return pd.concat(new_cols, axis='columns')

# get vulnerability of single record
def get_vuln_single(df_np, record_index, k):
    record = df_np[record_index]

    # calculate distance between record and other records
    distances = np.sqrt(np.sum(np.square(record - df_np), axis=1))
    distances.sort()

    # drop first distance because that will be the distance between the record and itself
    # calculate vulnerability score of record
    vuln = distances[1:k+1].mean()
    return vuln

# get vulnerability of a batch of records
def get_vuln_batch(df_np, record_inds, k):
    vulns = np.zeros(len(df_np))
    for record_ind in tqdm(record_inds, leave=False):
        record = df_np[int(record_ind)]

        # calculate distance between record and other records
        distances = np.sqrt(np.sum(np.square(record - df_np), axis=1))
        distances.sort()

        # drop first distance because that will be the distance between the record and itself
        # calculate vulnerability score of record
        vuln = distances[1:k+1].mean()

        vulns[record_ind] = vuln
    return vulns
    
# get vulnerability score of records
def get_vuln(df, full_uniq_vals, k=5, show_progress=False):
    vulns = np.zeros(len(df))

    # convert to onehot encoding
    df_np = onehot_encode(df, full_uniq_vals)
    df_np = df_np.astype(int).to_numpy()

    pbar = tqdm(range(len(df))) if show_progress else range(len(df))
    for i in pbar:
        vulns[i] = get_vuln_single(df_np, i, k)
    return vulns