"""
    Author: Moustafa Alzantot (malzantot@ucla.edu)
    All rights reserved.
"""

import json
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

import numpy as np


# Methods to identify the distinct set of values for each column

def valid_values_from_metdata(metadata, feat_name):
    """ collect distinct set of values according to the JSON metada file """
    return {
        k: v for k, v in enumerate(list(range(metadata[feat_name]['maxval']+1)))}


def valid_values_from_groundtruth(gt_df, feat_name):
    """ collect distinct set of values according from the input data.
    WE DO this ONLY for Geographic (state-dependent columns)
    """
    return {
        v: k for k, v in enumerate(gt_df[feat_name].unique().tolist())}


def valid_values_from_list(l):
    """
    Define distinct set of values according to a list we collected from the
    codebook file.
    """
    return {k: v for v, k in enumerate(l)}


def preprocess_data(df, metadata, subsample=False):
    """
    Applies pre-processing to the input data.

    returns the following
    original_df: the input data in the original format.
    input_df: the input data after beeing formatted.
    metadata: the metadata dictionary.
    subsample: this argument is True only when we debug the model to work on samller datasets.
    """
    original_df = df
    if subsample:
        original_df = original_df.sample(1000)
    columns_list = original_df.columns.tolist()

    # Make it easy iso perform one hot encoding
    # Mappings areisbased on the codebook file.
    col_maps = {
        cdict['name']: {v: k for k, v in enumerate(cdict['i2s'])}
        for cdict in metadata['columns']
    }
    #select_cols = ['SPLIT', 'OWNERSHP', 'VETWWI','VALUEH', 'CITYPOP']
    #original_df = original_df[select_cols]
    #col_maps = {k:col_maps[k] for k in select_cols}
    output_df_columns = []
    for k in columns_list:
        assert k in columns_list, 'Cannot find pre-prepossing of column {}'.format(k)
        v = col_maps[k]
        if isinstance(v, dict):
            mapped_col = original_df[k].map(v)
            if (len(v) == 2):
                output_df_columns.append(mapped_col)
            else:
                mapped_vals = mapped_col.values.reshape((-1, 1))
                ohe_col = pd.DataFrame(
                    data=OneHotEncoder(categories=[list(range(len(v)))], sparse=False).fit_transform(mapped_vals).astype(np.float32), index=original_df.index)
                output_df_columns.append(ohe_col)
        elif v == 'void':
            pass  # skip that column
        elif v == 'int':
            output_df_columns.append(original_df[k] / 100.0)
        elif v == 'int_v':
            val_column = original_df[k].values.astype(np.float32)
            is_valid_column = (
                original_df[k] != metadata[k]['maxval']).astype(np.float32)
            val_column_processed = (is_valid_column) * val_column / 100.0
            output_df_columns.append(
                is_valid_column.rename('{}_valid'.format(k)))
            output_df_columns.append(val_column_processed)
        else:
            raise Exception('Invalid mapping')
    output_df = pd.concat(output_df_columns, axis=1)
    return original_df, output_df, metadata, col_maps, columns_list


def postprocess_data(input_data, metadata, col_maps, columns_list, greedy=True):
    """ Applies post-processing to the generator model outputs """
    output_df_columns = {}
    cur_idx = 0
    if isinstance(input_data, np.ndarray):
        input_data = pd.DataFrame(data=input_data)
    for k in columns_list:
        assert k in col_maps, 'Coloumn mapping not found'
        v = col_maps[k]
        if isinstance(v, dict):
            col_start = cur_idx
            if len(v) == 2:
                # binary column
                col_end = col_start + 1
                output_col = (
                    input_data.iloc[:, col_start] > 0.5).astype(np.int32)
                cur_idx += 1
            else:
                if greedy:
                    col_end = col_start + len(v)
                    col_ohe = input_data.iloc[:, col_start: col_end]
                    output_col = pd.Series(
                        data=np.argmax(
                            col_ohe.values, axis=1).astype(np.int32),
                        index=input_data.index)

                    cur_idx += len(v)
                else:
                    col_end = col_start + 1
                    output_col = input_data.iloc[:, col_start].astype(np.int32)
                    cur_idx += 1
            inv_map = {cv: ck for ck, cv in v.items()}
            output_col = output_col.map(inv_map)
            output_df_columns[k] = output_col.astype(str)
        elif v == 'int':
            val_col = (100.0 * input_data.iloc[:, cur_idx]).astype(np.int32)
            val_col = np.clip(val_col, 0, metadata[k]['maxval'])
            output_df_columns[k] = val_col
            cur_idx += 1
        elif v == 'int_v':
            is_valid_column = (input_data.iloc[:, cur_idx] > 0.5)
            value_column = (
                100.0 * input_data.iloc[:, cur_idx+1]).astype(np.int32)
            value_column = np.clip(value_column, 0, metadata[k]['maxval'])
            val_column_processed = (
                is_valid_column * value_column) + (1-is_valid_column) * metadata[k]['maxval']
            output_df_columns[k] = val_column_processed.astype(np.int32)
            cur_idx += 2
        elif v == 'void':
            if k == 'MBPLD':
                output_df_columns[k] = (
                    output_df_columns['MBPL'].values.reshape((-1,)) * 100).astype(np.int32)
            elif k == 'FBPLD':
                output_df_columns[k] = (
                    output_df_columns['FBPL'].values.reshape(-1,) * 100).astype(np.int32)
            elif k == 'BPLD':
                output_df_columns[k] = (
                    output_df_columns['BPL'].values.reshape((-1,)) * 100).astype(np.int32)
            else:
                raise Exception('Invalid mapping for column {}'.format(k))
        else:
            raise Exception("Invalid mapping for column {}". format(k))
    output_df = pd.DataFrame(output_df_columns)
    return output_df



if __name__ == '__main__':
    """ Only for testing """
    df = pd.read_csv('../../datasets/adult/adult_cat.csv').astype(str)
    with open('../../datasets/adult/adult_cat.json', 'r') as f:
        metadata = json.load(f)
    original_df, output_data, metadata, col_maps, columns_list = preprocess_data(
        df,
        metadata,
        subsample=True)
    output_df = postprocess_data(output_data, metadata, col_maps, columns_list, greedy=True)
    assert(output_df.shape == original_df.shape)
    match_count = 0
    for i in range(output_df.shape[0]):
        if pd.DataFrame.equals(output_df.iloc[i, :], original_df.iloc[i, :]):
            match_count += 1
    print('Match ratio : {}/ {}'.format(match_count, output_df.shape[0]))