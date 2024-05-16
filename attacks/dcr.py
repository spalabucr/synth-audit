import pandas as pd
import numpy as np

from .utils import onehot_encode

def extract_dcr(target_record, synth_df, full_uniq_vals):
    # preprocess synthetic data and target record
    synth_df = onehot_encode(synth_df, full_uniq_vals)
    synth_df = synth_df.astype(int).to_numpy()

    target_record = onehot_encode(target_record, full_uniq_vals)
    target_record = target_record.astype(int).to_numpy()[0]

    return -np.min(np.sqrt(np.sum(np.square(target_record - synth_df), axis=1)))

def dcr_mia(target_record, shadow_synth_dfs_in, shadow_synth_dfs_out, target_synth_df_in, target_synth_df_out,
    full_uniq_vals):
    # pre-process datasets
    target_synth_df_in = onehot_encode(target_synth_df_in, full_uniq_vals)
    target_synth_df_out = onehot_encode(target_synth_df_out, full_uniq_vals)
    target_record = onehot_encode(target_record, full_uniq_vals)

    # shadow datasets are unused
    record = target_record.astype(int).to_numpy()[0]

    synth_in = target_synth_df_in.astype(int).to_numpy()
    synth_out = target_synth_df_out.astype(int).to_numpy()

    score_in = -np.min(np.sqrt(np.sum(np.square(record - synth_in), axis=1)))
    score_out = -np.min(np.sqrt(np.sum(np.square(record - synth_out), axis=1)))

    return score_in, score_out