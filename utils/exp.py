"""
Utility functions for running experiments
"""
import numpy as np
import pandas as pd
import os
import dill
from tqdm import tqdm
import psutil
import random
import gc
import torch
import traceback

def conv_to_cat(df):
    # convert all attributes of dataset to categorical attributes
    new_df = df.copy()
    for col in df.columns:
        new_df[col] = new_df[col].astype(str).astype('category')
    return new_df

def get_metadata(df):
    # extract metadata from dataset (assume all attributes are categorical)
    df = conv_to_cat(df)
    return {
        'columns': [
            {
                'name': col,
                'type': 'Categorical',
                'i2s': list(df[col].unique())
            }
            for col in df.columns
        ]
    }

def fit_models(pid, Model, start_rep, n_reps, df, n_synth, out_dir, target_record, seed=None, struct=None,
               return_model=False, save_model_only=False, **kwargs):
    """Fit `n_model` models to the same dataset and generate synthetic data of size `n_synth`"""
    if pid is not None:
        # set processors to use
        p = psutil.Process()
        p.cpu_affinity([pid])

    if seed is not None:
        # set seed if set
        np.random.seed(seed)
        random.seed(seed)

    for rep_id in tqdm(range(n_reps), leave=False):
        while True:
            # sometimes MST fails, retry
            try:
                rep_dir = f'{out_dir}/model_{start_rep + rep_id}/'
                if out_dir is not None:
                    if os.path.exists(f'{rep_dir}/synth_df.csv.gz') and os.path.exists(f'{rep_dir}/model.dill'):
                        break

                    os.makedirs(rep_dir, exist_ok=True)

                # define model
                if struct is not None:
                    model = Model(struct=struct, **kwargs)
                else:
                    model = Model(**kwargs)
            
                model.fit(conv_to_cat(df))
                if return_model:
                    return model

                if not save_model_only:
                    # save synthetic data 
                    synth_df = model.sample(n_synth)
                    synth_df.to_csv(f'{rep_dir}/synth_df.csv.gz', index=False, compression='gzip')

                if Model.__name__ in ['DPWGAN', 'DPWGANCity']:
                    if 'audit_world' in kwargs and Model.__name__ == 'DPWGAN':
                        # calculate active white-box score
                        score = model.disc_obs
                        np.savetxt(f'{rep_dir}/nasr_score.txt', [score])
                    else:
                        # calculate white-box score (output of discriminator)
                        score = model.get_logan_score(target_record)
                        np.savetxt(f'{rep_dir}/logan_score.txt', [score])

                    # clear memory on GPU
                    del model
                    gc.collect()
                    torch.cuda.empty_cache()
                else:
                    # save model and run white-box attack later
                    model.save_model(f'{rep_dir}/model.dill')
            except Exception as e:
                print(f'Failed trying again: {str(e)}')
                if Model.__name__ in ['MST', 'NIST_MST']:
                    continue
                else:
                    traceback.print_exc() 
            break

# prepare 'in' and 'out' raw dataset
# depending on whether neighbouring datasets are 'add/remove' or 'edit'
def prep_exp(data_name='adult', neighbour='addremove', target_idx=0,
             worstcase=False, narrow=False, repeat_target=False, 
             nonvuln_target=False, out_dir='exp_data', root_dir='../'):
    # check if neighbouring datasets already created
    exp_path = f'{out_dir}/exp.dill'
    if os.path.exists(exp_path):
        exp = dill.load(open(exp_path, 'rb'))
        return exp['df_in'], exp['df_out'], exp['metadata']
    
    # load full data
    df = pd.read_csv(f'{root_dir}/datasets/{data_name}/{data_name}_cat.csv')
    vulns = np.genfromtxt(f'{root_dir}/datasets/{data_name}/vulns.txt')
    df['vuln'] = vulns

    if not nonvuln_target:
        # choose target record at index `target_idx` sorted by vulnerability
        df_target = df.drop_duplicates().sort_values('vuln', ascending=False)
        df = df.drop(columns='vuln')
        df_target = df_target.drop(columns='vuln')
        target_record = df_target.iloc[[target_idx]]

        # choose remaining records 
        df_rem = pd.concat([df, target_record]).drop_duplicates(keep=False)
    else:
        # drop vulnerability scores
        df = df.drop(columns='vuln')

        # choose remaining records first
        dmin = df.sample(n=999)

        # choose target record such that its attributes appear in D-
        df_target = pd.concat([dmin, df.drop_duplicates()]).drop_duplicates(keep=False)

        complete = False
        while not complete:
            target_records = df_target.sample(n=2)
            
            for col in df.columns:
                if set(target_records[col].to_numpy()) > set(dmin[col].to_numpy()):
                    # target record contains attribute that is not present in D-
                    complete = False
                    break
                else:
                    complete = True
        
        target_record = target_records.iloc[[0]]
        df_rem = pd.concat([df, target_records[:1]])

    if worstcase:
        # construct worst-case dataset D-

        # choose record that does not agree with target record on any column
        for col in df.columns:
            df_rem = df_rem[~(df_rem[col].isin(target_record[col].unique()))]
        
        x_record = df_rem.sample(n=1).to_dict('records')[0]
        x_T_record = target_record.to_dict('records')[0]

        # mix records together
        y_record, z_record = {}, {}
        for i, col in enumerate(x_record.keys()):
            y_record[col] = x_record[col] if i % 2 != 0 else x_T_record[col]
            z_record[col] = x_record[col] if i % 2 == 0 else x_T_record[col]
        
        # case-by-case
        if neighbour == 'addremove' and not repeat_target:
            df_rem = pd.DataFrame.from_records([x_record, x_T_record, z_record])
            target_record = pd.DataFrame.from_records([y_record])
        elif neighbour == 'addremove' and repeat_target:
            df_rem = pd.DataFrame.from_records([x_record, x_T_record, z_record])
        elif neighbour == 'edit' and not repeat_target:
            df_rem = pd.DataFrame.from_records([x_record, x_T_record, z_record])
            target_record = pd.DataFrame.from_records([y_record])
        elif neighbour == 'edit' and repeat_target:
            df_rem = pd.DataFrame.from_records([x_record, x_record, x_T_record])
    else:
        # sample randomly
        df_rem = df_rem.sample(n=1000)

    if repeat_target:
        # | D- | has 1 occurence of target record
        # | D | has 2 occurrences of target record
        if neighbour == 'addremove':
            df_out = pd.concat([df_rem[:-2], target_record])
            df_in = pd.concat([df_rem[:-2], target_record, target_record])
        elif neighbour == 'edit':
            df_out = pd.concat([df_rem[:-1], target_record])
            df_in = pd.concat([df_rem[:-2], target_record, target_record])
    else:
        # | D- | has 0 occurence of target record
        # | D | has 1 occurrence of target record
        if neighbour == 'addremove':
            df_out = df_rem[:-1]
            df_in = pd.concat([df_rem[:-1], target_record])
        elif neighbour == 'edit':
            df_out = df_rem
            df_in = pd.concat([df_rem[:-1], target_record])
    
    # if narrow, trim columns of dataset
    if narrow:
        tight_cols = list(df.columns)[:3]
        target_record = target_record[tight_cols]
        df_out = df_out[tight_cols]
        df_in = df_in[tight_cols]
        metadata = get_metadata(df[tight_cols])
    else:
        metadata = get_metadata(df)
    
    # save neighbouring datasets
    exp =  { 'df_in': df_in, 'df_out': df_out, 'metadata': metadata }
    os.makedirs(out_dir, exist_ok=True)
    dill.dump(exp, open(exp_path, 'wb'))

    return df_in, df_out, metadata