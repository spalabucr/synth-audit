"""
Run black-box and white-box attacks and analyze empirical epsilon
"""
import numpy as np
import random
import os
import argparse
import pandas as pd
from sklearn.metrics import roc_auc_score
import random

# sys path hack
import sys; sys.path.insert(0, '..')
from generative_models import Models, ModelNames, DPWGAN, DPWGANCity
from utils.audit import estimate_eps
from utils.exp import prep_exp, conv_to_cat
from utils.attack import run_attack_bb, run_attack_wb, train_clf_and_attack

# ignore warnings
import warnings
warnings.filterwarnings("ignore")

def get_scoress(model_name, df_in, df_out, metadata, n_shadow, n_valid, n_test, out_dir, delta=1e-5, attack_type='bb_querybased', cv_shuffle_id=None):
    target_record = conv_to_cat(df_in).iloc[[-1]]
    Model = ModelNames[model_name.split('_eps')[0]]

    idxs = list(range(n_shadow + n_valid + n_test))
    if cv_shuffle_id is not None:
        # 5-fold cross validation
        n_total = n_shadow + n_valid + n_test
        n_test = int(n_total * 1/5)
        n_valid = n_test
        n_shadow = n_total - n_valid - n_test

        start_test_idx, end_test_idx = n_test * cv_shuffle_id, n_test * (cv_shuffle_id + 1)
        test_idxs = idxs[start_test_idx:end_test_idx]
        del idxs[start_test_idx:end_test_idx]

        idxs = idxs + test_idxs

    if attack_type.startswith('bb_'):
        # black-box attack

        # load synthetic datasets
        synth_dfs_in, synth_dfs_out = [], []
        for i in idxs:
            dir_in = f'{out_dir}/df_in/{model_name}/model_{i}/'
            synth_df_in = pd.read_csv(f'{dir_in}/synth_df.csv.gz', compression='gzip')
            synth_dfs_in.append(conv_to_cat(synth_df_in))

            dir_out = f'{out_dir}/df_out/{model_name}/model_{i}/'
            synth_df_out = pd.read_csv(f'{dir_out}/synth_df.csv.gz', compression='gzip')
            synth_dfs_out.append(conv_to_cat(synth_df_out))

        scoress = run_attack_bb(synth_dfs_in, synth_dfs_out, n_shadow, metadata, target_record,
                                attack_type=attack_type.split('bb_')[1])
    elif attack_type.startswith('wb'):
        # white-box attack
        if Model in [DPWGAN, DPWGANCity]:
            # load LOGAN scores
            feats_in = [[np.genfromtxt(f'{out_dir}/df_in/{model_name}/model_{rep}/logan_score.txt').tolist()]
                        for rep in idxs]
            feats_out = [[np.genfromtxt(f'{out_dir}/df_out/{model_name}/model_{rep}/logan_score.txt').tolist()]
                         for rep in idxs]
            scoress = train_clf_and_attack(feats_in, feats_out, n_shadow)
        else:
            # load synthetic data models
            synth_models_in, synth_models_out = [], []
            for i in idxs:
                dir_in = f'{out_dir}/df_in/{model_name}/model_{i}/'
                model = Model(metadata=metadata)
                model.restore_model(metadata, f'{dir_in}/model.dill')
                synth_models_in.append(model)

                dir_out = f'{out_dir}/df_out/{model_name}/model_{i}/'
                model = Model(metadata=metadata)
                model.restore_model(metadata, f'{dir_out}/model.dill')
                synth_models_out.append(model)
            
            feat_type = None if '_' not in attack_type else attack_type.split('_')[1]
            scoress = run_attack_wb(synth_models_in, synth_models_out, Model, n_shadow, metadata, df_in, df_out,
                                    target_record, feat_type=feat_type)
    elif attack_type == 'active_wb':
        # active white-box attack
        # load nasr scores
        scores_in = [np.genfromtxt(f'{out_dir}/df_in/{model_name}/model_{rep}/nasr_score.txt').tolist()
                    for rep in idxs]
        scores_out = [np.genfromtxt(f'{out_dir}/df_out/{model_name}/model_{rep}/nasr_score.txt').tolist()
                    for rep in idxs]

        mia_scores = scores_in + scores_out
        mia_labels = [1] * len(scores_in) + [0] * len(scores_out)
        scoress = np.array([mia_scores, mia_labels]).T
    
    return scoress

def run_exp(model_name, df_in, df_out, metadata, n_shadow, n_valid, n_test, out_dir, delta=1e-5, attack_type='bb_querybased', cv_shuffle_id=None):
    if 'PB' in model_name:
        # PrivBayes is exact DP
        delta = 0

    result = {
        'model': model_name.split('_eps')[0],
        'attack_type': attack_type,
        'theor_eps': float(model_name.split('_eps')[1])
    }
    
    try:
        scoress = get_scoress(model_name, df_in, df_out, metadata, n_shadow, n_valid, n_test, out_dir, delta=delta, attack_type=attack_type, cv_shuffle_id=cv_shuffle_id)

        auc = roc_auc_score(scoress[:, 1], scoress[:, 0])
        
        emp_eps_gdp = estimate_eps(scoress, n_valid, alpha=0.1, delta=delta, method='GDP', n_procs=1)
        emp_eps_approxdp = estimate_eps(scoress, n_valid, alpha=0.1, delta=delta, method='cp', n_procs=1)
            
        emp_eps_gdp = emp_eps_gdp if emp_eps_gdp is not None else -1
        emp_eps_approxdp = emp_eps_approxdp if emp_eps_approxdp is not None else -1
        emp_eps = emp_eps_approxdp if result['model'] in ['DPartPB', 'DSynthPB'] else max(emp_eps_gdp, emp_eps_approxdp)
    except Exception:
        emp_eps_gdp = None
        emp_eps_approxdp = None
        emp_eps = None
        auc = None

    result['emp_eps'] = emp_eps
    result['auc'] = auc
    result['emp_eps_gdp'] = emp_eps_gdp
    result['emp_eps_approxdp'] = emp_eps_approxdp

    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, default='adult', choices=['adult', 'fire'], help='dataset to use')
    parser.add_argument('--neighbour', type=str, default='addremove', choices=['addremove', 'edit'], help='neighbouring dataset (add/remove or edit)')
    parser.add_argument('--worstcase', action='store_true', help='use small datasets for D- and D')
    parser.add_argument('--narrow', action='store_true', help='limit columns of D- and D to 3')
    parser.add_argument('--repeat_target', action='store_true', help='see prep_exp function in utils.py for more details')
    parser.add_argument('--target_idx', type=int, default=0, help='index of target to use (after sorting by vulnerability score)')
    parser.add_argument('--n_shadow', type=int, default=2000, help='number of models to use as shadow models')
    parser.add_argument('--n_valid', type=int, default=1000, help='number of models to use to choose optimal threshold (validation)')
    parser.add_argument('--n_test', type=int, default=2000, help='number of models to test attack on and estimate empirical epsilon')
    parser.add_argument('--model', type=str, default='DPartPB', choices=list(Models.keys()), help='model to fit synthetic data')
    parser.add_argument('--epsilon', type=float, default=None, help='privacy parameter (drop parameter if Non-DP)')
    parser.add_argument('--out_dir', type=str, default='exp_data/', help='path to folder containing synthetic data')
    parser.add_argument('--seed', type=int, default=0, help='seed for reproducibility')
    parser.add_argument('--attack_type', type=str, default='bb_querybased', help='attack type (bb_querybased, bb_dcr, wb, etc.)')
    args = parser.parse_args()

    # use half of models for D and the other half for D-
    args.n_shadow, args.n_valid, args.n_test = args.n_shadow // 2, args.n_valid // 2, args.n_test // 2

    # for reproducibility purposes
    np.random.seed(args.seed)
    random.seed(args.seed)

    out_dir = f'{args.out_dir}/{args.neighbour}'
    if args.worstcase:
        out_dir = f'{out_dir}_worstcase'
    if args.narrow:
        out_dir = f'{out_dir}_narrow'
    if args.repeat_target:
        out_dir = f'{out_dir}_repeat'
    os.makedirs(out_dir, exist_ok=True)
    df_in, df_out, metadata = prep_exp(data_name=args.data_name, neighbour=args.neighbour, worstcase=args.worstcase,
        narrow=args.narrow, repeat_target=args.repeat_target, target_idx=args.target_idx, out_dir=out_dir)

    Model = Models[args.model]
    model_name = f'{Model.__name__}_eps{args.epsilon}'
    
    results = run_exp(model_name, df_in, df_out, metadata, args.n_shadow, args.n_valid, args.n_test, out_dir, attack_type=args.attack_type)

    print(f'Theoretical epsilon: {args.epsilon}')
    print(f'Empirical epsilon: {results["emp_eps"]}')
    print(f'AUC: {results["auc"]}')