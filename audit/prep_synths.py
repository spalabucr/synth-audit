"""
Prepare target synthetic data to be attacked and shadow synthetic data for attacks to be trained on
"""
import numpy as np
import random
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # disable tensorflow warnings
from tqdm import tqdm
import concurrent.futures
import argparse
from opacus.accountants.utils import get_noise_multiplier

# sys path hack
import sys; sys.path.insert(0, '..')
from generative_models import Models
from utils.exp import conv_to_cat, prep_exp, fit_models

# ignore warnings
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, default='adult', choices=['adult', 'fire'], help='dataset to use')
    parser.add_argument('--n_synth', type=int, default=1000, help='number of records in synthetic dataset')
    parser.add_argument('--n_reps', type=int, default=5000, help='total number of models to build (shadow + validation + test)')
    parser.add_argument('--model', type=str, default='DPartPB', choices=list(Models.keys()), help='model to fit synthetic data')
    parser.add_argument('--epsilon', type=float, default=None, help='privacy parameter (drop parameter if Non-DP)')
    parser.add_argument('--delta', type=float, default=1e-5, help='privacy parameter (drop parameter if Non-DP)')
    parser.add_argument('--neighbour', type=str, default='addremove', choices=['addremove', 'edit'], help='neighbouring dataset (add/remove or edit)')
    parser.add_argument('--worstcase', action='store_true', help='use worstcase datasets for D and D\'')
    parser.add_argument('--narrow', action='store_true', help='limit columns of D- to 2')
    parser.add_argument('--repeat_target', action='store_true', help='target_record appears once in D\' but twice in D')
    parser.add_argument('--target_idx', type=int, default=0, help='index of target to use (after sorting by vulnerability score)')
    parser.add_argument('--use_provisional', action='store_true', help='use provisional dataset to get "structure" of data (for PrivBayes and MST only)')
    parser.add_argument('--save_model_only', action='store_true', help='only save models and not synthetic data')
    parser.add_argument('--active_wb', action='store_true', help='active white-box attack (for DPWGAN only)')
    parser.add_argument('--out_dir', type=str, default='exp_data/', help='path to folder containing synthetic data')
    parser.add_argument('--start_proc', type=int, default=0, help='processor number to start using')
    parser.add_argument('--n_procs', type=int, default=32, help='number of processes to parallelize')
    parser.add_argument('--seed', type=int, default=0, help='seed for reproducibility')
    parser.add_argument('--worstcase_hyperparam', action='store_true', help='use worstcase hyperparameters to induce bug for DPWGAN')
    parser.add_argument('--nonvuln_target', action='store_true', help='choose target such that all of its attributes have already appeared in D-')
    args = parser.parse_args()

    args.n_reps = args.n_reps // 2 # use half of reps to fit D and the other half to fit D-

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
    df_in, df_out, metadata = prep_exp(data_name=args.data_name, neighbour=args.neighbour, target_idx=args.target_idx,
                                       worstcase=args.worstcase, narrow=args.narrow, repeat_target=args.repeat_target,
                                       nonvuln_target=args.nonvuln_target, out_dir=out_dir)
    target_record = df_in.iloc[[-1]]

    Model = Models[args.model]
    model_name = f'{Model.__name__}_eps{args.epsilon}'

    struct = None
    if args.use_provisional and Model.__name__ not in ['DPWGAN', 'DPWGANCity']:
        # use provisional dataset to build bayesian network / select marginals to preserve
        warnings.filterwarnings("ignore") # ignore warnings
        sample_synth_model = Model(epsilon=args.epsilon, metadata=metadata)
        sample_synth_model.fit(conv_to_cat(df_in))
        struct = sample_synth_model.get_struct()

    if Model.__name__ == 'DPWGANCity':
        # get encoding of target record for LOGAN white-box attack
        target_record = Model.get_target_record_enc(df_in)
    
    if args.worstcase_hyperparam and Model.__name__ == 'DPWGAN':
        kwargs = {
            'critic_iters': 1,
            'lr': 1,
            'weight_clip': 1000,
            'batch_size': 1,
            'gradient_l2norm_bound': 2.0,
            'num_epochs': 2,
            'sigma': None
        }
    elif args.worstcase and Model.__name__ == 'DPWGAN':
        kwargs = {
            'critic_iters': 1,
            'lr': 1,
            'weight_clip': 1000,
            'batch_size': 3,
            'gradient_l2norm_bound': 2.0,
            'sigma': get_noise_multiplier(target_epsilon=args.epsilon, target_delta=args.delta, sample_rate=1,
                                          epochs=5 * 10)
        }
    else:
        kwargs = {}

    if args.n_procs == 1:
        for audit_world, curr_df in zip(['in', 'out'], [df_in, df_out]):
            if args.active_wb:
                kwargs['audit_world'] = audit_world

            curr_out_dir = f'{out_dir}/df_{audit_world}/{model_name}'
            os.makedirs(curr_out_dir, exist_ok=True)

            fit_models(0, Model, 0, args.n_reps, curr_df, args.n_synth, curr_out_dir, target_record, metadata=metadata,
                       epsilon=args.epsilon, delta=args.delta, struct=struct, save_model_only=args.save_model_only,
                       **kwargs)
    else:
        # split generation of synthetic data amongst processes
        reps_split = [(curr_reps[0], len(curr_reps))
                      for curr_reps in np.array_split(np.arange(args.n_reps), args.n_procs)]

        for audit_world, curr_df in zip(['in', 'out'], [df_in, df_out]):
            if args.active_wb:
                kwargs['audit_world'] = audit_world

            curr_out_dir = f'{out_dir}/df_{audit_world}/{model_name}'
            os.makedirs(curr_out_dir, exist_ok=True)

            with concurrent.futures.ProcessPoolExecutor(max_workers=args.n_procs) as executor, \
                tqdm(total=args.n_procs, desc=f'{audit_world}') as pbar:
                futures = []
                for i, (start_rep, n_reps) in enumerate(reps_split):
                    # sample seeds to prevent same model from being fitted twice
                    seed = np.random.randint(0, 2147483647) 
                    futures.append(executor.submit(fit_models, args.start_proc + i, Model, start_rep, n_reps, curr_df,
                        args.n_synth, curr_out_dir, target_record, seed=seed, metadata=metadata, epsilon=args.epsilon,
                        delta=args.delta, struct=struct, save_model_only=args.save_model_only, **kwargs))

                for _ in concurrent.futures.as_completed(futures):
                    pbar.update(1)