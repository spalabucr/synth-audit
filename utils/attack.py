"""
Utility functions that switch between specific Black-box, White-box, and Active White-box attacks for each SDG
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import dill
import warnings

# sys path hack
import sys; sys.path.insert(0, '..')
from attacks.querybased import get_queries, extract_querybased
from attacks.dcr import extract_dcr
from utils.exp import fit_models

def train_clf_and_attack(feats_in, feats_out, n_shadow):
    # build classifier using first `n_shadow` feats and run inference on remaining feats

    # trim feats to `n_shadow` each for `in` and `out` worlds
    train_feats = np.concatenate([feats_in[:n_shadow], feats_out[:n_shadow]])
    train_labels = np.array([1] * len(feats_in[:n_shadow]) + [0] * len(feats_out[:n_shadow]))

    # train classifier on `n_shadow` feats each for `in` and `out` worlds
    clf = RandomForestClassifier()
    clf.fit(train_feats, train_labels)

    # run inference on remaining feats
    mia_scores = np.concatenate([clf.predict_proba(feats_in[n_shadow:])[:, 1],
                                 clf.predict_proba(feats_out[n_shadow:])[:, 1]])
    mia_labels = np.array([1] * len(feats_in[n_shadow:]) + [0] * len(feats_out[n_shadow:]))

    # return probabilities along with labels
    return np.array([mia_scores, mia_labels]).T


### Black-box ###
def extract_features_bb(synth_df, target_record, full_uniq_vals, queries=None, attack_type='querybased'):
    # extract black box features from synthetic dataset
    if attack_type == 'querybased':
        feat = extract_querybased(target_record, synth_df, queries)
    elif attack_type == 'dcr':
        feat = extract_dcr(target_record, synth_df, full_uniq_vals)
    
    return feat

def run_attack_bb(synth_dfs_in, synth_dfs_out, n_shadow, metadata, target_record, attack_type='querybased'):
    df_cols = list(synth_dfs_in[0].columns)
    full_uniq_vals = {cdict['name']: cdict['i2s'] for cdict in metadata['columns'] }

    # extract features
    queries = get_queries(df_cols) if attack_type == 'querybased' else None
    feats_in = [extract_features_bb(synth_df, target_record, full_uniq_vals, queries=queries, attack_type=attack_type)
                for synth_df in synth_dfs_in]
    feats_out = [extract_features_bb(synth_df, target_record, full_uniq_vals, queries=queries, attack_type=attack_type)
                for synth_df in synth_dfs_out]
    
    if attack_type in ['dcr', 'probest']:
        # features are already scores
        mia_scores = np.concatenate([feats_in[n_shadow:], feats_out[n_shadow:]])
        mia_labels = np.array([1] * len(feats_in[n_shadow:]) + [0] * len(feats_out[n_shadow:]))

        return np.array([mia_scores, mia_labels]).T
    else:
        # train classifier shadow features and run inference on remaining
        return train_clf_and_attack(feats_in, feats_out, n_shadow)

### White-box ###
def extract_features_wb(synth_model, exact_models_in, exact_models_out, feat_type=None):
    if feat_type is None:
        # set default feature type
        if synth_model.__class__.__name__ in ['DPartPB', 'DSynthPB']:
            feat_type = 'vals' # F_{naive}
        elif synth_model.__class__.__name__ in ['NIST_MST', 'MST']:
            feat_type = 'errors+sum' # F_{error}
        else:
            return []
    
    feats = []
    if 'network' in feat_type:
        feats = synth_model.get_struct_enc() 

    # calculate error in values extracted from synthetic model and corresponding values in the model w/o DP
    struct_str = dill.dumps(synth_model.get_struct())
    synth_vals = np.array(synth_model.get_raw_values())

    if 'errors' in feat_type:
        # calculate error to values from exact Model trained on df_in
        exact_model_in = exact_models_in[struct_str]
        exact_vals_in = np.array(exact_model_in.get_raw_values())
        if len(synth_vals) != len(exact_vals_in):
            exact_errors_in = np.ones_like(synth_vals) * -1000
        else:
            exact_errors_in = synth_vals - exact_vals_in

        # calculate error to values from exact Model trained on df_out
        exact_model_out = exact_models_out[struct_str]
        exact_vals_out = np.array(exact_model_out.get_raw_values())
        if len(synth_vals) != len(exact_vals_out):
            exact_errors_out = np.ones_like(synth_vals) * -1000
        else:
            exact_errors_out = synth_vals - exact_vals_out

        if 'sum' in feat_type:
            curr_errors = np.abs(exact_errors_in) if 'abs' in feat_type else exact_errors_in
            feats = feats + [np.sum(curr_errors)]
        else:
            feats = feats + exact_errors_in.tolist() + exact_errors_out.tolist()
    elif 'vals' in feat_type:
        feats = feats + synth_vals.tolist()
    
    return feats

def run_attack_wb(synth_models_in, synth_models_out, Model, n_shadow, metadata, df_in, df_out, target_record,
                  feat_type=None):
    warnings.filterwarnings("ignore")
    # build models for architectures encountered
    struct_strs = { dill.dumps(synth_model.get_struct()) for synth_model in synth_models_in + synth_models_out }
    exact_models_in = {struct: fit_models(None, Model, 0, 1, df_in, 0, None, None, struct=dill.loads(struct),
                                          metadata=metadata, return_model=True) for struct in struct_strs}
    exact_models_out = {struct: fit_models(None, Model, 0, 1, df_out, 0, None, None, struct=dill.loads(struct),
                                           metadata=metadata, return_model=True) for struct in struct_strs}

    # extract features
    feats_in = [extract_features_wb(synth_model, exact_models_in, exact_models_out, feat_type=feat_type)
                for synth_model in synth_models_in]
    feats_out = [extract_features_wb(synth_model, exact_models_in, exact_models_out, feat_type=feat_type)
                 for synth_model in synth_models_out]

    if len(feats_in[0]) == 1:
        # features is a single score
        scores_in = [feat[0] for feat in feats_in]
        labels_in = [1] * len(scores_in)
        scores_out = [feat[0] for feat in feats_out]
        labels_out = [0] * len(scores_out)

        mia_scores = scores_in + scores_out
        mia_labels = labels_in + labels_out
        scoress = np.array([mia_scores, mia_labels]).T
    else:
        # normalize all features to equal length
        max_feat_len = max({len(feat) for feat in feats_in + feats_out})
        feats_in = [feat + [0] * (max_feat_len - len(feat)) for feat in feats_in]
        feats_out = [feat + [0] * (max_feat_len - len(feat)) for feat in feats_out]

        scoress = train_clf_and_attack(feats_in, feats_out, n_shadow)

    return scoress