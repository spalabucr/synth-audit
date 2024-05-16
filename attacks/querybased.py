import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from itertools import product

from .qbs import SimpleQBS

def preprocess_dataset(df, target_record):
    df = df.copy()
    for col in target_record.columns:
        df[col] = (df[col] == target_record[col].to_numpy()[0]).astype(int)
    return df

def init_qbs(dataset):
    tupled_dataset = list(dataset.itertuples(index=False, name=None))
    return SimpleQBS(tupled_dataset)

def get_queries(df_cols, n_queries=100000):
    if len(df_cols) < np.log2(n_queries):
        # enumerate all queries instead
        queries = list(product([0, 1], repeat=len(df_cols)))
    else:
        queries = np.random.randint(0, 2, size=((n_queries, len(df_cols))))
        queries = [tuple(query) for query in queries]
    return queries

def extract_query_counts(df, queries):
    qbs = init_qbs(df)
    features = qbs.query(queries, queries)
    return features

def extract_querybased(target_record, synth_df, queries):
    encoded = preprocess_dataset(synth_df, target_record)
    return extract_query_counts(encoded, queries)

def querybased_mia(target_record, queries, shadow_synth_dfs_in, shadow_synth_dfs_out, target_synth_df_in,
    target_synth_df_out, clf_type='RF'):
    # pre-process dataset
    for df in shadow_synth_dfs_in + shadow_synth_dfs_out + [target_synth_df_in] + [target_synth_df_out]:
        preprocess_dataset(df, target_record)
    
    # extract features
    feats, labels = [], []
    for df in shadow_synth_dfs_in:
        feats.append(extract_query_counts(df, queries))
        labels.append(1)

    for df in shadow_synth_dfs_out:
        feats.append(extract_query_counts(df, queries))
        labels.append(0)
    
    if clf_type == 'LR' or clf_type == 'MLP':
        n_synth = len(target_synth_df_in)
        feats = np.array(feats) / n_synth

        if clf_type == 'LR':
            clf = LogisticRegression()
        elif clf_type == 'MLP':
            clf = MLPClassifier()
    elif clf_type == 'RF':
        clf = RandomForestClassifier()

    clf.fit(feats, labels)

    feat_target_in = extract_query_counts(target_synth_df_in, queries)
    feat_target_out = extract_query_counts(target_synth_df_out, queries)

    if clf_type == 'LR' or clf_type == 'MLP':
        feat_target_in = np.array(feat_target_in) / n_synth
        feat_target_out = np.array(feat_target_out) / n_synth

    return clf.predict_proba([feat_target_in, feat_target_out])[:, 1]