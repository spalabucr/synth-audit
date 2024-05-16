"""
Class that encapsulates PrivBayes model from DataSynthesizer
https://github.com/DataResponsibly/DataSynthesizer
"""
from DataSynthesizer.DataDescriber import DataDescriber
from DataSynthesizer_v014.DataDescriber import DataDescriber as DataDescriber_v014     

import os
import uuid
import contextlib
import numpy as np
import json
import pandas as pd

from .base import GenerativeModel

class DSynthPB(GenerativeModel):
    def __init__(self, n_parents = 2, tmp_dir='tmp/', epsilon=None, curr_id=None, metadata=None, struct='infer', delta=None, version='latest'):
        self.n_parents = n_parents
        self.tmp_dir = tmp_dir
        self.epsilon = epsilon if epsilon is not None else 0
        self.struct = struct
        self.metadata = metadata
        self.version = version

        if curr_id is None:
            # sample random uuid
            self.curr_id = uuid.uuid4().hex[:8]
        else:
            self.curr_id = curr_id
        
        self.seed = np.random.randint(0, 2147483647)
    
    def get_struct(self):
        # get graph structure of model
        return self.describer.bayesian_network
    
    def get_struct_enc(self):
        network = self.get_struct()

        # get graph structure of model encoded as adjacency matrix
        attrs = [col['name'] for col in self.metadata['columns']]
        n_attrs = len(attrs)
        adj_matrix = np.zeros((n_attrs, n_attrs))

        for (to_attr, from_attrs) in network:
            to_idx = attrs.index(to_attr)
            for from_attr in from_attrs:
                from_idx = attrs.index(from_attr)
                adj_matrix[to_idx, from_idx] = 1
        
        return adj_matrix.flatten().tolist()
    
    def get_raw_values(self):
        # get raw values stored in model
        probss = []
        for _, probs in self.describer.data_description['conditional_probabilities'].items():
            if isinstance(probs, list):
                probss.extend(probs)
            else:
                for sub_probs in probs.values():
                    probss.extend(sub_probs)
        return probss

    def fit(self, df):
        # create temporary directory to store dataset files
        os.makedirs(self.tmp_dir, exist_ok=True)

        # input dataset
        input_data = f'{self.tmp_dir}/df_{self.curr_id}.csv'

        # save dataframe
        df.to_csv(input_data, index=False)

        # location of two output files
        self.description_file = f'{self.tmp_dir}/{self.curr_id}.json'

        # An attribute is categorical if its domain size is less than this threshold.
        # set so native-country (41 unique values) is categorical
        threshold_value = 42

        # The maximum number of parents in Bayesian network, i.e., the maximum number of incoming edges.
        degree_of_bayesian_network = self.n_parents

        # fit model to dataset
        if self.version == 'latest':
            self.describer = DataDescriber(category_threshold=threshold_value)
        else:
            self.describer = DataDescriber_v014(category_threshold=threshold_value)
        attr_to_is_categorical = {col: True for col in list(df.columns)}
        attr_to_is_candidate_key = {col: False for col in list(df.columns)}
        # block print statements
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            self.describer.describe_dataset_in_correlated_attribute_mode(dataset_file=input_data, 
                                                                    epsilon=self.epsilon, 
                                                                    k=degree_of_bayesian_network,
                                                                    bayesian_network=self.struct,
                                                                    attribute_to_is_categorical=attr_to_is_categorical,
                                                                    attribute_to_is_candidate_key=attr_to_is_candidate_key,
                                                                    seed=self.seed)
        self.describer.save_dataset_description_to_file(self.description_file)

        # delete temporary file
        os.remove(input_data)

    def sample(self, n_synth, remove_desc=True):
        with open(self.description_file) as f:
            model = json.load(f)
        
        visit_order = model['bayesian_network']
        root_attr = visit_order[0][1][0]
        cond_probs = DSynthPB.parse_cond_probs(model['conditional_probabilities'])

        samples = []
        for _ in range(n_synth):
            sample = dict()

            # sample root attribute from its 1-way marginal
            root_attr_dist = cond_probs[root_attr]
            sample[root_attr] = np.random.choice(len(root_attr_dist), p=root_attr_dist)
            
            for curr_attr, curr_parents in visit_order:
                # sample subsequent attributes from conditional distributions
                parent_vals = tuple([sample[parent] for parent in curr_parents])
                attr_dist = cond_probs[curr_attr][parent_vals]
                sample[curr_attr] = np.random.choice(len(attr_dist), p=attr_dist)
            
            samples.append(sample)
        
        synth_df = pd.DataFrame.from_records(samples)
        for col in synth_df.columns:
            synth_df[col] = synth_df[col].apply(lambda x: model['attribute_description'][col]['distribution_bins'][x])
        
        if remove_desc:
            # delete temporary file
            os.remove(self.description_file)

        synth_df = synth_df[model['meta']['all_attributes']]
        return synth_df

    @staticmethod
    def parse_cond_probs(cond_probs):
        '''
        Convert conditional probabilities from dict child -> dict '[parent values]' -> [probs]
        to dict child -> tuple (parent values) -> [probs]
        '''
        new_cond_probs = dict()
        for attr, cond_prob in cond_probs.items():
            if isinstance(cond_prob, list):
                new_cond_probs[attr] = cond_prob
                continue

            new_cond_prob = dict()
            for parent_vals, probs in cond_prob.items():
                new_cond_prob[tuple(eval(parent_vals))] = probs
            new_cond_probs[attr] = new_cond_prob
        
        return new_cond_probs

class DSynthPB_v014(DSynthPB):
    def __init__(self, **kwargs):
        super().__init__(version='old', **kwargs)
