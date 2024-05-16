"""
Class that encapsulates PrivBayes model from dpart repo
https://github.com/hazy/dpart
"""
import numpy as np
from dpart.engines import PrivBayes

from .base import GenerativeModel

class DPartPB(GenerativeModel):
    def __init__(self, epsilon=None, bounds=None, metadata=None, n_parents=3, struct='infer', delta=None):
        # delta is unused
        self.metadata = metadata
        self.synth_model = PrivBayes(epsilon=epsilon, n_parents=n_parents, prediction_matrix=struct)
        if bounds is not None:
            self.synth_model.bounds = bounds
        
    def get_struct(self):
        # get graph structure of model
        return self.synth_model.dep_manager.prediction_matrix
    
    def get_struct_enc(self):
        # get graph structure of model encoded as adjacency matrix
        network = self.get_struct()

        attrs = [col['name'] for col in self.metadata['columns']]
        n_attrs = len(attrs)
        adj_matrix = np.zeros((n_attrs, n_attrs))

        for to_idx, to_attr in enumerate(attrs):
            from_attrs = network[to_attr]
            for from_attr in from_attrs:
                from_idx = attrs.index(from_attr)
                adj_matrix[to_idx, from_idx] = 1
        
        return adj_matrix.flatten().tolist()
    
    def get_raw_values(self):
        # get raw values stored in model
        probss = []
        for col in self.metadata['columns']:
            probss.extend(self.synth_model.methods[col['name']].conditional_dist.flatten().tolist())
        return probss

    def fit(self, df):
        self.synth_model.fit(df)
    
    def sample(self, n_synth):
        return  self.synth_model.sample(n_synth)