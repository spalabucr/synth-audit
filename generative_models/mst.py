"""
Class that encapsulates MST model from Microsoft's SmartNoise library
https://docs.smartnoise.org/synth/synthesizers/mst.html
"""
from snsynth import Synthesizer
from .base import GenerativeModel
import numpy as np

class MST(GenerativeModel):
    def __init__(self, epsilon=None, delta=1e-5, metadata=None, struct='infer'):
        self.epsilon = float(epsilon) if epsilon is not None else 10000.0
        self.metadata = metadata
        if not isinstance(struct, dict):
            struct = { 'cliques': None, 'compress_cols': None } 
        self.synth_model = Synthesizer.create("mst", epsilon=self.epsilon, delta=delta, verbose=False, cliques=struct['cliques'], compress_cols=struct['compress_cols'])
    
    def get_struct(self):
        # get graph structure of model
        return { 'cliques': self.synth_model.cliques, 'compress_cols': self.synth_model.compress_cols }
    
    def get_struct_enc(self):
        # get graph structure of model encoded as adjacency matrix
        cliques = self.get_struct()['cliques']

        attrs = [col['name'] for col in self.metadata['columns']]
        n_attrs = len(attrs)
        adj_matrix = np.zeros((n_attrs, n_attrs))

        for cl in cliques:
            idx1 = int(cl[0].split('col')[1])
            idx2 = int(cl[1].split('col')[1])

            adj_matrix[idx1, idx2] = 1
            adj_matrix[idx2, idx1] = 1
        
        return adj_matrix.flatten().tolist()
    
    def get_raw_values(self):
        # get raw values stored in model
        vals = []
        for _, y, _, _ in self.synth_model.log:
            vals.extend(y.flatten().tolist())
        return vals

    def fit(self, df):
        self.synth_model.fit(df, preprocessor_eps=1/3 * self.epsilon)
    
    def sample(self, n_synth):
        return self.synth_model.sample(n_synth)