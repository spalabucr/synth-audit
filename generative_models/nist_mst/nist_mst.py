"""
Class that encapsulates NIST MST model that won the 2018 NIST DP Synthetic Data Challenge
https://github.com/usnistgov/PrivacyEngCollabSpace/tree/master/tools/de-identification/Differential-Privacy-Synthetic-Data-Challenge-Algorithms/rmckenna
"""

import numpy as np
from itertools import chain

from .nist_mst_utils import Match3
import sys; sys.path.insert(0, '../..')
from generative_models.base import GenerativeModel

class NIST_MST(GenerativeModel):
    def __init__(self, epsilon=None, delta=1e-9, metadata=None, struct='infer'):
        self.epsilon = float(epsilon) if epsilon is not None else epsilon
        self.delta = delta
        self.struct = struct

        if self.epsilon and self.epsilon <= 0.3:
            self.iters = 7500
            self.weight3 = 8.0
        elif self.epsilon and self.epsilon >= 4.0:
            self.iters = 10000
            self.weight3 = 4.0
        else:
            self.iters = 7500
            self.weight3 = 6.0
        self.warmup = False

        self.domain_info = {cdict['name']: cdict['i2s'] for cdict in metadata['columns'] }
        self.specs = {cdict['name']: { 'maxval': len(cdict['i2s']) - 1 } for cdict in metadata['columns'] }
    
    def get_struct(self):
        return self.synth_model.supports

    def get_struct_enc(self):
        return []
    
    def get_raw_values(self):
        return list(chain(*[y.flatten().tolist() for (_, y, _, _) in self.synth_model.measurements]))
    
    def fit(self, df):
        supports = self.struct if isinstance(self.struct, dict) else None
        self.synth_model = Match3(df, self.specs, self.domain_info, iters=self.iters, weight3=self.weight3,
            warmup=self.warmup, supports=supports, epsilon=self.epsilon, delta=self.delta, save=False)

        if isinstance(self.struct, dict):
            self.synth_model.supports = self.struct

        self.synth_model.setup()
        self.synth_model.measure()
    
    def sample(self, n_synth):
        self.synth_model.postprocess(n_synth)
        self.synth_model.transform_domain()
        return self.synth_model.synthetic