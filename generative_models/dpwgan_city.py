"""
Class that encapsulates DP-WGAN model from SynthCity library
https://github.com/vanderschaarlab/synthcity
"""
# synthcity absolute
from synthcity.plugins import Plugins
from synthcity.plugins.core.dataloader import GenericDataLoader
import torch
import numpy as np

from .base import GenerativeModel

class DPWGANCity(GenerativeModel):
    def __init__(self, epsilon=1, delta=1e-5, device=None, **kwargs):
        self.device = device
        if device is None:
            # set device to CUDA if available
            self.device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

        epsilon = float(epsilon) if epsilon is not None else 10000.0
        seed = np.random.randint(0, 2147483647)
        self.synth_model = Plugins().get('dpgan', epsilon=epsilon, delta=delta, n_iter=10, device=self.device,
                                         random_state=seed)
    
    def fit(self, df):
        self.loader = GenericDataLoader(
            df,
            target_column=list(df.columns)[-1],
            sensitive_columns=list(df.columns),
        )

        self.synth_model.fit(self.loader)

    def sample(self, n_synth):
        return self.synth_model.generate(count=n_synth).dataframe()
    
    @staticmethod
    def get_target_record_enc(df, device=None):
        model = DPWGANCity(epsilon=10, delta=1e-5, device=device)
        model.fit(df)

        # encode entire dataset
        df_enc = model.loader.encode()[0].dataframe()
        df_enc = model.synth_model.model.encode(df_enc).to_numpy()
        df_enc = torch.from_numpy(df_enc).to(model.device)
        cond = model.synth_model.model.model._original_cond
        df_enc = model.synth_model.model.model._append_optional_cond(df_enc, cond)

        # extract encoded target record
        target_record_enc = df_enc[[-1]]
        return target_record_enc
        
    def get_logan_score(self, target_record_enc):
        return self.synth_model.model.model.discriminator(target_record_enc).squeeze().float().cpu().item()
