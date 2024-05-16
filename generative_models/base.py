import pandas as pd
import dill

class GenerativeModel:
    """Base class for all generative models"""

    def __init__(self, epsilon, metadata, struct):
        """Initialize a generative model"""
        pass

    def fit(self, df):
        """Fit a generative model"""
        return NotImplementedError('Method needs to be overwritten.')

    def sample(self, n_synth):
        """Generate a synthetic dataset of size n_synth"""
        return NotImplementedError('Method needs to be overwritten.')
    
    def fit_and_sample(self, df, n_synth):
        """Fit and sample records in a single function"""
        self.fit(df)
        return self.sample(n_synth)
    
    def save_model(self, save_path):
        """Save fitted generative model"""
        dill.dump(self, open(save_path, 'wb'))

    def restore_model(self, metadata, save_path):
        """Restore fitted generative model"""
        # use metadata to generate dummy dataset (required to restore DPWGAN)
        dummy_record = {}
        for cdict in metadata['columns']:
            attr_name = cdict['name']
            # assume categorical
            dummy_record[attr_name] = cdict['i2s'][0]
        dummy_df = pd.DataFrame.from_records([dummy_record, dummy_record])

        # restore model
        self._restore_model(dummy_df, save_path)

    def _restore_model(self, dummy_df, save_path):
        """Restore fitted generative model"""
        model = dill.load(open(save_path, 'rb'))
        self.__dict__.update(model.__dict__)
    
    def get_struct(self):
        """Get structure of synthetic data model (for white-box attacks)"""
        return None
    
    def get_struct_enc(self):
        """Get encoding of structure (for white-box attacks)"""
        return []
    
    def get_raw_values(self):
        """Get raw values from synthetic data model"""
        return []