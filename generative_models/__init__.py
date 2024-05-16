from .dpart_pb import DPartPB # PrivBayes (DS)
from .dsynth_pb import DSynthPB, DSynthPB_v014 # PrivBayes (Hazy)
from .nist_mst import NIST_MST # MST (NIST)
from .mst import MST # MST (Smartnoise)
from .dp_wgan import DPWGAN # DPWGAN (NIST)
from .dpwgan_city import DPWGANCity # DPWGAN (SynthCity)

Models = {
    Model.__name__: Model
    for Model in [DPartPB, DSynthPB, NIST_MST, MST, DPWGAN, DPWGANCity, DSynthPB_v014]
}

ModelNames = {
    model_name: Model
    for model_name, Model in Models.items()
}