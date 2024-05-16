import math
from opendp.transformations import make_bounded_sum, make_clamp
from opendp.mod import binary_search_param, enable_features
from opendp.measurements import make_base_discrete_laplace
from opendp.accuracy import laplacian_scale_to_accuracy
from opendp.typing import set_default_int_type
from .base import AdditiveNoiseMechanism, Mechanism

class DiscreteLaplace(AdditiveNoiseMechanism):
    def __init__(
            self, epsilon, *ignore, delta=0.0, sensitivity=None, max_contrib=1, upper=None, lower=None, **kwargs
        ):
        super().__init__(
                epsilon,
                mechanism=Mechanism.discrete_laplace,
                delta=0.0,  # ignored unless thresholding
                sensitivity=sensitivity,
                max_contrib=max_contrib,
                upper=upper,
                lower=lower
            )
        self._compute_noise_scale()
    def _compute_noise_scale(self):
        if self.scale is not None:
            return
        bit_depth = self.bit_depth
        set_default_int_type(f"i{bit_depth}")
        lower = self.lower
        upper = self.upper
        max_contrib = self.max_contrib
        bounds = (int(math.floor(lower)), int(math.ceil(upper)))

        rough_scale = (float(upper - lower) * max_contrib) / self.epsilon
        if rough_scale > 10_000_000:
            raise ValueError(f"Noise scale is too large using epsilon={self.epsilon} and bounds ({lower}, {upper}) with {self.mechanism}.  Try preprocessing to reduce senstivity, or try different privacy parameters.")

        enable_features('contrib')
        bounded_sum = (
            make_clamp(bounds=bounds) >>
            make_bounded_sum(bounds=bounds)
        )
        try:
            discovered_scale = binary_search_param(
                lambda s: bounded_sum >> make_base_discrete_laplace(scale=s),
                d_in=int(max_contrib),
                d_out=self.epsilon)
        except Exception as e:
            raise ValueError(f"Unable to find appropriate noise scale for {self.mechanism} with epsilon={self.epsilon} and bounds ({lower}, {upper}).  Try preprocessing to reduce senstivity, or try different privacy parameters.\n{e}")
        self.scale = discovered_scale
    @property
    def threshold(self):
        max_contrib = float(self.max_contrib)
        delta = self.delta
        epsilon = self.epsilon
        if delta == 0.0:
            raise ValueError("censor_dims requires delta to be > 0.0  Try delta=1/n*sqrt(n) where n is the number of individuals")
        log_term = math.log(2 * delta / max_contrib) 
        thresh = max_contrib * (1 - ( log_term / epsilon))
        return thresh
    def release(self, vals):
        enable_features('contrib')
        bit_depth = self.bit_depth
        set_default_int_type(f"i{bit_depth}")
        meas = make_base_discrete_laplace(self.scale)
        vals = [meas(int(round(v))) for v in vals]
        return vals
    def accuracy(self, alpha):
        bit_depth = self.bit_depth
        set_default_int_type(f"i{bit_depth}")
        return laplacian_scale_to_accuracy(self.scale, alpha)
