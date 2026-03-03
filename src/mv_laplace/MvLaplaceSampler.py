import warnings
import numpy as np
from scipy import stats
from typing import Optional
from numpy.typing import ArrayLike, NDArray
from numpy.random import Generator, RandomState


class MvLaplaceSampler:
    """Sampler for a multivariate Laplace-like construction."""

    loc: NDArray[np.float64]
    cov: NDArray[np.float64]
    var: NDArray[np.float64]
    
    def __init__(self, loc: ArrayLike, cov: ArrayLike):
        self.loc = np.asarray(loc, dtype=float)
        self.cov = np.asarray(cov, dtype=float)
        self.var = np.diag(self.cov)

        if not np.isfinite(self.loc).all():
            warnings.warn("Non-finite values detected in `loc`;"
                          " sampling may produce NaN/Inf outputs.",
                          RuntimeWarning, stacklevel=2)

        self.mv_normal = stats.multivariate_normal(mean=self.loc, cov=self.cov)
        self.normal = stats.norm(loc=self.loc, scale=np.sqrt(self.var))
        self.laplace = stats.laplace(loc=self.loc, scale=np.sqrt(self.var / 2.0))

    def sample(self,
               sample_size: Optional[int | tuple[int, ...]] = None,
               random_state: Optional[int | Generator | RandomState] = None,
               ) -> NDArray:
        mv_samples = self.mv_normal.rvs(sample_size, random_state=random_state)
        cdf_samples = self.normal.cdf(mv_samples)
        laplace_samples = self.laplace.ppf(cdf_samples)
        return laplace_samples
