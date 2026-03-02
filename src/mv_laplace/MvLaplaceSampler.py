import numpy as np
from scipy import stats
from typing import Optional
from numpy.typing import ArrayLike, NDArray


class MvLaplaceSampler:
    """Sampler for a multivariate Laplace-like construction."""

    loc: NDArray[np.float64]
    cov: NDArray[np.float64]
    var: NDArray[np.float64]
    
    def __init__(self, loc: ArrayLike, cov: ArrayLike):
        self.loc = np.asarray(loc)
        self.cov = np.asarray(cov)
        self.var = np.diag(self.cov)

        self.mv_normal = stats.multivariate_normal(mean=self.loc, cov=self.cov)
        self.normal = stats.norm(loc=self.loc, scale=np.sqrt(self.var))
        self.laplace = stats.laplace(loc=self.loc, scale=np.sqrt(self.var / 2.0))

    def sample(self, sample_size: Optional[int | tuple[int, ...]] = None) -> NDArray:
        mv_samples = self.mv_normal.rvs(sample_size)
        cdf_samples = self.normal.cdf(mv_samples)
        laplace_samples = self.laplace.ppf(cdf_samples)
        return laplace_samples
