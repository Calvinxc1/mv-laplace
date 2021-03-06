# Multivariate Laplace Distribution
A sampler package from a Multivariate Laplace distrbrution.

## Background
I've been exploring Quantitative Finance for awhile now, and one frustrating point is the lack of a good sampler for a Multivariate Laplace distribution. As such I re-purposed a multivariate normal, normal, and Laplace distribution to fill the role of a multivariate Laplace. This isn't a fully implemented scipy-like class, similar to my [PertDist package](https://pypi.org/project/pertdist/), but just enough for me to spin up a custom MCMC routine and do some proper financial modeling.

## Installation
This package is on PyPi, and can be installed using `pip install mv-laplace`

## Usage
Unlike the scipy implementations this is based on, this distribution class provides sampling abilities only and no distributional parameters. The package is pretty straightforward to use:

```
from mv_laplace import MvLaplaceSampler

sampler = MvLaplaceSampler(loc, cov)
samples = sampler.sample(sample_size)
```

The input should be a pair of arrays, the first of which containing the location values (the means) in a vector of length M, and the second containing the covariance matrix in a MxM matrix. The return will be a NxM matrix, with N being the `sample_size` input on the sampler. I used [Wikipedia](https://en.wikipedia.org/wiki/Multivariate_Laplace_distribution) for basic reference on terminology.

## Disclaimer
I provide no guarentees that this package fits proper statistical robustness of how a multivariate Laplace is supposed to work, just that it seems to work for the use case I have identified. Use at your own risk.

## Roadmap
* Figure out how to calculate summary information, and refactor API to a scipi-like interface
* Add a Multivariate Asymmetric Laplace distribution class

## [Version History](./VersionHistory.md)

## License
This project uses the GNU General Public License.

Short version: Have fun and use it for whatever, just make sure to attribute me for it (-:
