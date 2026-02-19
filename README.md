# mv-laplace
A sampler-focused implementation of the [Multivariate Laplace distribution](https://en.wikipedia.org/wiki/Multivariate_Laplace_distribution).

## Motivation
I have been exploring quantitative finance for a while, and one recurring pain point is the lack of a practical sampler for a Multivariate Laplace distribution. To fill that gap, I repurposed multivariate normal, normal, and Laplace sampling components into a single workflow for Multivariate Laplace sampling.

Unlike scipy-style distribution classes, this package currently focuses on sampling rather than full distributional APIs. The goal is to provide enough functionality to support custom MCMC routines and practical financial modeling workflows.

## Installation
Installation is straightforward: `pip install mv-laplace`

## Code Example
Usage is intentionally simple:

```python
from mv_laplace import MvLaplaceSampler

sampler = MvLaplaceSampler(loc, cov)
samples = sampler.sample(sample_size)
```

Input should be:
- `loc`: a length-`M` location vector.
- `cov`: an `M x M` covariance matrix.

The returned samples are an `N x M` matrix, where `N` is `sample_size`.

## Disclaimer
I do not guarantee that this implementation fully matches every statistical expectation for a Multivariate Laplace distribution. It is built to solve the use case above and appears to perform well there. Use at your own risk.

## Roadmap
* Add summary-statistic methods and move toward a scipy-like API.
* Add a Multivariate Asymmetric Laplace distribution class.
* Add unit tests to better validate behavior across common input types.

A version history is located [here](./VersionHistory.md).

## Contributing
Contributions are welcome. Feel free to open an issue or submit a pull request with improvements, fixes, or ideas.

## License
This project uses the [GNU General Public License](./LICENSE).

Short version: Have fun and use it for whatever, just make sure to attribute me for it (-:
