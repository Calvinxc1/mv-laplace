# mv-laplace
[![CI (main)](https://github.com/Calvinxc1/mv-laplace/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Calvinxc1/mv-laplace/actions/workflows/ci.yml?query=branch%3Amain)
[![PyPI version](https://img.shields.io/pypi/v/mv-laplace.svg)](https://pypi.org/project/mv-laplace/)

`mv-laplace` is a sampler-focused implementation of the [Multivariate Laplace distribution](https://en.wikipedia.org/wiki/Multivariate_Laplace_distribution).

## Overview
The package exposes a `MvLaplaceSampler` class for generating multivariate Laplace-like samples from a location vector and covariance matrix. It is intentionally focused on sampling and practical workflow support rather than a full SciPy-style distribution API.

## Installation
```bash
pip install mv-laplace
```

## Python Version
- `Python >= 3.11`

## Support
- Supported Python: `3.11+`
- Primary package index: [PyPI (`mv-laplace`)](https://pypi.org/project/mv-laplace/)

## Quick Start
```python
import numpy as np
from mv_laplace import MvLaplaceSampler

loc = np.array([0.0, 1.0, -2.0])
cov = np.array([
    [1.0, 0.2, 0.1],
    [0.2, 2.0, -0.3],
    [0.1, -0.3, 1.5],
])

sampler = MvLaplaceSampler(loc=loc, cov=cov)
samples = sampler.sample(sample_size=1000)
```

## Implemented API
### Constructor
- `MvLaplaceSampler(loc, cov)`

### Methods
- `sample(sample_size=None)`

## Input Expectations
- `loc` should be a length-`M` vector.
- `cov` should be an `M x M` covariance matrix.
- Input validation and most constructor error behavior are delegated to underlying SciPy distributions.
- `sample(sample_size=None)` returns:
  - shape `(M,)` when `sample_size` is `None`
  - shape `(N, M)` when `sample_size` is an integer `N`

## Testing
The project includes a pytest suite under [`tests/`](tests/), including coverage for:
- output shape behavior for single and batched sampling
- finite-value checks for generated samples
- empirical mean/variance checks for diagonal-covariance cases

Run tests with:
```bash
uv run pytest -q
```

## Development Setup
Install development dependencies:
```bash
uv sync --all-extras --dev
```

Run lint and tests:
```bash
uv run ruff check .
uv run pytest -q
```

## Roadmap
- Add richer input validation and explicit error messages.
- Add summary-statistic utilities and SciPy-style API extensions.
- Explore a Multivariate Asymmetric Laplace distribution class.

## Version History
See [`VersionHistory.md`](VersionHistory.md).

## Contributing
- Community and core-developer contribution workflow is documented in [`CONTRIBUTING.md`](CONTRIBUTING.md).
- Repository guardrails and policy details are defined in [`AGENTS.md`](AGENTS.md).

## Release Process (High-Level)
- Pull requests from `release/*` and `hotfix/*` into `main` run publish dry-run checks.
- Merged `release/*` / `hotfix/*` PRs to `main` trigger publish, tagging, release metadata, and post-release verification workflows.
- Recovery actions (including yank/unyank verification with runbook guidance) are run manually when needed.

## Reporting Issues
- Bug reports and feature requests: [GitHub Issues](https://github.com/Calvinxc1/mv-laplace/issues)
- Security-sensitive concerns can be reported privately using GitHub repository security reporting.

## Repository Policy
High-level development policy summary (full details in [`AGENTS.md`](AGENTS.md)):
- GitFlow is used: `feature/* -> dev`, `release/*|hotfix/* -> main`, with PR-based merges.
- Release and hotfix branches must use SemVer suffixes: `release/<MAJOR.MINOR.PATCH>`, `hotfix/<MAJOR.MINOR.PATCH>`.
- Community contributions are welcome through `feature/* -> dev` pull requests; `release/*` and `hotfix/*` flows are core-developer managed.
- CI runs on PRs to `dev` and `main`; release dry-runs run on `release/*`/`hotfix/*` PRs to `main`; release publish runs after merge to `main`.
- Semantic Versioning is required (`MAJOR.MINOR.PATCH`) and versioning must be intentional.
- Some defaults are guidance (for example draft PR by default) and developer discretion is explicitly supported.
- `uv.lock` is developer-local and is not tracked in this repository.
- Functional library code is authored manually.
- AI tooling may assist with test authoring, documentation drafting/editing, GitHub Actions/workflow authoring and maintenance, and development guidance for planning/decision support.

For release notes and historical change context, see [`VersionHistory.md`](VersionHistory.md).

## License
This project is licensed under the GNU GPL. See [`LICENSE`](LICENSE).
