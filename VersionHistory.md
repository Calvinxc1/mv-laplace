# Version History

## v0.2.1 (2026/03/03)
* Fixed release-note generation in `deploy-on-main-merge.yml` by escaping regex quantifier braces inside an f-string pattern.
* Restored strict SemVer branch policy for `release/*` and `hotfix/*` PRs to `main` in `version-integrity-main-pr.yml`.
* Restored hotfix patch-only release gate in `deploy-on-main-merge.yml` (`hotfix/*` merges must increment patch only).
* Tightened GitFlow PR branch-path checks to require SemVer suffixes for `release/<semver>` and `hotfix/<semver>` when targeting `main`.
* Updated policy/docs (`AGENTS.md`, `README.md`, `CONTRIBUTING.md`) to reflect required SemVer-suffixed release/hotfix branch naming.

## v0.2.0 (2026/03/03)
* Added `random_state` support to `MvLaplaceSampler.sample(...)` for reproducible sampling.
* Replaced global-seed test patterns with RNG-isolated tests and expanded sampling/reproducibility coverage.
* Split tests into focused modules and removed `tests/conftest.py` path injection.
* Added full PR/release workflow suite adapted from PertDist and required CI `build` smoke test checks.
* Enriched package/distribution metadata and public package surface (`__version__`, `__all__`, `py.typed`).
* Stopped tracking `uv.lock` in git and aligned repository policy/docs accordingly.

## v0.1.1 (2021/12/17)
* Renamed repo.
* Corrected `setup.py` classifiers.
* Added version history file/updated Readme w/ link to version history.
* Updated Python version minimum to 3.10.
* Added type hint return on MvLaplaceSampler.sampler method.

## v0.1.0 (2021/12/17)
* Initial Alpha Release.
