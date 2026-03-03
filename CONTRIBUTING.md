# Contributing to mv-laplace

This project uses GitFlow and PR-based development. See [`AGENTS.md`](AGENTS.md) for full repository policy and guardrails.

## Contribution Scope
- Community contributors are welcome to propose changes through `feature/* -> dev` pull requests.
- `release/* -> main` and `hotfix/* -> main` branches/pull requests are core-developer managed.

## Development Setup
Install development dependencies:

```bash
uv sync --all-extras --dev
```

Run local quality checks:

```bash
uv run ruff check .
uv run pytest -q
```

## Branch and PR Flow
- Start feature work from `dev`:
  - `feature/<name> -> dev`
- Releases and hotfixes are managed by core developers:
  - `release/<semver> -> main` (required SemVer branch suffix, e.g. `release/0.2.0`)
  - `hotfix/<semver> -> main` (required SemVer branch suffix, e.g. `hotfix/0.2.1`)

## Versioning and Release Notes
- Follow Semantic Versioning (`MAJOR.MINOR.PATCH`).
- Keep version metadata and [`VersionHistory.md`](VersionHistory.md) aligned with release changes.

## CI/CD Overview
- CI runs on pull requests to `dev` and `main`.
- Release dry-run checks run on `release/*` / `hotfix/*` pull requests to `main`.
- Release publish flow runs after merged `release/*` / `hotfix/*` pull requests to `main`.
- Recovery flows are manual.

## Issues
- Bug reports and feature requests: [GitHub Issues](https://github.com/Calvinxc1/mv-laplace/issues)
