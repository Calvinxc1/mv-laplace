import numpy as np

from mv_laplace import MvLaplaceSampler


def test_marginal_mean_and_variance_match_targets_for_diagonal_covariance():
    loc = np.array([1.0, -2.0])
    cov = np.diag([4.0, 9.0])
    sampler = MvLaplaceSampler(loc=loc, cov=cov)

    samples = sampler.sample(sample_size=40_000, random_state=42)

    sample_means = samples.mean(axis=0)
    sample_vars = samples.var(axis=0)

    assert np.allclose(sample_means, loc, atol=0.1)
    assert np.allclose(sample_vars, np.diag(cov), atol=0.3)


def test_off_diagonal_correlation_sign_is_preserved_in_samples():
    loc = np.array([0.0, 0.0])
    cov = np.array([[3.0, -1.2], [-1.2, 2.0]])
    sampler = MvLaplaceSampler(loc=loc, cov=cov)
    samples = sampler.sample(sample_size=25_000, random_state=7)

    empirical_corr = np.corrcoef(samples[:, 0], samples[:, 1])[0, 1]
    target_sign = np.sign(cov[0, 1])

    assert np.sign(empirical_corr) == target_sign
