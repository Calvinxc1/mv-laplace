import numpy as np

from mv_laplace import MvLaplaceSampler


def test_sample_shape_for_multiple_draws():
    loc = np.array([0.0, 1.0, -2.0])
    cov = np.array(
        [
            [1.0, 0.2, 0.1],
            [0.2, 2.0, -0.3],
            [0.1, -0.3, 1.5],
        ]
    )

    sampler = MvLaplaceSampler(loc=loc, cov=cov)
    samples = sampler.sample(sample_size=250)

    assert samples.shape == (250, 3)
    assert np.isfinite(samples).all()


def test_sample_without_size_returns_single_vector():
    loc = np.array([0.0, 0.5, 1.0])
    cov = np.eye(3)

    sampler = MvLaplaceSampler(loc=loc, cov=cov)
    sample = sampler.sample()

    assert sample.shape == (3,)
    assert np.isfinite(sample).all()


def test_marginal_mean_and_variance_match_targets_for_diagonal_cov():
    np.random.seed(42)

    loc = np.array([1.0, -2.0])
    cov = np.diag([4.0, 9.0])
    sampler = MvLaplaceSampler(loc=loc, cov=cov)

    samples = sampler.sample(sample_size=50000)

    sample_means = samples.mean(axis=0)
    sample_vars = samples.var(axis=0)

    # The construction targets Laplace marginals centered at loc with variances
    # equal to the diagonal covariance entries.
    assert np.allclose(sample_means, loc, atol=0.1)
    assert np.allclose(sample_vars, np.diag(cov), atol=0.3)
