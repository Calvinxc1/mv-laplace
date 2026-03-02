import numpy as np
import pytest

import mv_laplace
from mv_laplace import MvLaplaceSampler


def test_package_exports_and_version():
    assert "MvLaplaceSampler" in mv_laplace.__all__
    assert mv_laplace.__version__ == "0.1.1"


def test_constructor_initializes_internal_distributions():
    loc = np.array([0.0, 1.0, -2.0])
    cov = np.array(
        [
            [1.0, 0.2, 0.1],
            [0.2, 2.0, -0.3],
            [0.1, -0.3, 1.5],
        ]
    )

    sampler = MvLaplaceSampler(loc=loc, cov=cov)

    assert np.array_equal(sampler.loc, loc)
    assert np.array_equal(sampler.cov, cov)
    assert np.allclose(sampler.var, np.diag(cov))
    assert np.allclose(sampler.mv_normal.mean, loc)
    assert np.allclose(sampler.mv_normal.cov, cov)
    assert np.allclose(sampler.normal.kwds["loc"], loc)
    assert np.allclose(sampler.normal.kwds["scale"], np.sqrt(np.diag(cov)))
    assert np.allclose(sampler.laplace.kwds["loc"], loc)
    assert np.allclose(sampler.laplace.kwds["scale"], np.sqrt(np.diag(cov) / 2.0))


@pytest.mark.parametrize(
    ("sample_size", "expected_shape"),
    [
        (None, (3,)),
        (1, (3,)),
        (250, (250, 3)),
        (0, (0, 3)),
        ((2, 4), (2, 4, 3)),
    ],
)
def test_sample_shape_and_finiteness(sample_size, expected_shape):
    loc = np.array([0.0, 0.5, 1.0])
    cov = np.eye(3)

    sampler = MvLaplaceSampler(loc=loc, cov=cov)
    sample = sampler.sample(sample_size=sample_size)

    assert sample.shape == expected_shape
    assert np.isfinite(sample).all()


def test_sample_returns_array_for_single_dimension_single_draw_cases():
    loc = np.array([0.0])
    cov = np.array([[1.0]])
    sampler = MvLaplaceSampler(loc=loc, cov=cov)

    for sample_size in (None, 1, (1,)):
        sample = sampler.sample(sample_size=sample_size)
        assert isinstance(sample, np.ndarray)
        assert sample.shape == (1,)


def test_sampling_is_reproducible_with_fixed_seed():
    loc = np.array([1.0, -1.0])
    cov = np.array([[2.0, 0.4], [0.4, 1.5]])

    np.random.seed(1234)
    first = MvLaplaceSampler(loc=loc, cov=cov).sample(sample_size=500)

    np.random.seed(1234)
    second = MvLaplaceSampler(loc=loc, cov=cov).sample(sample_size=500)

    assert np.allclose(first, second)


def test_constructor_raises_for_non_psd_covariance():
    loc = np.array([0.0, 1.0])
    cov = np.array([[1.0, 2.0], [2.0, 1.0]])

    with pytest.raises(ValueError, match="positive semidefinite"):
        MvLaplaceSampler(loc=loc, cov=cov)


@pytest.mark.parametrize(
    ("loc", "cov"),
    [
        (0.0, 1.0),
        ([0.0, 1.0], np.eye(3)),
        ([0.0, 1.0], np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])),
    ],
)
def test_constructor_invalid_shapes_raise_value_error(loc, cov):
    with pytest.raises(ValueError):
        MvLaplaceSampler(loc=loc, cov=cov)


def test_constructor_warns_for_non_finite_loc():
    loc = np.array([np.inf, 1.0])
    cov = np.eye(2)

    with pytest.warns(RuntimeWarning, match="Non-finite values detected in `loc`"):
        MvLaplaceSampler(loc=loc, cov=cov)


def test_marginal_mean_and_variance_match_targets_for_diagonal_covariance():
    np.random.seed(42)

    loc = np.array([1.0, -2.0])
    cov = np.diag([4.0, 9.0])
    sampler = MvLaplaceSampler(loc=loc, cov=cov)

    samples = sampler.sample(sample_size=40_000)

    sample_means = samples.mean(axis=0)
    sample_vars = samples.var(axis=0)

    assert np.allclose(sample_means, loc, atol=0.1)
    assert np.allclose(sample_vars, np.diag(cov), atol=0.3)


def test_off_diagonal_correlation_sign_is_preserved_in_samples():
    np.random.seed(7)

    loc = np.array([0.0, 0.0])
    cov = np.array([[3.0, -1.2], [-1.2, 2.0]])
    sampler = MvLaplaceSampler(loc=loc, cov=cov)
    samples = sampler.sample(sample_size=25_000)

    empirical_corr = np.corrcoef(samples[:, 0], samples[:, 1])[0, 1]
    target_sign = np.sign(cov[0, 1])

    assert np.sign(empirical_corr) == target_sign
