import numpy as np
import pytest

from mv_laplace import MvLaplaceSampler


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
    sampler = MvLaplaceSampler(loc=loc, cov=cov)

    first = sampler.sample(sample_size=500, random_state=1234)
    second = sampler.sample(sample_size=500, random_state=1234)

    assert np.allclose(first, second)


def test_sampling_changes_with_different_seeds():
    loc = np.array([1.0, -1.0])
    cov = np.array([[2.0, 0.4], [0.4, 1.5]])
    sampler = MvLaplaceSampler(loc=loc, cov=cov)

    first = sampler.sample(sample_size=1000, random_state=1234)
    second = sampler.sample(sample_size=1000, random_state=5678)

    assert not np.allclose(first, second)


def test_sampling_reproducible_with_numpy_generator_state():
    loc = np.array([0.5, -0.25, 2.0])
    cov = np.array([[1.0, 0.1, 0.0], [0.1, 2.0, -0.2], [0.0, -0.2, 1.5]])
    sampler = MvLaplaceSampler(loc=loc, cov=cov)

    rng_a = np.random.default_rng(2026)
    rng_b = np.random.default_rng(2026)

    sample_a = sampler.sample(sample_size=(20, 5), random_state=rng_a)
    sample_b = sampler.sample(sample_size=(20, 5), random_state=rng_b)

    assert np.allclose(sample_a, sample_b)


def test_sample_without_random_state_still_returns_valid_samples():
    loc = np.array([1.0, 2.0])
    cov = np.array([[1.5, 0.3], [0.3, 0.7]])
    sampler = MvLaplaceSampler(loc=loc, cov=cov)

    sample = sampler.sample(sample_size=128)

    assert sample.shape == (128, 2)
    assert np.isfinite(sample).all()


def test_random_state_does_not_mutate_global_numpy_rng():
    loc = np.array([0.0, 1.0])
    cov = np.array([[1.0, 0.0], [0.0, 1.0]])
    sampler = MvLaplaceSampler(loc=loc, cov=cov)

    np.random.seed(24680)
    baseline = np.random.random(5)
    np.random.seed(24680)
    _ = sampler.sample(sample_size=32, random_state=123)
    after = np.random.random(5)

    assert np.allclose(after, baseline)
