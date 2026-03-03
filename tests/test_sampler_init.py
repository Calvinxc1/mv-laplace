import numpy as np
import pytest

from mv_laplace import MvLaplaceSampler


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
