import mv_laplace


def test_package_exports_and_version():
    assert "MvLaplaceSampler" in mv_laplace.__all__
    assert mv_laplace.__version__ == "0.2.1"
