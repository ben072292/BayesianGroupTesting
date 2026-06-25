import pytest


pytestmark = pytest.mark.e2e


def test_package_imports_from_installed_extension():
    import bayesian_group_testing as bgt

    assert bgt.Runtime is not None
    assert callable(bgt.run_simulation)
