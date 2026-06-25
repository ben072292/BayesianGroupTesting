import pytest

import bayesian_group_testing as bgt


pytestmark = pytest.mark.regression


def test_binary_single_subject_tree_regression():
    config = bgt.SimulationConfig()
    config.lattice_type = bgt.LatticeType.replicated_non_dilution
    config.subjects = 1
    config.variants = 1
    config.prior = [0.2]
    config.options.provider = bgt.Provider.cpu

    result = bgt.run_simulation(config)

    assert result.total_states == 2
    assert result.evaluated_states == 2
    assert result.stats.total_leaves == 2
    assert abs(result.stats.correct_probability - 0.792) < 1e-12
    assert abs(result.stats.incorrect_probability - 0.002) < 1e-12
    assert abs(result.stats.false_positive_probability - 0.0) < 1e-12
    assert abs(result.stats.false_negative_probability - 0.002) < 1e-12
    assert abs(result.stats.unclassified_probability - 0.206) < 1e-12
    assert abs(result.stats.expected_stages - 1.0) < 1e-12
    assert abs(result.stats.expected_tests - 1.0) < 1e-12
