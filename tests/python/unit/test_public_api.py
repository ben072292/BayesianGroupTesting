import numpy as np
import pytest

import bayesian_group_testing as bgt


pytestmark = pytest.mark.unit


def test_public_exports_include_core_types():
    expected = {
        "CompileOptions",
        "KernelSpec",
        "Lattice",
        "LogConfig",
        "LogLevel",
        "LogSubsystem",
        "ProfileBackend",
        "Runtime",
        "ProgressConfig",
        "SimulationConfig",
        "SimulationMode",
        "SimulationRunStatus",
        "StatusCode",
        "TerminationReason",
        "clear_simulation_cancellation",
        "request_simulation_cancellation",
        "set_log_config",
        "set_log_level",
        "run_simulation",
    }
    assert expected.issubset(set(bgt.__all__))


def test_logging_config_round_trip():
    config = bgt.LogConfig()
    config.level = bgt.LogLevel.debug
    config.subsystems = [bgt.LogSubsystem.runtime]
    config.rank_filter = [0, 2]
    config.color = False
    config.timestamps = False

    bgt.set_log_config(config)
    actual = bgt.get_log_config()

    assert actual.level == bgt.LogLevel.debug
    assert actual.subsystems == [bgt.LogSubsystem.runtime]
    assert actual.rank_filter == [0, 2]
    assert actual.color is False
    assert actual.timestamps is False

    bgt.set_log_level(bgt.LogLevel.warn)
    assert bgt.get_log_config().level == bgt.LogLevel.warn


def test_status_and_error_exports_are_public():
    assert bgt.StatusCode.success.name == "success"
    assert issubclass(bgt.Error, RuntimeError)


def test_parallel_mode_names_are_framework_agnostic():
    mode_names = [
        bgt.SimulationMode.parallel_global_tree.name,
        bgt.SimulationMode.parallel_dynamic_tree.name,
        bgt.SimulationMode.parallel_hybrid_tree.name,
        bgt.SimulationMode.parallel_partial_tree.name,
        bgt.SimulationMode.parallel_fusion_tree.name,
    ]
    assert all(name.startswith("parallel_") for name in mode_names)


def test_config_accepts_numpy_and_lists():
    config = bgt.SimulationConfig()
    config.subjects = 1
    config.variants = 2
    config.prior = np.array([0.2, 0.4], dtype=np.float64)
    config.evaluation_prior = [0.3, 0.5]

    assert config.prior == [0.2, 0.4]
    assert config.evaluation_prior == [0.3, 0.5]


def test_progress_config_defaults_are_public():
    config = bgt.ProgressConfig()
    assert config.enabled is False
    assert config.interval_seconds == 5.0
    assert config.write_jsonl is True
    assert config.print_stderr is False


def test_lattice_top_level_helpers():
    lattice = bgt.Lattice(
        bgt.LatticeType.replicated_non_dilution,
        2,
        np.array([0.2, 0.4], dtype=np.float64),
    )

    experiment = bgt.select_experiment(lattice, bgt.SelectorType.op_bha)
    assert experiment == 3

    bgt.update(lattice, 1, 1)
    denominator = 0.60 * 0.99 + 0.40 * 0.01
    assert abs(lattice.posterior_probability(1) - 0.12 * 0.99 / denominator) < 1e-12
