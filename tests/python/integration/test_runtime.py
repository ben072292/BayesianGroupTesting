import numpy as np
import pytest

import bayesian_group_testing as bgt


pytestmark = pytest.mark.integration


def test_cpu_tree_smoke():
    config = bgt.SimulationConfig()
    config.lattice_type = bgt.LatticeType.replicated_non_dilution
    config.subjects = 1
    config.variants = 1
    config.prior = np.array([0.2], dtype=np.float64)
    config.options.provider = bgt.Provider.cpu

    result = bgt.run_simulation(config)

    assert result.mode == bgt.SimulationMode.local_tree
    assert result.provider == bgt.Provider.cpu
    assert result.stats.total_leaves == 2
    assert abs(result.stats.correct_probability - 0.792) < 1e-12


def test_multinomial_lattice_and_kernel_spec():
    lattice = bgt.Lattice(
        bgt.LatticeType.replicated_non_dilution,
        1,
        np.array([0.2, 0.4], dtype=np.float64),
        variants=2,
    )
    assert lattice.subjects == 1
    assert lattice.variants == 2
    assert lattice.type == bgt.LatticeType.replicated_non_dilution

    options = bgt.SimulationOptions()
    options.provider = bgt.Provider.cpu
    options.selector = bgt.SelectorType.op_bbpa
    config = bgt.SimulationConfig()
    config.lattice_type = bgt.LatticeType.replicated_non_dilution
    config.subjects = 1
    config.variants = 2
    config.prior = np.array([0.2, 0.4], dtype=np.float64)
    config.options = options

    spec = bgt.Runtime().make_kernel_spec(config)

    assert spec.provider == bgt.Provider.cpu
    assert spec.variants == 2
    assert spec.selector == bgt.SelectorType.op_bbpa
    assert spec.precision == config.options.compile_options.posterior_precision
    assert spec.accumulator_precision == config.options.compile_options.accumulator_precision

    config.options.compile_options.posterior_precision = bgt.Precision.float32
    config.options.compile_options.accumulator_precision = bgt.Precision.float32
    float32_spec = bgt.Runtime().make_kernel_spec(config)
    assert float32_spec.precision == bgt.Precision.float32
    assert float32_spec.accumulator_precision == bgt.Precision.float32


def test_auto_selector_resolves_by_lattice_shape():
    runtime = bgt.Runtime()

    binary = bgt.SimulationConfig()
    binary.subjects = 2
    binary.variants = 1
    binary.prior = [0.2, 0.4]
    assert runtime.make_kernel_spec(binary).selector == bgt.SelectorType.op_bha

    multinomial = bgt.SimulationConfig()
    multinomial.subjects = 1
    multinomial.variants = 2
    multinomial.prior = [0.2, 0.4]
    assert runtime.make_kernel_spec(multinomial).selector == bgt.SelectorType.op_bbpa


def test_report_config_and_result_fields(tmp_path):
    config = bgt.SimulationConfig()
    config.lattice_type = bgt.LatticeType.replicated_non_dilution
    config.subjects = 1
    config.variants = 1
    config.prior = [0.2]
    config.options.provider = bgt.Provider.cpu
    config.report.write_csv = True
    config.report.output_directory = str(tmp_path)
    config.report.run_name = "python-smoke"

    result = bgt.run_simulation(config)

    assert result.kernel.subjects == 1
    assert result.total_states == 2
    assert result.evaluated_states == 2
    assert result.status == bgt.SimulationRunStatus.completed
    assert result.termination_reason == bgt.TerminationReason.none
    assert abs(result.evaluated_prior_mass - 1.0) < 1e-12
    assert result.timings.total_seconds >= 0.0
    assert result.report_path.endswith("python-smoke.csv")
