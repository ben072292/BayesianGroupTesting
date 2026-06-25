#include "bgt/bgt.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <vector>

namespace nb = nanobind;

namespace
{
std::vector<bgt::host_probability_t> to_vector(nb::ndarray<const bgt::host_probability_t, nb::ndim<1>, nb::c_contig> values)
{
	return std::vector<bgt::host_probability_t>(values.data(), values.data() + values.shape(0));
}

std::vector<bgt::host_probability_t> sequence_to_vector(nb::handle values)
{
	std::vector<bgt::host_probability_t> ret;
	const size_t size = nb::len(values);
	ret.reserve(size);
	for (size_t i = 0; i < size; i++)
		ret.push_back(nb::cast<bgt::host_probability_t>(values[i]));
	return ret;
}
}

NB_MODULE(_core, m)
{
	nb::enum_<bgt::LatticeType>(m, "LatticeType")
		.value("distributed_non_dilution", bgt::LatticeType::distributed_non_dilution)
		.value("distributed_dilution", bgt::LatticeType::distributed_dilution)
		.value("replicated_non_dilution", bgt::LatticeType::replicated_non_dilution)
		.value("replicated_dilution", bgt::LatticeType::replicated_dilution);

	nb::enum_<bgt::SelectorType>(m, "SelectorType")
		.value("auto_select", bgt::SelectorType::auto_select)
		.value("op_bha", bgt::SelectorType::op_bha)
		.value("op_bbpa", bgt::SelectorType::op_bbpa);

	nb::enum_<bgt::Provider>(m, "Provider")
		.value("auto_select", bgt::Provider::auto_select)
		.value("cpu", bgt::Provider::cpu)
		.value("cuda", bgt::Provider::cuda);

	nb::enum_<bgt::DilutionMode>(m, "DilutionMode")
		.value("non_dilution", bgt::DilutionMode::non_dilution)
		.value("dilution", bgt::DilutionMode::dilution);

	nb::enum_<bgt::Precision>(m, "Precision")
		.value("float32", bgt::Precision::float32)
		.value("float64", bgt::Precision::float64);

	nb::enum_<bgt::OptimizationLevel>(m, "OptimizationLevel")
		.value("debug", bgt::OptimizationLevel::debug)
		.value("release", bgt::OptimizationLevel::release)
		.value("aggressive", bgt::OptimizationLevel::aggressive);

	nb::enum_<bgt::ProfileBackend>(m, "ProfileBackend")
		.value("none", bgt::ProfileBackend::none)
		.value("caliper", bgt::ProfileBackend::caliper);

	nb::exception<bgt::Error>(m, "Error", PyExc_RuntimeError);

	nb::enum_<bgt::StatusCode>(m, "StatusCode")
		.value("success", bgt::StatusCode::success)
		.value("invalid_argument", bgt::StatusCode::invalid_argument)
		.value("unsupported", bgt::StatusCode::unsupported)
		.value("runtime_error", bgt::StatusCode::runtime_error)
		.value("mpi_error", bgt::StatusCode::mpi_error)
		.value("cuda_error", bgt::StatusCode::cuda_error)
		.value("internal_error", bgt::StatusCode::internal_error);

	nb::enum_<bgt::LogLevel>(m, "LogLevel")
		.value("off", bgt::LogLevel::off)
		.value("fatal", bgt::LogLevel::fatal)
		.value("error", bgt::LogLevel::error)
		.value("warn", bgt::LogLevel::warn)
		.value("info", bgt::LogLevel::info)
		.value("debug", bgt::LogLevel::debug)
		.value("trace", bgt::LogLevel::trace);

	nb::enum_<bgt::LogSubsystem>(m, "LogSubsystem")
		.value("core", bgt::LogSubsystem::core)
		.value("model", bgt::LogSubsystem::model)
		.value("tree", bgt::LogSubsystem::tree)
		.value("runtime", bgt::LogSubsystem::runtime)
		.value("parallel", bgt::LogSubsystem::parallel)
		.value("cuda", bgt::LogSubsystem::cuda)
		.value("jit", bgt::LogSubsystem::jit)
		.value("perf", bgt::LogSubsystem::perf);

	nb::class_<bgt::LogConfig>(m, "LogConfig")
		.def(nb::init<>())
		.def_rw("level", &bgt::LogConfig::level)
		.def_rw("subsystems", &bgt::LogConfig::subsystems)
		.def_rw("rank_filter", &bgt::LogConfig::rank_filter)
		.def_rw("color", &bgt::LogConfig::color)
		.def_rw("timestamps", &bgt::LogConfig::timestamps)
		.def_prop_rw("file_path",
			[](const bgt::LogConfig &config) { return config.file_path.string(); },
			[](bgt::LogConfig &config, const std::string &value) { config.file_path = value; });

	nb::enum_<bgt::SimulationMode>(m, "SimulationMode")
		.value("local_tree", bgt::SimulationMode::local_tree)
		.value("parallel_global_tree", bgt::SimulationMode::parallel_global_tree)
		.value("parallel_dynamic_tree", bgt::SimulationMode::parallel_dynamic_tree)
		.value("parallel_hybrid_tree", bgt::SimulationMode::parallel_hybrid_tree)
		.value("parallel_partial_tree", bgt::SimulationMode::parallel_partial_tree)
		.value("parallel_fusion_tree", bgt::SimulationMode::parallel_fusion_tree);

	nb::enum_<bgt::TrueStatePolicy>(m, "TrueStatePolicy")
		.value("all", bgt::TrueStatePolicy::all)
		.value("trimmed", bgt::TrueStatePolicy::trimmed)
		.value("symmetric", bgt::TrueStatePolicy::symmetric);

	nb::enum_<bgt::SimulationRunStatus>(m, "SimulationRunStatus")
		.value("completed", bgt::SimulationRunStatus::completed)
		.value("interrupted", bgt::SimulationRunStatus::interrupted)
		.value("running", bgt::SimulationRunStatus::running);

	nb::enum_<bgt::TerminationReason>(m, "TerminationReason")
		.value("none", bgt::TerminationReason::none)
		.value("user_signal", bgt::TerminationReason::user_signal)
		.value("cancellation_requested", bgt::TerminationReason::cancellation_requested);

	nb::class_<bgt::TreeStats>(m, "TreeStats")
		.def_ro("total_leaves", &bgt::TreeStats::total_leaves)
		.def_ro("correct_probability", &bgt::TreeStats::correct_probability)
		.def_ro("incorrect_probability", &bgt::TreeStats::incorrect_probability)
		.def_ro("false_positive_probability", &bgt::TreeStats::false_positive_probability)
		.def_ro("false_negative_probability", &bgt::TreeStats::false_negative_probability)
		.def_ro("unclassified_probability", &bgt::TreeStats::unclassified_probability)
		.def_ro("expected_stages", &bgt::TreeStats::expected_stages)
		.def_ro("expected_tests", &bgt::TreeStats::expected_tests);

	nb::class_<bgt::CompileOptions>(m, "CompileOptions")
		.def(nb::init<>())
		.def_rw("enable_simd", &bgt::CompileOptions::enable_simd)
		.def_rw("enable_openmp", &bgt::CompileOptions::enable_openmp)
		.def_rw("enable_cuda", &bgt::CompileOptions::enable_cuda)
		.def_rw("enable_nccl", &bgt::CompileOptions::enable_nccl)
		.def_rw("enable_nccl_gin", &bgt::CompileOptions::enable_nccl_gin)
		.def_rw("fast_math", &bgt::CompileOptions::fast_math)
		.def_rw("native_cpu", &bgt::CompileOptions::native_cpu)
		.def_rw("posterior_precision", &bgt::CompileOptions::posterior_precision)
		.def_rw("accumulator_precision", &bgt::CompileOptions::accumulator_precision)
		.def_rw("optimization", &bgt::CompileOptions::optimization)
		.def_rw("profile_backend", &bgt::CompileOptions::profile_backend)
		.def_rw("extra_cxx_flags", &bgt::CompileOptions::extra_cxx_flags)
		.def_rw("extra_cuda_flags", &bgt::CompileOptions::extra_cuda_flags)
		.def_rw("definitions", &bgt::CompileOptions::definitions);

	nb::class_<bgt::SimulationOptions>(m, "SimulationOptions")
		.def(nb::init<>())
		.def_rw("search_depth", &bgt::SimulationOptions::search_depth)
		.def_rw("threshold_up", &bgt::SimulationOptions::threshold_up)
		.def_rw("threshold_lo", &bgt::SimulationOptions::threshold_lo)
		.def_rw("branch_threshold", &bgt::SimulationOptions::branch_threshold)
		.def_rw("selector", &bgt::SimulationOptions::selector)
		.def_rw("provider", &bgt::SimulationOptions::provider)
		.def_rw("compile_options", &bgt::SimulationOptions::compile_options);

	nb::class_<bgt::DilutionParameters>(m, "DilutionParameters")
		.def(nb::init<>())
		.def_rw("alpha", &bgt::DilutionParameters::alpha)
		.def_rw("h", &bgt::DilutionParameters::h);

	nb::class_<bgt::ReportConfig>(m, "ReportConfig")
		.def(nb::init<>())
		.def_rw("write_csv", &bgt::ReportConfig::write_csv)
		.def_prop_rw("output_directory",
			[](const bgt::ReportConfig &config) { return config.output_directory.string(); },
			[](bgt::ReportConfig &config, const std::string &value) { config.output_directory = value; })
		.def_rw("run_name", &bgt::ReportConfig::run_name);

	nb::class_<bgt::SimulationTimings>(m, "SimulationTimings")
		.def(nb::init<>())
		.def_ro("setup_seconds", &bgt::SimulationTimings::setup_seconds)
		.def_ro("tree_build_seconds", &bgt::SimulationTimings::tree_build_seconds)
		.def_ro("evaluation_seconds", &bgt::SimulationTimings::evaluation_seconds)
		.def_ro("total_seconds", &bgt::SimulationTimings::total_seconds);

	nb::class_<bgt::ProgressConfig>(m, "ProgressConfig")
		.def(nb::init<>())
		.def_rw("enabled", &bgt::ProgressConfig::enabled)
		.def_rw("interval_seconds", &bgt::ProgressConfig::interval_seconds)
		.def_prop_rw("output_path",
			[](const bgt::ProgressConfig &config) { return config.output_path.string(); },
			[](bgt::ProgressConfig &config, const std::string &value) { config.output_path = value; })
		.def_rw("write_jsonl", &bgt::ProgressConfig::write_jsonl)
		.def_rw("print_stderr", &bgt::ProgressConfig::print_stderr);

	nb::class_<bgt::ProgressSnapshot>(m, "ProgressSnapshot")
		.def(nb::init<>())
		.def_ro("status", &bgt::ProgressSnapshot::status)
		.def_ro("reason", &bgt::ProgressSnapshot::reason)
		.def_ro("total_states", &bgt::ProgressSnapshot::total_states)
		.def_ro("evaluated_states", &bgt::ProgressSnapshot::evaluated_states)
		.def_ro("state_coverage_fraction", &bgt::ProgressSnapshot::state_coverage_fraction)
		.def_ro("evaluated_prior_mass", &bgt::ProgressSnapshot::evaluated_prior_mass)
		.def_ro("stats", &bgt::ProgressSnapshot::stats)
		.def_ro("timings", &bgt::ProgressSnapshot::timings)
		.def_ro("message", &bgt::ProgressSnapshot::message);

	nb::class_<bgt::SimulationConfig>(m, "SimulationConfig")
		.def(nb::init<>())
		.def_rw("lattice_type", &bgt::SimulationConfig::lattice_type)
		.def_rw("subjects", &bgt::SimulationConfig::subjects)
		.def_rw("variants", &bgt::SimulationConfig::variants)
		.def_prop_rw("prior",
			[](const bgt::SimulationConfig &config) { return config.prior; },
			[](bgt::SimulationConfig &config, nb::handle values) { config.prior = sequence_to_vector(values); })
		.def_prop_rw("evaluation_prior",
			[](const bgt::SimulationConfig &config) {
				return config.evaluation_prior.value_or(std::vector<bgt::host_probability_t>{});
			},
			[](bgt::SimulationConfig &config, nb::handle values) {
				if (values.is_none())
					config.evaluation_prior.reset();
				else
					config.evaluation_prior = sequence_to_vector(values);
			})
		.def_rw("options", &bgt::SimulationConfig::options)
		.def_rw("mode", &bgt::SimulationConfig::mode)
		.def_rw("true_state_policy", &bgt::SimulationConfig::true_state_policy)
		.def_rw("global_tree_depth", &bgt::SimulationConfig::global_tree_depth)
		.def_rw("workload_granularity", &bgt::SimulationConfig::workload_granularity)
		.def_rw("trim_percent", &bgt::SimulationConfig::trim_percent)
		.def_rw("dilution", &bgt::SimulationConfig::dilution)
		.def_rw("report", &bgt::SimulationConfig::report)
		.def_rw("progress", &bgt::SimulationConfig::progress);

	nb::class_<bgt::KernelSpec>(m, "KernelSpec")
		.def(nb::init<>())
		.def_rw("provider", &bgt::KernelSpec::provider)
		.def_rw("subjects", &bgt::KernelSpec::subjects)
		.def_rw("variants", &bgt::KernelSpec::variants)
		.def_rw("state_bits", &bgt::KernelSpec::state_bits)
		.def_rw("dilution", &bgt::KernelSpec::dilution)
		.def_rw("selector", &bgt::KernelSpec::selector)
		.def_rw("precision", &bgt::KernelSpec::precision)
		.def_rw("accumulator_precision", &bgt::KernelSpec::accumulator_precision)
		.def_rw("cuda_arch", &bgt::KernelSpec::cuda_arch)
		.def_rw("compile_options", &bgt::KernelSpec::compile_options);

	nb::class_<bgt::JitCacheInfo>(m, "JitCacheInfo")
		.def_ro("memory_entries", &bgt::JitCacheInfo::memory_entries)
		.def_ro("disk_entries", &bgt::JitCacheInfo::disk_entries)
		.def_ro("disk_cache_directory", &bgt::JitCacheInfo::disk_cache_directory)
		.def_ro("jit_enabled", &bgt::JitCacheInfo::jit_enabled);

	nb::class_<bgt::SimulationResult>(m, "SimulationResult")
		.def_ro("stats", &bgt::SimulationResult::stats)
		.def_ro("kernel", &bgt::SimulationResult::kernel)
		.def_ro("mode", &bgt::SimulationResult::mode)
		.def_ro("provider", &bgt::SimulationResult::provider)
		.def_ro("rank", &bgt::SimulationResult::rank)
		.def_ro("world_size", &bgt::SimulationResult::world_size)
		.def_ro("total_states", &bgt::SimulationResult::total_states)
		.def_ro("evaluated_states", &bgt::SimulationResult::evaluated_states)
		.def_ro("report_path", &bgt::SimulationResult::report_path)
		.def_ro("timings", &bgt::SimulationResult::timings)
		.def_ro("status", &bgt::SimulationResult::status)
		.def_ro("termination_reason", &bgt::SimulationResult::termination_reason)
		.def_ro("evaluated_prior_mass", &bgt::SimulationResult::evaluated_prior_mass)
		.def_ro("progress_path", &bgt::SimulationResult::progress_path);

	nb::class_<bgt::DilutionTable>(m, "DilutionTable")
		.def(nb::init<int, bgt::host_probability_t, bgt::host_probability_t>())
		.def_prop_ro("subjects", &bgt::DilutionTable::subjects)
		.def("at", &bgt::DilutionTable::at);

	nb::class_<bgt::Lattice>(m, "Lattice")
		.def("__init__", [](bgt::Lattice *self, bgt::LatticeType type, int subjects,
							nb::ndarray<const bgt::host_probability_t, nb::ndim<1>, nb::c_contig> prior, int variants) {
			if (variants == 1)
			{
				new (self) bgt::Lattice(type, subjects, to_vector(prior));
			}
			else
			{
				new (self) bgt::Lattice(type, subjects, variants, to_vector(prior));
			}
		}, nb::arg("type"), nb::arg("subjects"), nb::arg("prior"), nb::arg("variants") = 1)
		.def_prop_ro("subjects", &bgt::Lattice::subjects)
		.def_prop_ro("variants", &bgt::Lattice::variants)
		.def_prop_ro("type", &bgt::Lattice::type)
		.def("posterior_probability", &bgt::Lattice::posterior_probability)
		.def("upset_probability_mass", &bgt::Lattice::upset_probability_mass)
		.def("select_experiment", &bgt::Lattice::select_experiment, nb::arg("selector") = bgt::SelectorType::auto_select)
		.def("select_experiments", &bgt::Lattice::select_experiments, nb::arg("k"), nb::arg("selector") = bgt::SelectorType::op_bha)
		.def("update", &bgt::Lattice::update, nb::arg("experiment"), nb::arg("response"), nb::arg("dilution") = nullptr)
		.def("update_classification", &bgt::Lattice::update_classification)
		.def("positive_classification_mask", &bgt::Lattice::positive_classification_mask)
		.def("negative_classification_mask", &bgt::Lattice::negative_classification_mask)
		.def("is_classified", &bgt::Lattice::is_classified);

	nb::class_<bgt::Runtime>(m, "Runtime")
		.def(nb::init<>())
		.def("available_providers", &bgt::Runtime::available_providers)
		.def("jit_cache_info", &bgt::Runtime::jit_cache_info)
		.def("clear_jit_cache", &bgt::Runtime::clear_jit_cache)
		.def("make_kernel_spec", &bgt::Runtime::make_kernel_spec)
		.def("run_simulation", &bgt::Runtime::run_simulation);

	m.def("cuda_available", &bgt::cuda_available);
	m.def("available_providers", &bgt::available_providers);
	m.def("run_simulation", &bgt::run_simulation);
	m.def("select_experiment", [](const bgt::Lattice &lattice, bgt::SelectorType selector) {
		return lattice.select_experiment(selector);
	}, nb::arg("lattice"), nb::arg("selector") = bgt::SelectorType::auto_select);
	m.def("update", [](bgt::Lattice &lattice, bgt::state_t experiment, bgt::state_t response,
					   const bgt::DilutionTable *dilution) {
		lattice.update(experiment, response, dilution);
	}, nb::arg("lattice"), nb::arg("experiment"), nb::arg("response"), nb::arg("dilution") = nullptr);
	m.def("jit_cache_info", [] { return bgt::Runtime{}.jit_cache_info(); });
	m.def("clear_jit_cache", [] { bgt::Runtime{}.clear_jit_cache(); });
	m.def("clear_simulation_cancellation", &bgt::clear_simulation_cancellation);
	m.def("request_simulation_cancellation", &bgt::request_simulation_cancellation,
		  nb::arg("reason") = bgt::TerminationReason::cancellation_requested);
	m.def("simulation_cancellation_requested", &bgt::simulation_cancellation_requested);
	m.def("simulation_cancellation_reason", &bgt::simulation_cancellation_reason);
	m.def("set_log_level", &bgt::set_log_level);
	m.def("set_log_config", &bgt::set_log_config);
	m.def("get_log_config", &bgt::get_log_config);
}
