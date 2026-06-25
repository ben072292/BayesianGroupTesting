#include "bgt/detail/runtime/simulation_helpers.hpp"
#include "bgt/detail/model/lattice.hpp"
#include "bgt/detail/performance.hpp"
#include "bgt/detail/runtime/progress.hpp"

#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

using bgt::runtime_detail::Clock;
using bgt::runtime_detail::elapsed_seconds;
using bgt::runtime_detail::is_parallel_mode;
using bgt::runtime_detail::make_dilution_table;
using bgt::runtime_detail::lattice_type_is_distributed;
using bgt::runtime_detail::lattice_type_uses_dilution;
using bgt::runtime_detail::run_parallel_simulation;
using bgt::runtime_detail::resolve_selector;
using bgt::runtime_detail::selector_is_single_test;
using bgt::runtime_detail::validate_simulation_config;
using bgt::runtime_detail::write_report_if_requested;
using bgt::model::create_lattice;
using bgt::tree::LocalTree;
using bgt::tree::Tree;
using bgt::tree::TreeStatsAccumulator;

namespace bgt::runtime_detail
{

void validate_simulation_config(const SimulationConfig &config)
{
	if (config.subjects <= 0 || config.variants <= 0 || config.options.search_depth < 0)
		throw std::invalid_argument("invalid simulation dimensions.");
	if (config.workload_granularity <= 0)
		throw std::invalid_argument("workload_granularity must be positive.");
	if (config.progress.interval_seconds <= 0.0)
		throw std::invalid_argument("progress interval must be positive.");
	if (static_cast<int>(config.prior.size()) != config.subjects * config.variants)
		throw std::invalid_argument("prior length must equal subjects * variants.");
	if (config.evaluation_prior && static_cast<int>(config.evaluation_prior->size()) != config.subjects * config.variants)
		throw std::invalid_argument("evaluation_prior length must equal subjects * variants.");
	if (!selector_is_single_test(config.options.selector))
		throw std::invalid_argument("unsupported selector.");
	if (resolve_selector(config.options.selector, config.variants) == SelectorType::op_bha && config.variants != 1)
		throw std::invalid_argument("Op-BHA tree selection is only supported for binary lattices.");
	if (lattice_type_uses_dilution(config.lattice_type) &&
		(config.dilution.alpha <= 0.0 || config.dilution.h <= 0.0))
		throw std::invalid_argument("dilution parameters must be positive.");
	if (config.mode == SimulationMode::local_tree && lattice_type_is_distributed(config.lattice_type))
		throw std::invalid_argument("local_tree requires a replicated lattice.");
	if ((config.mode == SimulationMode::parallel_dynamic_tree || config.mode == SimulationMode::parallel_hybrid_tree) &&
		lattice_type_is_distributed(config.lattice_type))
		throw std::invalid_argument("dynamic and hybrid parallel tree modes currently require replicated lattices.");
	if (config.mode == SimulationMode::parallel_partial_tree)
	{
		if (!lattice_type_is_distributed(config.lattice_type))
			throw std::invalid_argument("partial parallel tree mode requires a distributed lattice.");
		if (config.true_state_policy == TrueStatePolicy::all)
			throw std::invalid_argument("partial parallel tree mode requires trimmed or symmetric true_state_policy.");
	}
	if (is_parallel_mode(config.mode))
	{
		int initialized = 0;
		BGT_MPI_CHECK(MPI_Initialized(&initialized));
		if (!initialized)
			throw std::runtime_error("parallel simulation modes require the communication backend to be initialized by the caller.");
	}
}

std::vector<host_probability_t> evaluation_prior(const SimulationConfig &config)
{
	return config.evaluation_prior.value_or(config.prior);
}

std::unique_ptr<DilutionTable> make_dilution_table(const SimulationConfig &config)
{
	if (!lattice_type_uses_dilution(config.lattice_type))
		return nullptr;
	return std::make_unique<DilutionTable>(config.subjects, config.dilution.alpha, config.dilution.h);
}

void configure_tree_globals(const SimulationConfig &config, const DilutionTable *dilution)
{
	Tree::search_depth(config.options.search_depth);
	Tree::thres_up(config.options.threshold_up);
	Tree::thres_lo(config.options.threshold_lo);
	Tree::thres_branch(config.options.branch_threshold);
	Tree::selector(resolve_selector(config.options.selector, config.variants));
	Tree::dilution(dilution == nullptr ? nullptr : const_cast<DilutionTable *>(dilution)->rows());
}

TreeStats run_local_cpu_tree(const SimulationConfig &config, const DilutionTable *dilution,
							 ProgressTracker &progress)
{
	const lattice_type_t internal_type = to_internal_lattice_type(config.lattice_type);
	std::vector<host_probability_t> prior_holder(config.prior.begin(), config.prior.end());
	auto original_lattice = create_lattice(internal_type, config.subjects, config.variants, prior_holder.data());
	auto tree_lattice = create_lattice(internal_type, config.subjects, config.variants, prior_holder.data());
	configure_tree_globals(config, dilution);

	LocalTree tree(std::move(tree_lattice), 0, 0, 1, 0);
	std::vector<const Tree *> leaves;
	Tree::find_all_leaves(&tree, &leaves);

	TreeStats stats;
	clear_tree_stats(stats);
	stats.total_leaves = static_cast<int>(leaves.size());
	progress.set_total_leaves(stats.total_leaves);

	const int total_states = state_count(config.subjects * config.variants);
	for (int true_state = 0; true_state < total_states; true_state++)
	{
		if (progress.should_stop())
			break;
		TreeStatsAccumulator state_stat(config.options.search_depth, 1);
		const auto encoded_state = static_cast<state_t>(true_state);
		tree.apply_true_state(original_lattice.get(), encoded_state);
		tree.parse(encoded_state, original_lattice.get(), 1.0, &state_stat);
		const host_probability_t prior_mass = original_lattice->prior_prob(encoded_state, model::Lattice::pi0());
		progress.record_state(state_stat, prior_mass);
		progress.maybe_emit("evaluating true states");
	}

	stats.correct_probability = progress.stats().correct_probability;
	stats.incorrect_probability = progress.stats().incorrect_probability;
	stats.false_positive_probability = progress.stats().false_positive_probability;
	stats.false_negative_probability = progress.stats().false_negative_probability;
	stats.unclassified_probability = progress.stats().unclassified_probability;
	stats.expected_stages = progress.stats().expected_stages;
	stats.expected_tests = progress.stats().expected_tests;
	return stats;
}

}

namespace bgt
{

Runtime::Runtime() : Runtime(std::filesystem::temp_directory_path() / "bayesian_group_testing_jit")
{
}

Runtime::Runtime(std::filesystem::path cache_directory) : jit_(std::move(cache_directory))
{
}

std::vector<Provider> Runtime::available_providers() const
{
	std::vector<Provider> providers{Provider::cpu};
	if (cuda_.available())
		providers.push_back(Provider::cuda);
	return providers;
}

JitCacheInfo Runtime::jit_cache_info() const
{
	return JitCacheInfo{registry_.size(), jit_.cache_entry_count(), jit_.cache_directory().string(), jit_.enabled()};
}

void Runtime::clear_jit_cache()
{
	registry_.clear();
	jit_.clear_cache();
}

KernelSpec Runtime::make_kernel_spec(const SimulationConfig &config) const
{
	KernelSpec spec;
	spec.provider = config.options.provider;
	spec.subjects = config.subjects;
	spec.variants = config.variants;
	spec.state_bits = BGT_STATE_BITS;
	spec.dilution = lattice_type_uses_dilution(config.lattice_type) ? DilutionMode::dilution : DilutionMode::non_dilution;
	spec.selector = resolve_selector(config.options.selector, config.variants);
	spec.compile_options = config.options.compile_options;
	spec.precision = config.options.compile_options.posterior_precision;
	spec.accumulator_precision = config.options.compile_options.accumulator_precision;
	return spec;
}

SimulationResult Runtime::run_simulation(const SimulationConfig &config)
{
	BGT_PERF_SCOPE_FOR(LogSubsystem::runtime, "runtime.run_simulation");
	const auto total_start = Clock::now();
	validate_simulation_config(config);
	std::unique_ptr<DilutionTable> owned_dilution = make_dilution_table(config);
	const DilutionTable *dilution = owned_dilution.get();
	KernelSpec spec = make_kernel_spec(config);
	if (!registry_.contains(spec) && jit_.compile(spec))
		registry_.insert(spec);

	if (is_parallel_mode(config.mode))
		return run_parallel_simulation(config, spec, dilution, total_start);

	SimulationResult result;
	result.kernel = spec;
	result.mode = config.mode;
	result.rank = 0;
	result.world_size = 1;
	result.total_states = state_count(config.subjects * config.variants);
	result.evaluated_states = 0;
	runtime_detail::ProgressTracker progress(config, result.rank, result.world_size, result.total_states, total_start);

	const auto eval_start = Clock::now();
	if (config.options.provider == Provider::cuda && !spec.compile_options.enable_cuda)
		throw std::runtime_error("CUDA provider requested, but CUDA is disabled in compile options.");
	if (progress.should_stop())
	{
		result.status = SimulationRunStatus::interrupted;
		result.termination_reason = simulation_cancellation_reason();
		result.progress_path = progress.progress_path();
		result.timings.total_seconds = elapsed_seconds(total_start, Clock::now());
		progress.finish(result.status, result.termination_reason, "cancelled before evaluation started");
		result.report_path = write_report_if_requested(config, result);
		return result;
	}

	if ((config.options.provider == Provider::auto_select || config.options.provider == Provider::cuda) &&
		spec.compile_options.enable_cuda && cuda_.available())
	{
		try
		{
			progress.maybe_emit("starting CUDA evaluation", true);
			result.stats = cuda_.run(config.lattice_type, config.subjects, config.variants, config.prior, config.options, dilution);
			result.provider = Provider::cuda;
			result.kernel.provider = Provider::cuda;
			result.evaluated_states = result.total_states;
			result.evaluated_prior_mass = 1.0;
			progress.replace_state(result.evaluated_states, result.evaluated_prior_mass, result.stats);
			result.timings.evaluation_seconds = elapsed_seconds(eval_start, Clock::now());
			result.timings.total_seconds = elapsed_seconds(total_start, Clock::now());
			result.progress_path = progress.progress_path();
			progress.finish(result.status, result.termination_reason, "completed");
			result.report_path = write_report_if_requested(config, result);
			return result;
		}
		catch (const std::exception &)
		{
			if (config.options.provider == Provider::cuda)
				throw;
		}
	}
	result.stats = runtime_detail::run_local_cpu_tree(config, dilution, progress);
	result.provider = Provider::cpu;
	result.kernel.provider = Provider::cpu;
	result.evaluated_states = progress.evaluated_states();
	result.evaluated_prior_mass = progress.evaluated_prior_mass();
	if (result.evaluated_states < result.total_states)
	{
		result.status = SimulationRunStatus::interrupted;
		result.termination_reason = simulation_cancellation_reason();
		if (result.termination_reason == TerminationReason::none)
			result.termination_reason = TerminationReason::cancellation_requested;
	}
	result.timings.evaluation_seconds = elapsed_seconds(eval_start, Clock::now());
	result.timings.total_seconds = elapsed_seconds(total_start, Clock::now());
	result.progress_path = progress.progress_path();
	progress.finish(result.status, result.termination_reason,
					result.status == SimulationRunStatus::completed ? "completed" : "interrupted");
	result.report_path = write_report_if_requested(config, result);
	return result;
}


SimulationResult run_simulation(const SimulationConfig &config)
{
	return Runtime{}.run_simulation(config);
}

} // namespace bgt
