#include "bgt/detail/model/lattice.hpp"
#include "bgt/detail/performance.hpp"
#include "bgt/detail/runtime/progress.hpp"
#include "bgt/detail/runtime/simulation_helpers.hpp"

#include <algorithm>
#include <array>
#include <memory>
#include <stdexcept>
#include <vector>

using bgt::model::create_lattice;
using bgt::model::lattice_mpi_finalize;
using bgt::model::lattice_mpi_initialize;
using bgt::tree::DistributedTree;
using bgt::tree::FusionTree;
using bgt::tree::GlobalPartialTree;
using bgt::tree::GlobalTree;
using bgt::tree::Tree;
using bgt::tree::TreeStatsAccumulator;
using bgt::tree::generate_symmetric_true_states;
using bgt::tree::trim_true_states;

namespace bgt::runtime_detail
{

namespace
{
const int BGT_MPI_READY_TAG = 9101;
const int BGT_MPI_TASK_TAG = 9102;
const int BGT_MPI_STOP_TAG = 9103;
const int BGT_MPI_CANCEL_TAG = 9104;
const int BGT_MPI_PROGRESS_TAG = 9105;
const int BGT_MPI_PROGRESS_FIELDS = 9;

enum ProgressField
{
	progress_evaluated_states = 0,
	progress_prior_mass = 1,
	progress_correct_probability = 2,
	progress_incorrect_probability = 3,
	progress_false_positive_probability = 4,
	progress_false_negative_probability = 5,
	progress_unclassified_probability = 6,
	progress_expected_stages = 7,
	progress_expected_tests = 8
};


std::vector<bgt::host_probability_t> evaluation_prior(const SimulationConfig &config)
{
	return config.evaluation_prior.value_or(config.prior);
}

std::array<double, BGT_MPI_PROGRESS_FIELDS> make_progress_payload(
	int evaluated_states, host_probability_t prior_mass, TreeStatsAccumulator &partial)
{
	const TreeStats stats = stats_from_tree_stat(partial, 0);
	return {
		static_cast<double>(evaluated_states),
		prior_mass,
		stats.correct_probability,
		stats.incorrect_probability,
		stats.false_positive_probability,
		stats.false_negative_probability,
		stats.unclassified_probability,
		stats.expected_stages,
		stats.expected_tests};
}

void send_progress_payload(int evaluated_states, host_probability_t prior_mass, TreeStatsAccumulator &partial)
{
	const auto payload = make_progress_payload(evaluated_states, prior_mass, partial);
	BGT_MPI_CHECK(MPI_Send(payload.data(), BGT_MPI_PROGRESS_FIELDS, MPI_DOUBLE, 0, BGT_MPI_PROGRESS_TAG, MPI_COMM_WORLD));
}

TreeStats stats_from_progress_payloads(const std::vector<std::array<double, BGT_MPI_PROGRESS_FIELDS>> &payloads)
{
	TreeStats stats;
	clear_tree_stats(stats);
	for (const auto &payload : payloads)
	{
		stats.correct_probability += payload[progress_correct_probability];
		stats.incorrect_probability += payload[progress_incorrect_probability];
		stats.false_positive_probability += payload[progress_false_positive_probability];
		stats.false_negative_probability += payload[progress_false_negative_probability];
		stats.unclassified_probability += payload[progress_unclassified_probability];
		stats.expected_stages += payload[progress_expected_stages];
		stats.expected_tests += payload[progress_expected_tests];
	}
	return stats;
}

void update_master_progress(ProgressTracker *progress,
							const std::vector<std::array<double, BGT_MPI_PROGRESS_FIELDS>> &payloads)
{
	if (progress == nullptr)
		return;
	int evaluated_states = 0;
	host_probability_t prior_mass = 0.0;
	for (const auto &payload : payloads)
	{
		evaluated_states += static_cast<int>(payload[progress_evaluated_states]);
		prior_mass += payload[progress_prior_mass];
	}
	progress->replace_state(evaluated_states, prior_mass, stats_from_progress_payloads(payloads));
	progress->maybe_emit("evaluating true states");
}

void master_dispatch_contiguous_work(int total_items, int granularity, int world_size, ProgressTracker *progress)
{
	int next_item = 0;
	int stopped_workers = 0;
	std::vector<std::array<double, BGT_MPI_PROGRESS_FIELDS>> latest_progress(static_cast<std::size_t>(world_size));
	while (stopped_workers < world_size - 1)
	{
		// Dynamic tree simulation is scheduled by worker readiness, not by a
		// collective selector call. Any worker can report READY first; rank 0
		// replies with the next contiguous true-state chunk. The worker then runs
		// its local tree evaluation and local Op-BHA/BBPA selector path.
		MPI_Status status;
		BGT_MPI_CHECK(MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status));
		if (status.MPI_TAG == BGT_MPI_PROGRESS_TAG)
		{
			std::array<double, BGT_MPI_PROGRESS_FIELDS> payload{};
			BGT_MPI_CHECK(MPI_Recv(payload.data(), BGT_MPI_PROGRESS_FIELDS, MPI_DOUBLE, status.MPI_SOURCE,
								   BGT_MPI_PROGRESS_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
			latest_progress[static_cast<std::size_t>(status.MPI_SOURCE)] = payload;
			update_master_progress(progress, latest_progress);
			continue;
		}
		int ready = 0;
		BGT_MPI_CHECK(MPI_Recv(&ready, 1, MPI_INT, status.MPI_SOURCE, status.MPI_TAG, MPI_COMM_WORLD, &status));
		if (status.MPI_TAG == BGT_MPI_CANCEL_TAG)
			bgt::request_simulation_cancellation(TerminationReason::cancellation_requested);
		int task = -1;
		if (!bgt::simulation_cancellation_requested() && next_item < total_items)
		{
			task = next_item;
			next_item += granularity;
			BGT_MPI_CHECK(MPI_Send(&task, 1, MPI_INT, status.MPI_SOURCE, BGT_MPI_TASK_TAG, MPI_COMM_WORLD));
		}
		else
		{
			BGT_MPI_CHECK(MPI_Send(&task, 1, MPI_INT, status.MPI_SOURCE, BGT_MPI_STOP_TAG, MPI_COMM_WORLD));
			stopped_workers++;
		}
	}
}

void notify_master_cancellation()
{
	int cancelled = 1;
	BGT_MPI_CHECK(MPI_Send(&cancelled, 1, MPI_INT, 0, BGT_MPI_CANCEL_TAG, MPI_COMM_WORLD));
	int stop = -1;
	MPI_Status status;
	BGT_MPI_CHECK(MPI_Recv(&stop, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status));
}

int strided_evaluation_count(int total_items, int rank, int world_size)
{
	if (rank >= total_items)
		return 0;
	return 1 + (total_items - 1 - rank) / world_size;
}

bool should_use_cuda_provider(const SimulationConfig &config, int world_size)
{
	if (config.options.provider == Provider::cuda)
		return true;
	return config.options.provider == Provider::auto_select &&
		   config.options.compile_options.enable_cuda &&
		   (world_size == 1 || config.options.compile_options.enable_nccl) &&
		   bgt::CudaProvider{}.available();
}

LatticeType cuda_parallel_lattice_type(LatticeType type, int world_size)
{
	if (world_size == 1 || lattice_type_is_distributed(type))
		return type;
	return lattice_type_uses_dilution(type)
			   ? LatticeType::distributed_dilution
			   : LatticeType::distributed_non_dilution;
}

SimulationResult run_cuda_parallel_global_tree(const SimulationConfig &config, const KernelSpec &kernel, const DilutionTable *dilution,
											   Clock::time_point total_start, int rank, int world_size)
{
	if (!config.options.compile_options.enable_cuda)
		throw std::runtime_error("CUDA provider requested, but CUDA is disabled in compile options.");
	if (world_size > 1 && !config.options.compile_options.enable_nccl)
		throw std::runtime_error("distributed CUDA tree simulation requires NCCL collectives.");
	if (config.evaluation_prior)
		throw std::invalid_argument("CUDA parallel simulation does not yet support evaluation_prior misspecification runs.");

	const int total_states = state_count(config.subjects * config.variants);
	const auto eval_start = Clock::now();
	SimulationResult result;
	ProgressTracker progress(config, rank, world_size, total_states, total_start);
	progress.maybe_emit("starting CUDA parallel evaluation", true);
	result.stats = bgt::CudaProvider{}.run(
		cuda_parallel_lattice_type(config.lattice_type, world_size),
		config.subjects, config.variants, config.prior, config.options, dilution);
	result.kernel = kernel;
	result.kernel.provider = Provider::cuda;
	result.mode = config.mode;
	result.provider = Provider::cuda;
	result.rank = rank;
	result.world_size = world_size;
	result.total_states = total_states;
	result.evaluated_states = rank == 0 ? total_states : strided_evaluation_count(total_states, rank, world_size);
	result.evaluated_prior_mass = rank == 0 ? 1.0 : 0.0;
	result.timings.evaluation_seconds = elapsed_seconds(eval_start, Clock::now());
	result.timings.total_seconds = elapsed_seconds(total_start, Clock::now());
	result.progress_path = progress.progress_path();
	if (rank == 0)
	{
		progress.replace_state(result.evaluated_states, result.evaluated_prior_mass, result.stats);
		progress.finish(result.status, result.termination_reason, "completed");
	}
	result.report_path = write_report_if_requested(config, result);
	return result;
}

class MpiLatticeScope
{
public:
	MpiLatticeScope(lattice_type_t type, int subjects, int variants) : type_(type)
	{
		lattice_mpi_initialize(type_, subjects, variants);
	}

	~MpiLatticeScope()
	{
		lattice_mpi_finalize(type_);
	}

	MpiLatticeScope(const MpiLatticeScope &) = delete;
	MpiLatticeScope &operator=(const MpiLatticeScope &) = delete;

private:
	lattice_type_t type_;
};

class GlobalTreeMpiScope
{
public:
	GlobalTreeMpiScope(int subjects, int variants)
	{
		GlobalTree::initialize_mpi(subjects, variants);
	}
	~GlobalTreeMpiScope()
	{
		GlobalTree::finalize_mpi();
	}
	GlobalTreeMpiScope(const GlobalTreeMpiScope &) = delete;
	GlobalTreeMpiScope &operator=(const GlobalTreeMpiScope &) = delete;
};

class DistributedTreeMpiScope
{
public:
	DistributedTreeMpiScope(int subjects, int variants, int search_depth)
	{
		DistributedTree::initialize_mpi(subjects, variants, search_depth);
	}
	~DistributedTreeMpiScope()
	{
		DistributedTree::finalize_mpi();
	}
	DistributedTreeMpiScope(const DistributedTreeMpiScope &) = delete;
	DistributedTreeMpiScope &operator=(const DistributedTreeMpiScope &) = delete;
};

class FusionTreeMpiScope
{
public:
	FusionTreeMpiScope(int subjects, int variants)
	{
		FusionTree::initialize_mpi(subjects, variants);
	}
	~FusionTreeMpiScope()
	{
		FusionTree::finalize_mpi();
	}
	FusionTreeMpiScope(const FusionTreeMpiScope &) = delete;
	FusionTreeMpiScope &operator=(const FusionTreeMpiScope &) = delete;
};

SimulationResult run_parallel_global_tree(const SimulationConfig &config, const KernelSpec &kernel,
									 const DilutionTable *dilution, Clock::time_point total_start)
{
	BGT_PERF_SCOPE_FOR(LogSubsystem::parallel, "parallel.global_tree");
	int rank = 0;
	int world_size = 1;
	BGT_MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
	BGT_MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &world_size));
	if (should_use_cuda_provider(config, world_size))
		return run_cuda_parallel_global_tree(config, kernel, dilution, total_start, rank, world_size);

	const auto setup_start = Clock::now();
	const lattice_type_t internal_type = to_internal_lattice_type(config.lattice_type);
	std::vector<host_probability_t> prior = config.prior;
	std::vector<host_probability_t> eval_prior = evaluation_prior(config);
	GlobalTreeMpiScope tree_scope(config.subjects, config.variants);
	MpiLatticeScope lattice_scope(internal_type, config.subjects, config.variants);
	configure_tree_globals(config, dilution);

	auto reference_lattice = create_lattice(internal_type, config.subjects, config.variants, eval_prior.data());
	const auto setup_stop = Clock::now();

	const auto tree_start = Clock::now();
	auto tree_lattice = create_lattice(internal_type, config.subjects, config.variants, prior.data());
	auto tree = std::make_unique<GlobalTree>(std::move(tree_lattice), invalid_state(), invalid_state(), 1, 0);
	std::vector<const Tree *> leaves;
	Tree::find_all_leaves(tree.get(), &leaves);
	const auto tree_stop = Clock::now();

	const int total_states = state_count(config.subjects * config.variants);
	TreeStatsAccumulator partial(config.options.search_depth, 1);
	TreeStatsAccumulator state_stat(config.options.search_depth, 1);
	TreeStatsAccumulator summary(config.options.search_depth, 1);
	int local_evaluated = 0;
	host_probability_t local_prior_mass = 0.0;
	ProgressTracker progress(config, rank, world_size, total_states, total_start);
	progress.set_phase_timings(elapsed_seconds(setup_start, setup_stop), elapsed_seconds(tree_start, tree_stop));
	progress.set_total_leaves(static_cast<int>(leaves.size()));
	progress.maybe_emit("starting parallel evaluation", true);

	const auto eval_start = Clock::now();
	if (world_size == 1)
	{
		for (int true_state = 0; true_state < total_states; true_state++)
		{
			if (progress.should_stop())
				break;
			const auto encoded_state = static_cast<state_t>(true_state);
			tree->apply_true_state(reference_lattice.get(), encoded_state);
			tree->parse(encoded_state, reference_lattice.get(), 1.0, &state_stat);
			partial.merge(&state_stat);
			const host_probability_t prior_mass = reference_lattice->prior_prob(encoded_state, model::Lattice::pi0());
			local_prior_mass += prior_mass;
			local_evaluated++;
			progress.record_state(state_stat, prior_mass);
			progress.maybe_emit("evaluating true states");
		}
	}
	else if (rank == 0)
	{
		master_dispatch_contiguous_work(total_states, config.workload_granularity, world_size, &progress);
	}
	else
	{
		auto last_progress_send = Clock::now();
		while (true)
		{
			if (bgt::simulation_cancellation_requested())
			{
				if (config.progress.enabled)
					send_progress_payload(local_evaluated, local_prior_mass, partial);
				notify_master_cancellation();
				break;
			}
			int ready = 1;
			BGT_MPI_CHECK(MPI_Send(&ready, 1, MPI_INT, 0, BGT_MPI_READY_TAG, MPI_COMM_WORLD));
			int start_state = -1;
			MPI_Status status;
			BGT_MPI_CHECK(MPI_Recv(&start_state, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status));
			if (status.MPI_TAG == BGT_MPI_STOP_TAG)
				break;
			const int stop_state = std::min(start_state + config.workload_granularity, total_states);
			for (int true_state = start_state; true_state < stop_state; true_state++)
			{
				const auto encoded_state = static_cast<state_t>(true_state);
				tree->apply_true_state(reference_lattice.get(), encoded_state);
				tree->parse(encoded_state, reference_lattice.get(), 1.0, &state_stat);
				partial.merge(&state_stat);
				local_prior_mass += reference_lattice->prior_prob(encoded_state, model::Lattice::pi0());
				local_evaluated++;
				const auto now = Clock::now();
				if (config.progress.enabled &&
					elapsed_seconds(last_progress_send, now) >= config.progress.interval_seconds)
				{
					send_progress_payload(local_evaluated, local_prior_mass, partial);
					last_progress_send = now;
				}
				if (bgt::simulation_cancellation_requested())
					break;
			}
			if (bgt::simulation_cancellation_requested())
			{
				if (config.progress.enabled)
					send_progress_payload(local_evaluated, local_prior_mass, partial);
				notify_master_cancellation();
				break;
			}
		}
	}
	const auto eval_stop = Clock::now();

	int evaluated_states = 0;
	host_probability_t evaluated_prior_mass = 0.0;
	BGT_MPI_CHECK(MPI_Reduce(&local_evaluated, &evaluated_states, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD));
	BGT_MPI_CHECK(MPI_Reduce(&local_prior_mass, &evaluated_prior_mass, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD));
	BGT_MPI_CHECK(MPI_Reduce(&partial, &summary, 1, GlobalTree::stats_mpi_type, GlobalTree::stats_mpi_op, 0, MPI_COMM_WORLD));
	BGT_MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

	SimulationResult result;
	result.kernel = kernel;
	result.mode = config.mode;
	result.provider = Provider::cpu;
	result.rank = rank;
	result.world_size = world_size;
	result.total_states = total_states;
	result.evaluated_states = rank == 0 ? evaluated_states : local_evaluated;
	result.evaluated_prior_mass = rank == 0 ? evaluated_prior_mass : local_prior_mass;
	if (rank == 0)
		result.stats = stats_from_tree_stat(summary, static_cast<int>(leaves.size()));
	if ((rank == 0 && evaluated_states < total_states) || (rank != 0 && bgt::simulation_cancellation_requested()))
	{
		result.status = SimulationRunStatus::interrupted;
		result.termination_reason = bgt::simulation_cancellation_reason();
		if (result.termination_reason == TerminationReason::none)
			result.termination_reason = TerminationReason::cancellation_requested;
	}
	result.timings.setup_seconds = elapsed_seconds(setup_start, setup_stop);
	result.timings.tree_build_seconds = elapsed_seconds(tree_start, tree_stop);
	result.timings.evaluation_seconds = elapsed_seconds(eval_start, eval_stop);
	result.timings.total_seconds = elapsed_seconds(total_start, Clock::now());
	result.progress_path = progress.progress_path();
	if (rank == 0 && world_size > 1)
		progress.replace_state(result.evaluated_states, result.evaluated_prior_mass, result.stats);
	progress.finish(result.status, result.termination_reason,
					result.status == SimulationRunStatus::completed ? "completed" : "interrupted");
	result.report_path = write_report_if_requested(config, result);
	return result;
}

SimulationResult run_parallel_dynamic_tree(const SimulationConfig &config, const KernelSpec &kernel,
										  const DilutionTable *dilution, Clock::time_point total_start)
{
	BGT_PERF_SCOPE_FOR(LogSubsystem::parallel,
					   config.mode == SimulationMode::parallel_hybrid_tree ? "parallel.hybrid_tree" : "parallel.dynamic_tree");
	int rank = 0;
	int world_size = 1;
	BGT_MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
	BGT_MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &world_size));

	const auto setup_start = Clock::now();
	const lattice_type_t internal_type = to_internal_lattice_type(config.lattice_type);
	std::vector<host_probability_t> prior = config.prior;
	std::vector<host_probability_t> eval_prior = evaluation_prior(config);
	DistributedTreeMpiScope tree_scope(config.subjects, config.variants, config.options.search_depth);
	MpiLatticeScope lattice_scope(internal_type, config.subjects, config.variants);
	configure_tree_globals(config, dilution);
	const auto setup_stop = Clock::now();

	const auto tree_start = Clock::now();
	auto tree_lattice = create_lattice(internal_type, config.subjects, config.variants, prior.data());
	std::unique_ptr<DistributedTree> tree;
	if (config.mode == SimulationMode::parallel_hybrid_tree)
	{
		tree = std::make_unique<DistributedTree>(
			std::move(tree_lattice), invalid_state(), invalid_state(), invalid_state(), 1, 0, config.global_tree_depth);
	}
	else
	{
		tree = std::make_unique<DistributedTree>(
			std::move(tree_lattice), invalid_state(), invalid_state(), invalid_state(), 0, 1.0);
	}
	tree->branch_prob(1.0);
	const auto tree_stop = Clock::now();

	auto reference_lattice = create_lattice(internal_type, config.subjects, config.variants, eval_prior.data());
	const int total_states = reference_lattice->total_states();
	TreeStatsAccumulator partial(config.options.search_depth, 1);
	TreeStatsAccumulator state_stat(config.options.search_depth, 1);
	TreeStatsAccumulator summary(config.options.search_depth, 1);
	int local_evaluated = 0;
	host_probability_t local_prior_mass = 0.0;
	ProgressTracker progress(config, rank, world_size, total_states, total_start);
	progress.set_phase_timings(elapsed_seconds(setup_start, setup_stop), elapsed_seconds(tree_start, tree_stop));
	progress.maybe_emit("starting parallel evaluation", true);

	const auto eval_start = Clock::now();
	if (world_size == 1)
	{
		for (int true_state = 0; true_state < total_states; true_state++)
		{
			if (progress.should_stop())
				break;
			const auto encoded_state = static_cast<state_t>(true_state);
			tree->lazy_eval(tree.get(), reference_lattice.get(), encoded_state);
			tree->parse(encoded_state, reference_lattice.get(), 1.0, &state_stat);
			partial.merge(&state_stat);
			const host_probability_t prior_mass = reference_lattice->prior_prob(encoded_state, model::Lattice::pi0());
			local_prior_mass += prior_mass;
			local_evaluated++;
			progress.record_state(state_stat, prior_mass);
			progress.maybe_emit("evaluating true states");
		}
	}
	else if (rank == 0)
	{
		master_dispatch_contiguous_work(total_states, config.workload_granularity, world_size, &progress);
	}
	else
	{
		auto last_progress_send = Clock::now();
		while (true)
		{
			if (bgt::simulation_cancellation_requested())
			{
				if (config.progress.enabled)
					send_progress_payload(local_evaluated, local_prior_mass, partial);
				notify_master_cancellation();
				break;
			}
			int ready = 1;
			BGT_MPI_CHECK(MPI_Send(&ready, 1, MPI_INT, 0, BGT_MPI_READY_TAG, MPI_COMM_WORLD));
			int start_state = -1;
			MPI_Status status;
			BGT_MPI_CHECK(MPI_Recv(&start_state, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status));
			if (status.MPI_TAG == BGT_MPI_STOP_TAG)
				break;
			const int stop_state = std::min(start_state + config.workload_granularity, total_states);
			for (int true_state = start_state; true_state < stop_state; true_state++)
			{
				const auto encoded_state = static_cast<state_t>(true_state);
				tree->lazy_eval(tree.get(), reference_lattice.get(), encoded_state);
				tree->parse(encoded_state, reference_lattice.get(), 1.0, &state_stat);
				partial.merge(&state_stat);
				local_prior_mass += reference_lattice->prior_prob(encoded_state, model::Lattice::pi0());
				local_evaluated++;
				const auto now = Clock::now();
				if (config.progress.enabled &&
					elapsed_seconds(last_progress_send, now) >= config.progress.interval_seconds)
				{
					send_progress_payload(local_evaluated, local_prior_mass, partial);
					last_progress_send = now;
				}
				if (bgt::simulation_cancellation_requested())
					break;
			}
			if (bgt::simulation_cancellation_requested())
			{
				if (config.progress.enabled)
					send_progress_payload(local_evaluated, local_prior_mass, partial);
				notify_master_cancellation();
				break;
			}
		}
	}
	const auto eval_stop = Clock::now();

	int evaluated_states = 0;
	host_probability_t evaluated_prior_mass = 0.0;
	BGT_MPI_CHECK(MPI_Reduce(&local_evaluated, &evaluated_states, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD));
	BGT_MPI_CHECK(MPI_Reduce(&local_prior_mass, &evaluated_prior_mass, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD));
	BGT_MPI_CHECK(MPI_Reduce(&partial, &summary, 1, DistributedTree::stats_mpi_type, DistributedTree::stats_mpi_op, 0, MPI_COMM_WORLD));

	std::vector<const Tree *> leaves;
	Tree::find_all_leaves(tree.get(), &leaves);
	const int local_leaf_count = static_cast<int>(leaves.size());
	int leaf_count = 0;
	BGT_MPI_CHECK(MPI_Reduce(&local_leaf_count, &leaf_count, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD));
	BGT_MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
	progress.set_total_leaves(rank == 0 ? leaf_count : local_leaf_count);

	SimulationResult result;
	result.kernel = kernel;
	result.mode = config.mode;
	result.provider = Provider::cpu;
	result.rank = rank;
	result.world_size = world_size;
	result.total_states = total_states;
	result.evaluated_states = rank == 0 ? evaluated_states : local_evaluated;
	result.evaluated_prior_mass = rank == 0 ? evaluated_prior_mass : local_prior_mass;
	if (rank == 0)
		result.stats = stats_from_tree_stat(summary, leaf_count);
	if ((rank == 0 && evaluated_states < total_states) || (rank != 0 && bgt::simulation_cancellation_requested()))
	{
		result.status = SimulationRunStatus::interrupted;
		result.termination_reason = bgt::simulation_cancellation_reason();
		if (result.termination_reason == TerminationReason::none)
			result.termination_reason = TerminationReason::cancellation_requested;
	}
	result.timings.setup_seconds = elapsed_seconds(setup_start, setup_stop);
	result.timings.tree_build_seconds = elapsed_seconds(tree_start, tree_stop);
	result.timings.evaluation_seconds = elapsed_seconds(eval_start, eval_stop);
	result.timings.total_seconds = elapsed_seconds(total_start, Clock::now());
	result.progress_path = progress.progress_path();
	if (rank == 0 && world_size > 1)
		progress.replace_state(result.evaluated_states, result.evaluated_prior_mass, result.stats);
	progress.finish(result.status, result.termination_reason,
					result.status == SimulationRunStatus::completed ? "completed" : "interrupted");
	result.report_path = write_report_if_requested(config, result);
	return result;
}

SimulationResult run_parallel_partial_tree(const SimulationConfig &config, const KernelSpec &kernel,
									  const DilutionTable *dilution, Clock::time_point total_start)
{
	BGT_PERF_SCOPE_FOR(LogSubsystem::parallel, "parallel.partial_tree");
	int rank = 0;
	int world_size = 1;
	BGT_MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
	BGT_MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &world_size));

	const auto setup_start = Clock::now();
	const lattice_type_t internal_type = to_internal_lattice_type(config.lattice_type);
	std::vector<host_probability_t> prior = config.prior;
	std::vector<host_probability_t> eval_prior = evaluation_prior(config);
	DistributedTreeMpiScope tree_scope(config.subjects, config.variants, config.options.search_depth);
	MpiLatticeScope lattice_scope(internal_type, config.subjects, config.variants);
	configure_tree_globals(config, dilution);
	const auto setup_stop = Clock::now();

	const auto tree_start = Clock::now();
	auto tree_lattice = create_lattice(internal_type, config.subjects, config.variants, prior.data());
	auto tree = std::make_unique<GlobalPartialTree>(std::move(tree_lattice), invalid_state(), invalid_state(), invalid_state(), 0, 1.0);
	const auto tree_stop = Clock::now();

	auto reference_lattice = create_lattice(internal_type, config.subjects, config.variants, eval_prior.data());
	std::vector<state_t> true_states;
	std::vector<int> coefficients;
	if (config.true_state_policy == TrueStatePolicy::symmetric)
	{
		true_states = generate_symmetric_true_states(config.subjects, config.variants, coefficients);
	}
	else
	{
		auto trimming_lattice = create_lattice(REPL_NON_DILUTION, config.subjects, config.variants, eval_prior.data());
		std::vector<int> trimmed = trim_true_states(trimming_lattice->posterior_probs(), trimming_lattice->total_states(), config.trim_percent);
		true_states.reserve(trimmed.size());
		coefficients.assign(trimmed.size(), 1);
		for (int state : trimmed)
			true_states.push_back(static_cast<state_t>(state));
	}
	if (coefficients.empty())
		coefficients.assign(true_states.size(), 1);

	TreeStatsAccumulator partial(config.options.search_depth, 1);
	TreeStatsAccumulator state_stat(config.options.search_depth, 1);
	const int total_states = static_cast<int>(true_states.size());
	int local_evaluated = 0;
	host_probability_t local_prior_mass = 0.0;
	ProgressTracker progress(config, rank, world_size, total_states, total_start);
	progress.set_phase_timings(elapsed_seconds(setup_start, setup_stop), elapsed_seconds(tree_start, tree_stop));
	progress.maybe_emit("starting parallel evaluation", true);

	const auto eval_start = Clock::now();
	for (std::size_t index = 0; index < true_states.size(); index++)
	{
		if (progress.should_stop())
			break;
		tree->lazy_eval(tree.get(), reference_lattice.get(), true_states[index]);
		tree->parse(true_states[index], reference_lattice.get(), static_cast<host_probability_t>(coefficients[index]), &state_stat);
		partial.merge(&state_stat);
		const host_probability_t prior_mass =
			reference_lattice->prior_prob(true_states[index], model::Lattice::pi0()) *
			static_cast<host_probability_t>(coefficients[index]);
		local_prior_mass += prior_mass;
		local_evaluated++;
		progress.record_state(state_stat, prior_mass);
		progress.maybe_emit("evaluating true states");
	}
	const auto eval_stop = Clock::now();

	BGT_MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

	std::vector<const Tree *> leaves;
	Tree::find_all_leaves(tree.get(), &leaves);
	progress.set_total_leaves(static_cast<int>(leaves.size()));

	SimulationResult result;
	result.kernel = kernel;
	result.mode = config.mode;
	result.provider = Provider::cpu;
	result.rank = rank;
	result.world_size = world_size;
	result.total_states = total_states;
	result.evaluated_states = local_evaluated;
	result.evaluated_prior_mass = local_prior_mass;
	if (rank == 0)
		result.stats = stats_from_tree_stat(partial, static_cast<int>(leaves.size()));
	if (local_evaluated < total_states)
	{
		result.status = SimulationRunStatus::interrupted;
		result.termination_reason = bgt::simulation_cancellation_reason();
		if (result.termination_reason == TerminationReason::none)
			result.termination_reason = TerminationReason::cancellation_requested;
	}
	result.timings.setup_seconds = elapsed_seconds(setup_start, setup_stop);
	result.timings.tree_build_seconds = elapsed_seconds(tree_start, tree_stop);
	result.timings.evaluation_seconds = elapsed_seconds(eval_start, eval_stop);
	result.timings.total_seconds = elapsed_seconds(total_start, Clock::now());
	result.progress_path = progress.progress_path();
	progress.finish(result.status, result.termination_reason,
					result.status == SimulationRunStatus::completed ? "completed" : "interrupted");
	result.report_path = write_report_if_requested(config, result);
	return result;
}

SimulationResult run_parallel_fusion_tree(const SimulationConfig &config, const KernelSpec &kernel,
									 const DilutionTable *dilution, Clock::time_point total_start)
{
	BGT_PERF_SCOPE_FOR(LogSubsystem::parallel, "parallel.fusion_tree");
	int rank = 0;
	int world_size = 1;
	BGT_MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
	BGT_MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &world_size));

	const auto setup_start = Clock::now();
	const lattice_type_t internal_type = to_internal_lattice_type(config.lattice_type);
	std::vector<host_probability_t> prior = config.prior;
	std::vector<host_probability_t> eval_prior = evaluation_prior(config);
	FusionTreeMpiScope tree_scope(config.subjects, config.variants);
	MpiLatticeScope lattice_scope(internal_type, config.subjects, config.variants);
	configure_tree_globals(config, dilution);
	auto reference_lattice = create_lattice(internal_type, config.subjects, config.variants, eval_prior.data());
	const auto setup_stop = Clock::now();

	const auto tree_start = Clock::now();
	auto tree_lattice = create_lattice(internal_type, config.subjects, config.variants, prior.data());
	auto tree = std::make_unique<FusionTree>(std::move(tree_lattice), invalid_state(), invalid_state(), 1, 0, 0.01, 0.0, 1e-6);
	std::vector<const Tree *> leaves;
	Tree::find_all_leaves(tree.get(), &leaves);
	const auto tree_stop = Clock::now();

	const int total_states = reference_lattice->total_states();
	TreeStatsAccumulator partial(config.options.search_depth, 1);
	TreeStatsAccumulator state_stat(config.options.search_depth, 1);
	TreeStatsAccumulator summary(config.options.search_depth, 1);
	int local_evaluated = 0;
	host_probability_t local_prior_mass = 0.0;
	ProgressTracker progress(config, rank, world_size, total_states, total_start);
	progress.set_phase_timings(elapsed_seconds(setup_start, setup_stop), elapsed_seconds(tree_start, tree_stop));
	progress.set_total_leaves(static_cast<int>(leaves.size()));
	progress.maybe_emit("starting parallel evaluation", true);

	const auto eval_start = Clock::now();
	for (int true_state = rank; true_state < total_states; true_state += world_size)
	{
		if (progress.should_stop())
			break;
		const auto encoded_state = static_cast<state_t>(true_state);
		tree->apply_true_state(reference_lattice.get(), encoded_state);
		tree->parse(encoded_state, reference_lattice.get(), 1.0, &state_stat);
		partial.merge(&state_stat);
		local_prior_mass += reference_lattice->prior_prob(encoded_state, model::Lattice::pi0());
		local_evaluated++;
	}
	const auto eval_stop = Clock::now();

	int evaluated_states = 0;
	host_probability_t evaluated_prior_mass = 0.0;
	BGT_MPI_CHECK(MPI_Reduce(&local_evaluated, &evaluated_states, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD));
	BGT_MPI_CHECK(MPI_Reduce(&local_prior_mass, &evaluated_prior_mass, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD));
	BGT_MPI_CHECK(MPI_Reduce(&partial, &summary, 1, GlobalTree::stats_mpi_type, GlobalTree::stats_mpi_op, 0, MPI_COMM_WORLD));
	BGT_MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

	SimulationResult result;
	result.kernel = kernel;
	result.mode = config.mode;
	result.provider = Provider::cpu;
	result.rank = rank;
	result.world_size = world_size;
	result.total_states = total_states;
	result.evaluated_states = rank == 0 ? evaluated_states : local_evaluated;
	result.evaluated_prior_mass = rank == 0 ? evaluated_prior_mass : local_prior_mass;
	if (rank == 0)
		result.stats = stats_from_tree_stat(summary, static_cast<int>(leaves.size()));
	if ((rank == 0 && evaluated_states < total_states) || (rank != 0 && bgt::simulation_cancellation_requested()))
	{
		result.status = SimulationRunStatus::interrupted;
		result.termination_reason = bgt::simulation_cancellation_reason();
		if (result.termination_reason == TerminationReason::none)
			result.termination_reason = TerminationReason::cancellation_requested;
	}
	result.timings.setup_seconds = elapsed_seconds(setup_start, setup_stop);
	result.timings.tree_build_seconds = elapsed_seconds(tree_start, tree_stop);
	result.timings.evaluation_seconds = elapsed_seconds(eval_start, eval_stop);
	result.timings.total_seconds = elapsed_seconds(total_start, Clock::now());
	result.progress_path = progress.progress_path();
	if (rank == 0)
		progress.replace_state(result.evaluated_states, result.evaluated_prior_mass, result.stats);
	progress.finish(result.status, result.termination_reason,
					result.status == SimulationRunStatus::completed ? "completed" : "interrupted");
	result.report_path = write_report_if_requested(config, result);
	return result;
}

} // namespace

SimulationResult run_parallel_simulation(const SimulationConfig &config, const KernelSpec &kernel,
									const DilutionTable *dilution, Clock::time_point total_start)
{
	if (config.options.provider == Provider::cuda && config.mode != SimulationMode::parallel_global_tree)
		throw std::invalid_argument("CUDA parallel simulation currently supports parallel_global_tree only.");

	switch (config.mode)
	{
	case SimulationMode::parallel_global_tree:
		return run_parallel_global_tree(config, kernel, dilution, total_start);
	case SimulationMode::parallel_dynamic_tree:
	case SimulationMode::parallel_hybrid_tree:
		return run_parallel_dynamic_tree(config, kernel, dilution, total_start);
	case SimulationMode::parallel_partial_tree:
		return run_parallel_partial_tree(config, kernel, dilution, total_start);
	case SimulationMode::parallel_fusion_tree:
		return run_parallel_fusion_tree(config, kernel, dilution, total_start);
	case SimulationMode::local_tree:
		break;
	}
	throw std::invalid_argument("unsupported parallel simulation mode.");
}
} // namespace bgt::runtime_detail
