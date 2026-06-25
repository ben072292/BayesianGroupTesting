#pragma once

#include "bgt/types.hpp"

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace bgt
{

enum class SimulationMode
{
	local_tree,
	parallel_global_tree,
	parallel_dynamic_tree,
	parallel_hybrid_tree,
	parallel_partial_tree,
	parallel_fusion_tree
};

enum class TrueStatePolicy
{
	all,
	trimmed,
	symmetric
};

enum class SimulationRunStatus
{
	completed,
	interrupted,
	running
};

enum class TerminationReason
{
	none,
	user_signal,
	cancellation_requested
};

struct DilutionParameters
{
	host_probability_t alpha = 0.99;
	host_probability_t h = 0.005;
};

struct ReportConfig
{
	bool write_csv = false;
	std::filesystem::path output_directory = ".";
	std::string run_name;
};

struct SimulationTimings
{
	seconds_t setup_seconds = 0.0;
	seconds_t tree_build_seconds = 0.0;
	seconds_t evaluation_seconds = 0.0;
	seconds_t total_seconds = 0.0;
};

struct ProgressConfig
{
	bool enabled = false;
	seconds_t interval_seconds = 5.0;
	std::filesystem::path output_path;
	bool write_jsonl = true;
	bool print_stderr = false;
};

struct ProgressSnapshot
{
	SimulationRunStatus status = SimulationRunStatus::completed;
	TerminationReason reason = TerminationReason::none;
	int total_states = 0;
	int evaluated_states = 0;
	seconds_t state_coverage_fraction = 0.0;
	host_probability_t evaluated_prior_mass = 0.0;
	TreeStats stats;
	SimulationTimings timings;
	std::string message;
};

struct SimulationConfig
{
	LatticeType lattice_type = LatticeType::replicated_non_dilution;
	int subjects = 0;
	int variants = 1;
	std::vector<host_probability_t> prior;
	std::optional<std::vector<host_probability_t>> evaluation_prior;
	SimulationOptions options;
	SimulationMode mode = SimulationMode::local_tree;
	TrueStatePolicy true_state_policy = TrueStatePolicy::all;
	int global_tree_depth = 0;
	int workload_granularity = 1;
	host_probability_t trim_percent = 1.0;
	DilutionParameters dilution;
	ReportConfig report;
	ProgressConfig progress;
};

struct SimulationResult
{
	TreeStats stats;
	KernelSpec kernel;
	SimulationMode mode = SimulationMode::local_tree;
	Provider provider = Provider::auto_select;
	int rank = 0;
	int world_size = 1;
	int total_states = 0;
	int evaluated_states = 0;
	std::string report_path;
	SimulationTimings timings;
	SimulationRunStatus status = SimulationRunStatus::completed;
	TerminationReason termination_reason = TerminationReason::none;
	host_probability_t evaluated_prior_mass = 0.0;
	std::string progress_path;
};

void clear_simulation_cancellation();
void request_simulation_cancellation(TerminationReason reason = TerminationReason::cancellation_requested);
bool simulation_cancellation_requested();
TerminationReason simulation_cancellation_reason();
SimulationResult run_simulation(const SimulationConfig &config);

} // namespace bgt
