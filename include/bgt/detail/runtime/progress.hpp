#pragma once

#include "bgt/detail/runtime/simulation_helpers.hpp"

#include <filesystem>
#include <string>

namespace bgt::runtime_detail
{

std::string simulation_run_status_name(SimulationRunStatus status);
std::string termination_reason_name(TerminationReason reason);

class ProgressTracker
{
public:
	ProgressTracker(const SimulationConfig &config, int rank, int world_size, int total_states,
					Clock::time_point total_start);

	void set_phase_timings(seconds_t setup_seconds, seconds_t tree_build_seconds);
	void set_total_leaves(int total_leaves);
	void record_state(tree::TreeStatsAccumulator &state_stat, host_probability_t prior_mass);
	void record_states(int evaluated_states, host_probability_t prior_mass, const TreeStats &stats);
	void replace_state(int evaluated_states, host_probability_t prior_mass, const TreeStats &stats);
	bool should_stop() const;
	bool maybe_emit(const std::string &message = {}, bool force = false);
	ProgressSnapshot snapshot(SimulationRunStatus status, TerminationReason reason,
							  const std::string &message) const;
	void finish(SimulationRunStatus status, TerminationReason reason, const std::string &message);

	int evaluated_states() const { return evaluated_states_; }
	host_probability_t evaluated_prior_mass() const { return evaluated_prior_mass_; }
	const TreeStats &stats() const { return stats_; }
	const std::string &progress_path() const { return progress_path_; }

private:
	bool enabled_for_rank() const;
	std::filesystem::path latest_path() const;
	std::filesystem::path history_path() const;
	void write_snapshot(const ProgressSnapshot &snapshot) const;
	void print_snapshot(const ProgressSnapshot &snapshot) const;

	const SimulationConfig &config_;
	int rank_ = 0;
	int total_states_ = 0;
	Clock::time_point total_start_;
	Clock::time_point last_emit_;
	TreeStats stats_;
	SimulationTimings timings_;
	int evaluated_states_ = 0;
	host_probability_t evaluated_prior_mass_ = 0.0;
	std::string progress_path_;
};

} // namespace bgt::runtime_detail
