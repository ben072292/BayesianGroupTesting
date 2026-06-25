#include "bgt/detail/runtime/progress.hpp"

#include <atomic>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace
{

std::atomic<int> g_cancellation_reason{static_cast<int>(bgt::TerminationReason::none)};

std::string json_escape(const std::string &value)
{
	std::string escaped;
	escaped.reserve(value.size());
	for (char ch : value)
	{
		switch (ch)
		{
		case '\\':
			escaped += "\\\\";
			break;
		case '"':
			escaped += "\\\"";
			break;
		case '\n':
			escaped += "\\n";
			break;
		case '\r':
			escaped += "\\r";
			break;
		case '\t':
			escaped += "\\t";
			break;
		default:
			escaped += ch;
			break;
		}
	}
	return escaped;
}

std::string snapshot_json(const bgt::ProgressSnapshot &snapshot)
{
	std::ostringstream out;
	out << std::setprecision(17)
		<< "{"
		<< "\"status\":\"" << bgt::runtime_detail::simulation_run_status_name(snapshot.status) << "\","
		<< "\"termination_reason\":\"" << bgt::runtime_detail::termination_reason_name(snapshot.reason) << "\","
		<< "\"total_states\":" << snapshot.total_states << ','
		<< "\"evaluated_states\":" << snapshot.evaluated_states << ','
		<< "\"state_coverage_fraction\":" << snapshot.state_coverage_fraction << ','
		<< "\"evaluated_prior_mass\":" << snapshot.evaluated_prior_mass << ','
		<< "\"stats\":{"
		<< "\"total_leaves\":" << snapshot.stats.total_leaves << ','
		<< "\"correct_probability\":" << snapshot.stats.correct_probability << ','
		<< "\"incorrect_probability\":" << snapshot.stats.incorrect_probability << ','
		<< "\"false_positive_probability\":" << snapshot.stats.false_positive_probability << ','
		<< "\"false_negative_probability\":" << snapshot.stats.false_negative_probability << ','
		<< "\"unclassified_probability\":" << snapshot.stats.unclassified_probability << ','
		<< "\"expected_stages\":" << snapshot.stats.expected_stages << ','
		<< "\"expected_tests\":" << snapshot.stats.expected_tests
		<< "},"
		<< "\"timings\":{"
		<< "\"setup_seconds\":" << snapshot.timings.setup_seconds << ','
		<< "\"tree_build_seconds\":" << snapshot.timings.tree_build_seconds << ','
		<< "\"evaluation_seconds\":" << snapshot.timings.evaluation_seconds << ','
		<< "\"total_seconds\":" << snapshot.timings.total_seconds
		<< "},"
		<< "\"message\":\"" << json_escape(snapshot.message) << "\""
		<< "}";
	return out.str();
}

} // namespace

namespace bgt
{

void clear_simulation_cancellation()
{
	g_cancellation_reason.store(static_cast<int>(TerminationReason::none), std::memory_order_release);
}

void request_simulation_cancellation(TerminationReason reason)
{
	if (reason == TerminationReason::none)
		reason = TerminationReason::cancellation_requested;
	int expected = static_cast<int>(TerminationReason::none);
	g_cancellation_reason.compare_exchange_strong(expected, static_cast<int>(reason), std::memory_order_acq_rel);
}

bool simulation_cancellation_requested()
{
	return g_cancellation_reason.load(std::memory_order_acquire) != static_cast<int>(TerminationReason::none);
}

TerminationReason simulation_cancellation_reason()
{
	return static_cast<TerminationReason>(g_cancellation_reason.load(std::memory_order_acquire));
}

} // namespace bgt

namespace bgt::runtime_detail
{

std::string simulation_run_status_name(SimulationRunStatus status)
{
	switch (status)
	{
	case SimulationRunStatus::completed:
		return "completed";
	case SimulationRunStatus::interrupted:
		return "interrupted";
	case SimulationRunStatus::running:
		return "running";
	}
	return "unknown";
}

std::string termination_reason_name(TerminationReason reason)
{
	switch (reason)
	{
	case TerminationReason::none:
		return "none";
	case TerminationReason::user_signal:
		return "user_signal";
	case TerminationReason::cancellation_requested:
		return "cancellation_requested";
	}
	return "unknown";
}

ProgressTracker::ProgressTracker(const SimulationConfig &config, int rank, int world_size, int total_states,
								 Clock::time_point total_start)
	: config_(config),
	  rank_(rank),
	  total_states_(total_states),
	  total_start_(total_start),
	  last_emit_(total_start)
{
	(void)world_size;
	clear_tree_stats(stats_);
	if (enabled_for_rank())
		progress_path_ = latest_path().string();
}

void ProgressTracker::set_phase_timings(seconds_t setup_seconds, seconds_t tree_build_seconds)
{
	timings_.setup_seconds = setup_seconds;
	timings_.tree_build_seconds = tree_build_seconds;
}

void ProgressTracker::set_total_leaves(int total_leaves)
{
	stats_.total_leaves = total_leaves;
}

void ProgressTracker::record_state(tree::TreeStatsAccumulator &state_stat, host_probability_t prior_mass)
{
	accumulate_tree_stats(state_stat, stats_);
	evaluated_states_++;
	evaluated_prior_mass_ += prior_mass;
}

void ProgressTracker::record_states(int evaluated_states, host_probability_t prior_mass, const TreeStats &stats)
{
	evaluated_states_ += evaluated_states;
	evaluated_prior_mass_ += prior_mass;
	stats_.total_leaves = stats.total_leaves;
	stats_.correct_probability += stats.correct_probability;
	stats_.incorrect_probability += stats.incorrect_probability;
	stats_.false_positive_probability += stats.false_positive_probability;
	stats_.false_negative_probability += stats.false_negative_probability;
	stats_.unclassified_probability += stats.unclassified_probability;
	stats_.expected_stages += stats.expected_stages;
	stats_.expected_tests += stats.expected_tests;
}

void ProgressTracker::replace_state(int evaluated_states, host_probability_t prior_mass, const TreeStats &stats)
{
	const int known_leaves = stats_.total_leaves;
	evaluated_states_ = evaluated_states;
	evaluated_prior_mass_ = prior_mass;
	stats_ = stats;
	if (stats_.total_leaves == 0)
		stats_.total_leaves = known_leaves;
}

bool ProgressTracker::should_stop() const
{
	return bgt::simulation_cancellation_requested();
}

bool ProgressTracker::maybe_emit(const std::string &message, bool force)
{
	if (!enabled_for_rank())
		return false;
	const auto now = Clock::now();
	const seconds_t elapsed = elapsed_seconds(last_emit_, now);
	if (!force && elapsed < config_.progress.interval_seconds)
		return false;
	const TerminationReason reason = bgt::simulation_cancellation_reason();
	const SimulationRunStatus status = reason == TerminationReason::none
										   ? SimulationRunStatus::running
										   : SimulationRunStatus::interrupted;
	write_snapshot(snapshot(status, reason, message));
	last_emit_ = now;
	return true;
}

ProgressSnapshot ProgressTracker::snapshot(SimulationRunStatus status, TerminationReason reason,
										   const std::string &message) const
{
	ProgressSnapshot snap;
	snap.status = status;
	snap.reason = reason;
	snap.total_states = total_states_;
	snap.evaluated_states = evaluated_states_;
	snap.state_coverage_fraction = total_states_ == 0
										? 0.0
										: static_cast<seconds_t>(evaluated_states_) / static_cast<seconds_t>(total_states_);
	snap.evaluated_prior_mass = evaluated_prior_mass_;
	snap.stats = stats_;
	snap.timings = timings_;
	snap.timings.total_seconds = elapsed_seconds(total_start_, Clock::now());
	snap.message = message;
	return snap;
}

void ProgressTracker::finish(SimulationRunStatus status, TerminationReason reason, const std::string &message)
{
	if (!enabled_for_rank())
		return;
	write_snapshot(snapshot(status, reason, message));
}

bool ProgressTracker::enabled_for_rank() const
{
	return config_.progress.enabled && rank_ == 0;
}

std::filesystem::path ProgressTracker::latest_path() const
{
	if (!config_.progress.output_path.empty())
		return config_.progress.output_path;
	const std::string run_name = config_.report.run_name.empty() ? simulation_mode_name(config_.mode) : config_.report.run_name;
	return config_.report.output_directory / (run_name + ".progress.json");
}

std::filesystem::path ProgressTracker::history_path() const
{
	std::filesystem::path path = latest_path();
	path.replace_extension(".jsonl");
	return path;
}

void ProgressTracker::write_snapshot(const ProgressSnapshot &snapshot) const
{
	const std::filesystem::path latest = latest_path();
	if (!latest.parent_path().empty())
		std::filesystem::create_directories(latest.parent_path());
	const std::string body = snapshot_json(snapshot);
	const std::filesystem::path temp = latest.string() + ".tmp";
	{
		std::ofstream out(temp, std::ios::trunc);
		out << body << '\n';
	}
	std::filesystem::rename(temp, latest);
	if (config_.progress.write_jsonl)
	{
		const std::filesystem::path history = history_path();
		if (!history.parent_path().empty())
			std::filesystem::create_directories(history.parent_path());
		std::ofstream out(history, std::ios::app);
		out << body << '\n';
	}
	if (config_.progress.print_stderr)
		print_snapshot(snapshot);
}

void ProgressTracker::print_snapshot(const ProgressSnapshot &snapshot) const
{
	std::cerr << "bgt progress: status=" << simulation_run_status_name(snapshot.status)
			  << " evaluated=" << snapshot.evaluated_states << '/' << snapshot.total_states
			  << " coverage=" << snapshot.state_coverage_fraction
			  << " prior_mass=" << snapshot.evaluated_prior_mass
			  << " elapsed=" << snapshot.timings.total_seconds
			  << "s";
	if (!snapshot.message.empty())
		std::cerr << " " << snapshot.message;
	std::cerr << '\n';
}

} // namespace bgt::runtime_detail
