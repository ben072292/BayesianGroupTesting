#pragma once

#include "bgt/runtime.hpp"
#include "bgt/detail/tree/tree.hpp"

#include <chrono>
#include <memory>
#include <string>

namespace bgt::runtime_detail
{

using Clock = std::chrono::steady_clock;

bool selector_is_single_test(SelectorType selector);
SelectorType resolve_selector(SelectorType selector, int variants);
bool lattice_type_is_distributed(LatticeType type);
bool lattice_type_uses_dilution(LatticeType type);
bool is_parallel_mode(SimulationMode mode);
std::string simulation_mode_name(SimulationMode mode);
seconds_t elapsed_seconds(Clock::time_point start, Clock::time_point stop);

void clear_tree_stats(TreeStats &stats);
void accumulate_tree_stats(tree::TreeStatsAccumulator &source, TreeStats &target);
TreeStats stats_from_tree_stat(tree::TreeStatsAccumulator &summary, int total_leaves);
void configure_tree_globals(const SimulationConfig &config, const DilutionTable *dilution);
std::string write_report_if_requested(const SimulationConfig &config, const SimulationResult &result);
SimulationResult run_parallel_simulation(const SimulationConfig &config, const KernelSpec &kernel,
										 const DilutionTable *dilution, Clock::time_point total_start);
std::unique_ptr<DilutionTable> make_dilution_table(const SimulationConfig &config);
void validate_simulation_config(const SimulationConfig &config);

} // namespace bgt::runtime_detail
