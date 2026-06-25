#include "bgt/detail/runtime/simulation_helpers.hpp"

namespace bgt::runtime_detail
{

void clear_tree_stats(bgt::TreeStats &stats)
{
	stats = bgt::TreeStats{};
}

void accumulate_tree_stats(tree::TreeStatsAccumulator &source, bgt::TreeStats &target)
{
	target.unclassified_probability += source.unclassified();
	target.expected_stages += source.expected_stage();
	target.expected_tests += source.expected_test();
	for (int i = 0; i <= source.depth() * source.k(); i++)
	{
		target.correct_probability += source.correct()[i];
		target.incorrect_probability += source.incorrect()[i];
		target.false_positive_probability += source.fp()[i];
		target.false_negative_probability += source.fn()[i];
	}
}
TreeStats stats_from_tree_stat(tree::TreeStatsAccumulator &summary, int total_leaves)
{
	TreeStats stats;
	clear_tree_stats(stats);
	stats.total_leaves = total_leaves;
	accumulate_tree_stats(summary, stats);
	return stats;
}

bool selector_is_single_test(bgt::SelectorType selector)
{
	return selector == bgt::SelectorType::auto_select ||
		   selector == bgt::SelectorType::op_bha ||
		   selector == bgt::SelectorType::op_bbpa;
}

SelectorType resolve_selector(SelectorType selector, int variants)
{
	if (selector == SelectorType::auto_select)
		return variants == 1 ? SelectorType::op_bha : SelectorType::op_bbpa;
	return selector;
}

bool lattice_type_is_distributed(bgt::LatticeType type)
{
	return type == bgt::LatticeType::distributed_non_dilution ||
		   type == bgt::LatticeType::distributed_dilution;
}

bool lattice_type_uses_dilution(bgt::LatticeType type)
{
	return type == bgt::LatticeType::replicated_dilution ||
		   type == bgt::LatticeType::distributed_dilution;
}

seconds_t elapsed_seconds(Clock::time_point start, Clock::time_point stop)
{
	return std::chrono::duration<seconds_t>(stop - start).count();
}

bool is_parallel_mode(SimulationMode mode)
{
	return mode != SimulationMode::local_tree;
}

std::string simulation_mode_name(SimulationMode mode)
{
	switch (mode)
	{
	case SimulationMode::local_tree:
		return "local_tree";
	case SimulationMode::parallel_global_tree:
		return "parallel_global_tree";
	case SimulationMode::parallel_dynamic_tree:
		return "parallel_dynamic_tree";
	case SimulationMode::parallel_hybrid_tree:
		return "parallel_hybrid_tree";
	case SimulationMode::parallel_partial_tree:
		return "parallel_partial_tree";
	case SimulationMode::parallel_fusion_tree:
		return "parallel_fusion_tree";
	}
	return "unknown";
}

}
