#include "support/assertions.hpp"
#include "bgt/detail/state_encoding.hpp"

#include <cmath>
#include <memory>
#include <numeric>
#include <vector>

namespace
{
constexpr double kNegativeResponse = 0.99;

struct ReferenceOptions
{
	int subjects = 0;
	int variants = 1;
	int search_depth = 1;
	double threshold_up = 0.01;
	double threshold_lo = 0.01;
	double branch_threshold = 0.0;
	bool dilution = false;
	double alpha = 0.99;
	double h = 0.005;
};

struct ReferenceNode
{
	bgt::state_t experiment = 0;
	bgt::state_t response = 0;
	int stage = 0;
	std::vector<double> posterior;
	bgt::state_t positive_mask = 0;
	bgt::state_t negative_mask = 0;
	std::vector<std::unique_ptr<ReferenceNode>> children;
};

double prior_probability(bgt::state_t state, const std::vector<double> &prior, int atoms)
{
	double probability = 1.0;
	for (int atom = 0; atom < atoms; ++atom)
		probability *= (state & bgt::state_bit(atom)) ? 1.0 - prior[static_cast<std::size_t>(atom)] : prior[static_cast<std::size_t>(atom)];
	return probability;
}

double dilution_probability(int pool_size, int missing_count, double alpha, double h)
{
	if (missing_count == 0)
		return alpha;
	const int positives = pool_size - missing_count;
	return 1.0 - alpha * static_cast<double>(missing_count) /
					 (static_cast<double>(positives) * h + static_cast<double>(missing_count));
}

double response_probability(bgt::state_t experiment, bgt::state_t response, bgt::state_t state,
							const ReferenceOptions &options)
{
	double probability = 1.0;
	const int pool_size = bgt::state_popcount(experiment);
	for (int variant = 0; variant < options.variants; ++variant)
	{
		const bgt::state_t variant_state = state >> (variant * options.subjects);
		const bool response_bit = (response & bgt::state_bit(variant)) != 0;
		if (options.dilution)
		{
			const int missing_count = pool_size - bgt::state_popcount(experiment & variant_state);
			const double positive_response = dilution_probability(pool_size, missing_count, options.alpha, options.h);
			probability *= response_bit ? positive_response : 1.0 - positive_response;
		}
		else
		{
			const bool all_present = (experiment & variant_state) == experiment;
			probability *= all_present
							   ? (response_bit ? kNegativeResponse : 1.0 - kNegativeResponse)
							   : (response_bit ? 1.0 - kNegativeResponse : kNegativeResponse);
		}
	}
	return probability;
}

bool subject_classified(bgt::state_t classified_atoms, int subject, const ReferenceOptions &options)
{
	for (int variant = 0; variant < options.variants; ++variant)
	{
		if (!(classified_atoms & bgt::state_bit(variant * options.subjects + subject)))
			return false;
	}
	return true;
}

bool classified(const ReferenceNode &node, const ReferenceOptions &options)
{
	return bgt::state_popcount(node.positive_mask | node.negative_mask) == options.subjects * options.variants;
}

std::vector<int> active_subjects(const ReferenceNode &node, const ReferenceOptions &options)
{
	std::vector<int> subjects;
	const bgt::state_t classified_atoms = node.positive_mask | node.negative_mask;
	for (int subject = 0; subject < options.subjects; ++subject)
	{
		if (!subject_classified(classified_atoms, subject, options))
			subjects.push_back(subject);
	}
	return subjects;
}

double atom_mass(const std::vector<double> &posterior, int atom)
{
	double mass = 0.0;
	for (std::size_t state = 0; state < posterior.size(); ++state)
	{
		if (static_cast<bgt::state_t>(state) & bgt::state_bit(atom))
			mass += posterior[state];
	}
	return mass;
}

void update_classification(ReferenceNode &node, const ReferenceOptions &options)
{
	const bgt::state_t classified_atoms = node.positive_mask | node.negative_mask;
	for (int atom = 0; atom < options.subjects * options.variants; ++atom)
	{
		const bgt::state_t atom_mask = bgt::state_bit(atom);
		if (classified_atoms & atom_mask)
			continue;
		const double mass = atom_mass(node.posterior, atom);
		if (mass < options.threshold_lo)
			node.positive_mask |= atom_mask;
		else if (mass > 1.0 - options.threshold_up)
			node.negative_mask |= atom_mask;
	}
}

bgt::state_t compact_state(bgt::state_t full_state, const std::vector<int> &subjects, const ReferenceOptions &options)
{
	bgt::state_t compact = 0;
	for (int variant = 0; variant < options.variants; ++variant)
	{
		for (std::size_t index = 0; index < subjects.size(); ++index)
		{
			if (full_state & bgt::state_bit(variant * options.subjects + subjects[index]))
				compact |= bgt::state_bit(variant * static_cast<int>(subjects.size()) + static_cast<int>(index));
		}
	}
	return compact;
}

std::vector<double> projected_posterior(const ReferenceNode &node, const std::vector<int> &subjects,
										const ReferenceOptions &options)
{
	std::vector<double> projected(std::size_t{1} << (subjects.size() * options.variants), 0.0);
	for (std::size_t state = 0; state < node.posterior.size(); ++state)
		projected[compact_state(static_cast<bgt::state_t>(state), subjects, options)] += node.posterior[state];
	return projected;
}

int partition_id(bgt::state_t experiment, bgt::state_t compact_state, int active_subject_count, int variants)
{
	int id = 0;
	for (int variant = 0; variant < variants; ++variant)
	{
		const bgt::state_t variant_state = compact_state >> (variant * active_subject_count);
		if ((experiment & variant_state) != experiment)
			id |= static_cast<int>(bgt::state_bit(variant));
	}
	return id;
}

bgt::state_t select_compact_experiment(const std::vector<double> &projected, int active_subject_count, int variants)
{
	const int experiment_count = 1 << active_subject_count;
	const int response_count = 1 << variants;
	const double target = 1.0 / static_cast<double>(response_count);
	bgt::state_t best_experiment = 0;
	double best_score = 2.0;
	for (int experiment = 0; experiment < experiment_count; ++experiment)
	{
		std::vector<double> mass(static_cast<std::size_t>(response_count), 0.0);
		for (std::size_t state = 0; state < projected.size(); ++state)
			mass[static_cast<std::size_t>(partition_id(static_cast<bgt::state_t>(experiment), static_cast<bgt::state_t>(state), active_subject_count, variants))] += projected[state];
		double score = 0.0;
		for (double value : mass)
			score += std::abs(value - target);
		if (score < best_score)
		{
			best_score = score;
			best_experiment = static_cast<bgt::state_t>(experiment);
		}
	}
	return best_experiment;
}

bgt::state_t full_experiment(bgt::state_t compact_experiment, const std::vector<int> &subjects)
{
	bgt::state_t experiment = 0;
	for (std::size_t index = 0; index < subjects.size(); ++index)
	{
		if (compact_experiment & bgt::state_bit(static_cast<int>(index)))
			experiment |= bgt::state_bit(subjects[index]);
	}
	return experiment;
}

std::vector<double> updated_posterior(const std::vector<double> &posterior, bgt::state_t experiment,
									  bgt::state_t response, const ReferenceOptions &options)
{
	std::vector<double> updated(posterior.size(), 0.0);
	double denominator = 0.0;
	for (std::size_t state = 0; state < posterior.size(); ++state)
	{
		updated[state] = posterior[state] * response_probability(experiment, response, static_cast<bgt::state_t>(state), options);
		denominator += updated[state];
	}
	for (double &value : updated)
		value /= denominator;
	return updated;
}

void build_reference_tree(ReferenceNode &node, const ReferenceOptions &options)
{
	if (classified(node, options) || node.stage >= options.search_depth)
		return;
	const std::vector<int> active = active_subjects(node, options);
	const std::vector<double> projected = projected_posterior(node, active, options);
	const bgt::state_t compact_experiment = select_compact_experiment(projected, static_cast<int>(active.size()), options.variants);
	const bgt::state_t experiment = full_experiment(compact_experiment, active);
	const int response_count = 1 << options.variants;
	node.children.resize(static_cast<std::size_t>(response_count));
	for (int response = 0; response < response_count; ++response)
	{
		auto child = std::make_unique<ReferenceNode>();
		child->experiment = experiment;
		child->response = static_cast<bgt::state_t>(response);
		child->stage = node.stage + 1;
		child->posterior = updated_posterior(node.posterior, experiment, static_cast<bgt::state_t>(response), options);
		child->positive_mask = node.positive_mask;
		child->negative_mask = node.negative_mask;
		update_classification(*child, options);
		build_reference_tree(*child, options);
		node.children[static_cast<std::size_t>(response)] = std::move(child);
	}
}

int leaf_count(const ReferenceNode &node)
{
	if (node.children.empty())
		return 1;
	int count = 0;
	for (const auto &child : node.children)
		count += leaf_count(*child);
	return count;
}

void accumulate_for_true_state(const ReferenceNode &node, bgt::state_t true_state, double branch_probability,
							   double prior_probability, const ReferenceOptions &options, bgt::TreeStats &stats)
{
	if (branch_probability < options.branch_threshold)
		return;
	if (node.children.empty())
	{
		const double weighted = branch_probability * prior_probability;
		if (classified(node, options) && node.negative_mask == true_state)
		{
			stats.correct_probability += weighted;
		}
		else if (classified(node, options))
		{
			stats.incorrect_probability += weighted;
			const double total_positive = static_cast<double>(bgt::state_popcount(node.positive_mask));
			const double total_negative = static_cast<double>(bgt::state_popcount(node.negative_mask));
			if (total_positive != 0.0)
				stats.false_positive_probability += static_cast<double>(bgt::state_popcount((node.negative_mask ^ true_state) & true_state)) /
													total_positive * weighted;
			if (total_negative != 0.0)
				stats.false_negative_probability += static_cast<double>(bgt::state_popcount((node.negative_mask ^ true_state) & ~true_state)) /
												   total_negative * weighted;
		}
		else
		{
			stats.unclassified_probability += weighted;
		}
		stats.expected_stages += static_cast<double>(node.stage) * weighted;
		stats.expected_tests += static_cast<double>(node.stage) * weighted;
		return;
	}

	for (const auto &child : node.children)
	{
		const double child_probability = branch_probability *
			response_probability(child->experiment, child->response, true_state, options);
		accumulate_for_true_state(*child, true_state, child_probability, prior_probability, options, stats);
	}
}

bgt::TreeStats run_reference(const std::vector<double> &prior, const ReferenceOptions &options)
{
	ReferenceNode root;
	root.posterior.resize(std::size_t{1} << (options.subjects * options.variants));
	for (std::size_t state = 0; state < root.posterior.size(); ++state)
		root.posterior[state] = prior_probability(static_cast<bgt::state_t>(state), prior, options.subjects * options.variants);
	build_reference_tree(root, options);

	bgt::TreeStats stats;
	stats.total_leaves = leaf_count(root);
	for (std::size_t state = 0; state < root.posterior.size(); ++state)
	{
		const double state_prior = prior_probability(static_cast<bgt::state_t>(state), prior, options.subjects * options.variants);
		accumulate_for_true_state(root, static_cast<bgt::state_t>(state), 1.0, state_prior, options, stats);
	}
	return stats;
}

bgt::TreeStats run_bgt(const std::vector<double> &prior, const ReferenceOptions &reference_options)
{
	bgt::SimulationConfig config;
	config.lattice_type = reference_options.dilution ? bgt::LatticeType::replicated_dilution : bgt::LatticeType::replicated_non_dilution;
	config.subjects = reference_options.subjects;
	config.variants = reference_options.variants;
	config.prior = prior;
	config.options.search_depth = reference_options.search_depth;
	config.options.threshold_up = reference_options.threshold_up;
	config.options.threshold_lo = reference_options.threshold_lo;
	config.options.branch_threshold = reference_options.branch_threshold;
	config.options.provider = bgt::Provider::cpu;
	config.dilution.alpha = reference_options.alpha;
	config.dilution.h = reference_options.h;
	return bgt::run_simulation(config).stats;
}
}

int main()
{
	{
		ReferenceOptions options;
		options.subjects = 2;
		options.variants = 1;
		options.search_depth = 2;
		options.threshold_up = 0.1;
		options.threshold_lo = 0.1;
		const std::vector<double> prior{0.2, 0.7};
		bgt::test::expect_same_stats(run_bgt(prior, options), run_reference(prior, options), "non-dilution no-shrink reference");
	}

	{
		ReferenceOptions options;
		options.subjects = 2;
		options.variants = 1;
		options.search_depth = 2;
		options.threshold_up = 0.1;
		options.threshold_lo = 0.1;
		options.dilution = true;
		const std::vector<double> prior{0.2, 0.7};
		bgt::test::expect_same_stats(run_bgt(prior, options), run_reference(prior, options), "dilution no-shrink reference");
	}

	{
		ReferenceOptions options;
		options.subjects = 2;
		options.variants = 2;
		options.search_depth = 2;
		options.threshold_up = 0.1;
		options.threshold_lo = 0.1;
		const std::vector<double> prior{0.2, 0.7, 0.4, 0.6};
		bgt::test::expect_same_stats(run_bgt(prior, options), run_reference(prior, options), "multinomial no-shrink reference");
	}

	{
		ReferenceOptions options;
		options.subjects = 6;
		options.variants = 2;
		options.search_depth = 6;
		options.threshold_up = 0.01;
		options.threshold_lo = 0.01;
		options.branch_threshold = 0.001;
		const std::vector<double> prior(static_cast<std::size_t>(options.subjects * options.variants), 0.02);
		bgt::test::expect_same_stats(run_bgt(prior, options), run_reference(prior, options), "multinomial shrink reference N6");
	}

	return 0;
}
