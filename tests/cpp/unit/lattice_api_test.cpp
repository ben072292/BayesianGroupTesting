#include "support/assertions.hpp"

#include <cmath>
#include <limits>
#include <vector>

namespace
{

int response_bucket(bgt::state_t experiment, bgt::state_t state, int subjects, int variants)
{
	int response = 0;
	const bgt::state_t subject_mask = static_cast<bgt::state_t>((bgt::state_t{1} << subjects) - 1);
	for (int variant = 0; variant < variants; variant++)
	{
		const bgt::state_t variant_state = static_cast<bgt::state_t>((state >> (variant * subjects)) & subject_mask);
		const bgt::state_t missing = static_cast<bgt::state_t>(experiment & ~variant_state);
		response |= static_cast<int>(missing != 0) << variant;
	}
	return response;
}

bgt::state_t brute_force_bbpa_candidate(const bgt::Lattice &lattice, int subjects, int variants)
{
	const int experiment_count = 1 << subjects;
	const int response_count = 1 << variants;
	const int state_count = 1 << (subjects * variants);
	const double target_mass = 1.0 / static_cast<double>(response_count);
	bgt::state_t best_candidate = 0;
	double best_score = std::numeric_limits<double>::infinity();

	for (int experiment = 0; experiment < experiment_count; experiment++)
	{
		std::vector<double> response_mass(static_cast<std::size_t>(response_count), 0.0);
		for (int state = 0; state < state_count; state++)
		{
			const int response =
				response_bucket(static_cast<bgt::state_t>(experiment), static_cast<bgt::state_t>(state), subjects, variants);
			response_mass[static_cast<std::size_t>(response)] +=
				lattice.posterior_probability(static_cast<bgt::state_t>(state));
		}

		double score = 0.0;
		for (double mass : response_mass)
			score += std::abs(mass - target_mass);
		if (score < best_score)
		{
			best_score = score;
			best_candidate = static_cast<bgt::state_t>(experiment);
		}
	}
	return best_candidate;
}

} // namespace

int main()
{
	{
		std::vector<double> prior{0.2, 0.4};
		bgt::Lattice lattice(bgt::LatticeType::replicated_non_dilution, 2, prior);

		bgt::test::expect_near(lattice.posterior_probability(0), 0.08, "prior state 0");
		bgt::test::expect_near(lattice.posterior_probability(1), 0.12, "prior state 1");
		bgt::test::expect_near(lattice.posterior_probability(2), 0.32, "prior state 2");
		bgt::test::expect_near(lattice.posterior_probability(3), 0.48, "prior state 3");

		bgt::test::expect_near(lattice.upset_probability_mass(1), 0.60, "upset mass experiment 1");
		bgt::test::expect_near(lattice.upset_probability_mass(2), 0.80, "upset mass experiment 2");
		bgt::test::expect_near(lattice.upset_probability_mass(3), 0.48, "upset mass experiment 3");
		bgt::test::expect_equal(lattice.select_experiment(bgt::SelectorType::auto_select), bgt::state_t(3), "auto selector experiment");
		bgt::test::expect_equal(lattice.select_experiment(bgt::SelectorType::op_bha), bgt::state_t(3), "Op-BHA selector experiment");
		bgt::test::expect_equal(lattice.select_experiment(bgt::SelectorType::op_bbpa), bgt::state_t(3), "Op-BBPA selector experiment");

		std::vector<bgt::state_t> lookahead = lattice.select_experiments(2, bgt::SelectorType::op_bha);
		bgt::test::expect_equal(lookahead[0], bgt::state_t(3), "k-lookahead first experiment");
		bgt::test::expect_equal(lookahead[1], bgt::state_t(2), "k-lookahead second experiment");
		std::vector<bgt::state_t> single_bbpa = lattice.select_experiments(1, bgt::SelectorType::op_bbpa);
		bgt::test::expect_equal(single_bbpa[0], bgt::state_t(3), "selector BBPA k=1 experiment");

		lattice.update(1, 1);
		const double denominator = 0.60 * 0.99 + 0.40 * 0.01;
		bgt::test::expect_near(lattice.posterior_probability(0), 0.08 * 0.01 / denominator, "posterior state 0");
		bgt::test::expect_near(lattice.posterior_probability(1), 0.12 * 0.99 / denominator, "posterior state 1");
		bgt::test::expect_near(lattice.posterior_probability(2), 0.32 * 0.01 / denominator, "posterior state 2");
		bgt::test::expect_near(lattice.posterior_probability(3), 0.48 * 0.99 / denominator, "posterior state 3");

		lattice.update_classification(0.01, 0.01);
		bgt::test::expect_equal(lattice.positive_classification_mask(), bgt::state_t(0), "positive classification mask");
		bgt::test::expect_equal(lattice.negative_classification_mask(), bgt::state_t(1), "negative classification mask");
		bgt::test::expect_equal(lattice.is_classified(), false, "classified flag");
	}

	std::vector<double> multinomial_prior{0.11, 0.23, 0.31, 0.41, 0.17, 0.29};
	bgt::Lattice multinomial_lattice(bgt::LatticeType::replicated_non_dilution, 3, 2, multinomial_prior);
	bgt::test::expect_equal(
		multinomial_lattice.select_experiment(bgt::SelectorType::op_bbpa),
		brute_force_bbpa_candidate(multinomial_lattice, 3, 2),
		"multinomial exact-transform BBPA matches brute force");

	return 0;
}
