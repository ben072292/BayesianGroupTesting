#include "bgt/detail/model/lattice.hpp"
#include "support/assertions.hpp"

#include <numeric>
#include <random>
#include <vector>

namespace
{
bgt_state_t expand(bgt_state_t compact, const std::vector<bgt_state_t> &masks)
{
	bgt_state_t expanded = 0;
	for (std::size_t i = 0; i < masks.size(); ++i)
	{
		if (compact & bgt::state_bit(static_cast<int>(i)))
			expanded |= masks[i];
	}
	return expanded;
}

std::vector<double> normalized_distribution(std::mt19937 &rng, std::size_t count)
{
	std::uniform_real_distribution<double> distribution(0.01, 1.0);
	std::vector<double> values(count);
	for (double &value : values)
		value = distribution(rng);
	const double sum = std::accumulate(values.begin(), values.end(), 0.0);
	for (double &value : values)
		value /= sum;
	return values;
}

std::vector<double> brute_force_projection(const std::vector<double> &source, const bgt::model::ShrinkPlan &plan)
{
	std::vector<double> expected(plan.projected_layout.state_count, 0.0);
	const std::size_t reduced_states = std::size_t{1} << plan.reduced_atom_count;
	for (std::size_t base = 0; base < expected.size(); ++base)
	{
		const bgt_state_t base_state = expand(static_cast<bgt_state_t>(base), plan.base_atom_masks);
		for (std::size_t reduced = 0; reduced < reduced_states; ++reduced)
		{
			const bgt_state_t reduced_state = expand(static_cast<bgt_state_t>(reduced), plan.reduced_atom_masks);
			expected[base] += source[base_state | reduced_state];
		}
	}
	return expected;
}

double source_atom_mass(const std::vector<double> &source, bgt_state_t atom)
{
	double mass = 0.0;
	for (std::size_t state = 0; state < source.size(); ++state)
	{
		if (static_cast<bgt_state_t>(state) & atom)
			mass += source[state];
	}
	return mass;
}

double projected_atom_mass(const std::vector<double> &projected, int atom)
{
	double mass = 0.0;
	for (std::size_t state = 0; state < projected.size(); ++state)
	{
		if (static_cast<bgt_state_t>(state) & bgt::state_bit(atom))
			mass += projected[state];
	}
	return mass;
}

bgt_state_t random_classification_mask(std::mt19937 &rng, int subjects, int variants, bgt_state_t already_classified_subjects)
{
	std::uniform_int_distribution<int> choice(0, 3);
	bgt_state_t classified_atoms = 0;
	for (int subject = 0; subject < subjects; ++subject)
	{
		if (already_classified_subjects & bgt::state_bit(subject))
			continue;
		const int selected = choice(rng);
		if (selected == 0)
			continue;
		if (selected == 1 && variants > 1)
		{
			classified_atoms |= bgt::state_bit(subject);
			continue;
		}
		for (int variant = 0; variant < variants; ++variant)
			classified_atoms |= bgt::state_bit(variant * subjects + subject);
	}
	return classified_atoms;
}

void check_projection_properties(const std::vector<double> &source, const bgt::model::ShrinkPlan &plan, const std::string &label)
{
	std::vector<double> actual(plan.projected_layout.state_count, 0.0);
	bgt::model::project_posterior(source.data(), actual.data(), plan);
	std::vector<double> expected = brute_force_projection(source, plan);
	for (std::size_t state = 0; state < expected.size(); ++state)
		bgt::test::expect_near(actual[state], expected[state], label + " projected state " + std::to_string(state), 1e-11);

	const double source_sum = std::accumulate(source.begin(), source.end(), 0.0);
	const double actual_sum = std::accumulate(actual.begin(), actual.end(), 0.0);
	bgt::test::expect_near(actual_sum, source_sum, label + " posterior mass", 1e-11);

	for (std::size_t atom = 0; atom < plan.base_atom_masks.size(); ++atom)
	{
		bgt::test::expect_near(
			projected_atom_mass(actual, static_cast<int>(atom)),
			source_atom_mass(source, plan.base_atom_masks[atom]),
			label + " retained atom mass " + std::to_string(atom),
			1e-11);
	}
}
}

int main()
{
	using namespace bgt::model;

	std::mt19937 rng(20260625);
	const struct
	{
		int subjects;
		int variants;
	} cases[] = {
		{3, 1},
		{2, 2},
		{3, 2},
	};

	for (const auto &test_case : cases)
	{
		for (int iteration = 0; iteration < 32; ++iteration)
		{
			bgt_state_t already_classified_subjects = 0;
			if (iteration % 5 == 0 && test_case.subjects > 2)
				already_classified_subjects = bgt::state_bit(1);

			const int current_subjects = test_case.subjects - bgt::state_popcount(already_classified_subjects);
			const int current_atoms = current_subjects * test_case.variants;
			std::vector<double> source = normalized_distribution(rng, std::size_t{1} << current_atoms);
			const bgt_state_t classified_atoms = random_classification_mask(
				rng, test_case.subjects, test_case.variants, already_classified_subjects);
			ShrinkPlan plan = make_shrink_plan(
				test_case.subjects, current_subjects, test_case.variants,
				classified_atoms, already_classified_subjects);
			check_projection_properties(source, plan, "case " + std::to_string(test_case.subjects) + "x" +
												 std::to_string(test_case.variants) + " iter " +
												 std::to_string(iteration));
		}
	}

	ShrinkPlan partial = make_shrink_plan(2, 2, 2, bgt::state_bit(0), 0);
	bgt::test::expect_equal(partial.has_shrink(), false, "partial multinomial classification is not shrinkable");

	ShrinkPlan binary = make_shrink_plan(2, 2, 1, bgt::state_bit(0), 0);
	ShrinkPlan multinomial = make_shrink_plan(2, 2, 2, bgt::state_bit(0) | bgt::state_bit(2), 0);
	bgt::test::expect_equal(binary.reduced_atom_count, 1, "binary shrink removes one atom");
	bgt::test::expect_equal(multinomial.reduced_atom_count, 2, "multinomial shrink removes all subject variants");

	return 0;
}
