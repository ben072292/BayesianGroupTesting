#include "bgt/detail/model/lattice.hpp"
#include "support/assertions.hpp"

#include <numeric>
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

void expect_projection_matches(const std::vector<double> &source, const bgt::model::ShrinkPlan &plan, const std::string &label)
{
	std::vector<double> actual(plan.projected_layout.state_count, 0.0);
	bgt::model::project_posterior(source.data(), actual.data(), plan);
	std::vector<double> expected = brute_force_projection(source, plan);
	for (std::size_t i = 0; i < expected.size(); ++i)
		bgt::test::expect_near(actual[i], expected[i], label + " state " + std::to_string(i));
	const double source_sum = std::accumulate(source.begin(), source.end(), 0.0);
	const double actual_sum = std::accumulate(actual.begin(), actual.end(), 0.0);
	bgt::test::expect_near(actual_sum, source_sum, label + " mass conservation");
}
}

int main()
{
	using namespace bgt::model;

	{
		std::vector<double> source{0.1, 0.2, 0.3, 0.4};
		ShrinkPlan plan = make_shrink_plan(2, 2, 1, bgt::state_bit(0), 0);
		expect_projection_matches(source, plan, "binary projection");
	}

	{
		std::vector<double> source(16);
		std::iota(source.begin(), source.end(), 1.0);
		const double sum = std::accumulate(source.begin(), source.end(), 0.0);
		for (double &value : source)
			value /= sum;

		const bgt_state_t subject0_all_variants = bgt::state_bit(0) | bgt::state_bit(2);
		ShrinkPlan plan = make_shrink_plan(2, 2, 2, subject0_all_variants, 0);
		expect_projection_matches(source, plan, "multinomial projection");
	}

	{
		std::vector<double> source(16);
		std::iota(source.begin(), source.end(), 1.0);
		const double sum = std::accumulate(source.begin(), source.end(), 0.0);
		for (double &value : source)
			value /= sum;

		const bgt_state_t already_classified_subject1 = bgt::state_bit(1);
		const bgt_state_t subject2_all_variants = bgt::state_bit(2) | bgt::state_bit(5);
		ShrinkPlan plan = make_shrink_plan(3, 2, 2, subject2_all_variants, already_classified_subject1);
		expect_projection_matches(source, plan, "repeated multinomial projection");
	}

	return 0;
}
