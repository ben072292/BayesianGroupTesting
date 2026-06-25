#include "bgt/detail/model/lattice.hpp"
#include "support/assertions.hpp"

int main()
{
	using namespace bgt::model;

	{
		StateLayout layout8 = make_state_layout(8);
		bgt::test::expect_equal(static_cast<int>(layout8.word_width), 8, "8-bit state layout");
		bgt::test::expect_equal(layout8.state_count, std::size_t{256}, "8-bit state count");
		if constexpr (bgt::kStateBits >= 9)
		{
			StateLayout layout16 = make_state_layout(9);
			bgt::test::expect_equal(static_cast<int>(layout16.word_width), 16, "16-bit state layout");
		}
	}

	{
		ShrinkPlan plan = make_shrink_plan(3, 3, 1, bgt::state_bit(1), 0);
		bgt::test::expect_equal(plan.current_classified_atoms, bgt::state_bit(1), "binary classified atom");
		bgt::test::expect_equal(plan.removable_current_atoms, bgt::state_bit(1), "binary removable atom");
		bgt::test::expect_equal(plan.base_atom_count, 2, "binary base atom count");
		bgt::test::expect_equal(plan.reduced_atom_count, 1, "binary reduced atom count");
		bgt::test::expect_equal(plan.new_current_subjects, 2, "binary new subject count");
	}

	{
		const bgt_state_t subject0_variant0 = bgt::state_bit(0);
		ShrinkPlan partial = make_shrink_plan(2, 2, 2, subject0_variant0, 0);
		bgt::test::expect_equal(partial.removable_current_atoms, bgt_state_t{0}, "partial multinomial subject does not shrink");

		const bgt_state_t subject0_all_variants = bgt::state_bit(0) | bgt::state_bit(2);
		ShrinkPlan full = make_shrink_plan(2, 2, 2, subject0_all_variants, 0);
		bgt::test::expect_equal(full.removable_current_atoms, static_cast<bgt_state_t>(bgt::state_bit(0) | bgt::state_bit(2)), "full multinomial subject shrinks");
		bgt::test::expect_equal(full.base_atom_count, 2, "multinomial base atom count");
		bgt::test::expect_equal(full.reduced_atom_count, 2, "multinomial reduced atom count");
		bgt::test::expect_equal(full.new_current_subjects, 1, "multinomial new subject count");
	}

	{
		const bgt_state_t already_classified_subject1 = bgt::state_bit(1);
		const bgt_state_t subject2_all_variants = bgt::state_bit(2) | bgt::state_bit(5);
		ShrinkPlan repeated = make_shrink_plan(3, 2, 2, subject2_all_variants, already_classified_subject1);
		bgt::test::expect_equal(repeated.original_to_current_atom_position[2], 1, "repeated shrink variant 0 mapping");
		bgt::test::expect_equal(repeated.original_to_current_atom_position[5], 3, "repeated shrink variant 1 mapping");
		bgt::test::expect_equal(repeated.removable_current_atoms, static_cast<bgt_state_t>(bgt::state_bit(1) | bgt::state_bit(3)), "repeated shrink removable atoms");
		bgt::test::expect_equal(repeated.new_current_subjects, 2, "repeated shrink keeps unrelated unclassified subject");
	}

	{
		clear_posterior_buffer_pool();
		{
			PosteriorBuffer buffer = PosteriorBuffer::allocate(8);
			bgt::test::expect_true(buffer.data() != nullptr, "allocated posterior buffer");
			bgt::test::expect_equal(posterior_pool_cached_buffer_count(8), std::size_t{0}, "pool empty while buffer owned");
		}
		bgt::test::expect_equal(posterior_pool_cached_buffer_count(8), std::size_t{1}, "buffer returned to pool");
		PosteriorBuffer reused = PosteriorBuffer::allocate(8);
		bgt::test::expect_true(reused.data() != nullptr, "reused posterior buffer");
		bgt::test::expect_equal(posterior_pool_cached_buffer_count(8), std::size_t{0}, "pool consumed by reuse");
		clear_posterior_buffer_pool();
	}

	return 0;
}
