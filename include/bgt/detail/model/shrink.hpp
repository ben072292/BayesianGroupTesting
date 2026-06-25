#pragma once

#include "bgt/detail/common.hpp"

#include <cstddef>
#include <vector>

namespace bgt::model
{

enum class StateWordWidth
{
	bits8 = 8,
	bits16 = 16,
	bits32 = 32,
	bits64 = 64
};

struct StateLayout
{
	int active_atoms = 0;
	StateWordWidth word_width = StateWordWidth::bits8;
	std::size_t state_count = 1;
};

struct ShrinkPlan
{
	int original_subjects = 0;
	int current_subjects = 0;
	int variants = 1;
	int source_atoms = 0;
	int base_atom_count = 0;
	int reduced_atom_count = 0;
	int new_current_subjects = 0;
	bgt_state_t current_classified_atoms = 0;
	bgt_state_t removable_current_atoms = 0;
	bgt_state_t new_classified_subjects = 0;
	std::vector<int> original_to_current_atom_position;
	std::vector<bgt_state_t> base_atom_masks;
	std::vector<bgt_state_t> reduced_atom_masks;
	StateLayout source_layout;
	StateLayout projected_layout;

	bool has_shrink() const { return removable_current_atoms != 0; }
	int projected_state_count_int() const;
};

StateLayout make_state_layout(int active_atoms);
ShrinkPlan make_shrink_plan(int original_subjects, int current_subjects, int variants,
							bgt_state_t classified_atoms, bgt_state_t already_classified_subjects);
bgt_state_t orig_curr_ind_conv(bgt_state_t orig_index_pos, bgt_state_t clas_subjs, int orig_subjs, int variants);
bgt_state_t curr_shrinkable_atoms(bgt_state_t curr_clas_atoms, int curr_subjs, int variants);
bgt_state_t update_clas_subj(bgt_state_t clas_atoms, int orig_subjs, int variants);
bool subj_is_classified(bgt_state_t clas_atoms, int subj_pos, int orig_subjs, int variants);

namespace detail
{
inline bgt_state_t expand_compact_state(bgt_state_t compact_state, const std::vector<bgt_state_t> &atom_masks)
{
	bgt_state_t expanded = 0;
	for (std::size_t i = 0; i < atom_masks.size(); ++i)
	{
		if (compact_state & bgt::state_bit(static_cast<int>(i)))
			expanded |= atom_masks[i];
	}
	return expanded;
}
} // namespace detail

template <typename SourceValue, typename DestinationValue>
void project_posterior(const SourceValue *source, DestinationValue *destination, const ShrinkPlan &plan)
{
	const std::size_t base_states = plan.projected_layout.state_count;
	const std::size_t reduced_states = std::size_t{1} << plan.reduced_atom_count;
	for (std::size_t base = 0; base < base_states; ++base)
	{
		const bgt_state_t base_state = detail::expand_compact_state(static_cast<bgt_state_t>(base), plan.base_atom_masks);
		bgt::accumulator_t sum = 0.0;
		for (std::size_t reduced = 0; reduced < reduced_states; ++reduced)
		{
			const bgt_state_t reduced_state = detail::expand_compact_state(static_cast<bgt_state_t>(reduced), plan.reduced_atom_masks);
			sum += static_cast<bgt::accumulator_t>(source[base_state | reduced_state]);
		}
		destination[base] = static_cast<DestinationValue>(sum);
	}
}

} // namespace bgt::model
