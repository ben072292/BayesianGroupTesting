#include "bgt/detail/model/shrink.hpp"

#include <climits>
#include <limits>
#include <stdexcept>

namespace bgt::model
{

bgt_state_t orig_curr_ind_conv(bgt_state_t orig_index_pos, bgt_state_t clas_subjs, int orig_subjs, int variants)
{
	int ret = static_cast<int>(orig_index_pos);
	for (int i = 0; i < orig_subjs; i++)
	{
		for (int j = 0; j < variants; j++)
		{
			const auto classified_atom = static_cast<bgt_state_t>(j * orig_subjs + i);
			if ((clas_subjs & bgt::state_bit(i)) && (orig_index_pos > classified_atom))
			{
				ret--;
			}
			else if ((clas_subjs & bgt::state_bit(i)) && (orig_index_pos == classified_atom))
			{ // handle the case where already classified subjects are considered
				return 0;
			}
		}
	}
	return bgt::state_bit(ret);
}

bgt_state_t curr_shrinkable_atoms(bgt_state_t curr_clas_atoms, int curr_subjs, int variants)
{
	bgt_state_t ret = 0;
	for (int i = 0; i < curr_subjs; i++)
	{
		bool subj_classified = true;
		for (int j = 0; j < variants; j++)
		{
			if (!(curr_clas_atoms & bgt::state_bit(j * curr_subjs + i)))
			{
				subj_classified = false;
				continue;
			}
		}
		if (subj_classified)
		{
			for (int j = 0; j < variants; j++)
			{
				ret |= bgt::state_bit(j * curr_subjs + i);
			}
		}
	}
	return ret;
}

bgt_state_t update_clas_subj(bgt_state_t clas_atoms, int orig_subjs, int variants)
{
	bgt_state_t ret = 0;
	for (int i = 0; i < orig_subjs; i++)
	{
		bool subj_classified = true;
		for (int j = 0; j < variants; j++)
		{
			if (!(clas_atoms & bgt::state_bit(j * orig_subjs + i)))
			{
				subj_classified = false;
				continue;
			}
		}
		if (subj_classified)
		{
			ret |= bgt::state_bit(i);
		}
	}
	return ret;
}

bool subj_is_classified(bgt_state_t clas_atoms, int subj_pos, int orig_subjs, int variants)
{
	for (int i = 0; i < variants; i++)
	{
		if (!(bgt::state_bit(i * orig_subjs + subj_pos) & clas_atoms))
			return false;
	}
	return true;
}

namespace
{
std::size_t checked_state_count_size(int bits)
{
	if (bits < 0 || bits > bgt::kStateBits || bits >= static_cast<int>(sizeof(std::size_t) * CHAR_BIT))
	{
		throw std::logic_error("state count exceeds configured state encoding width or size_t range.");
	}
	return std::size_t{1} << bits;
}

} // namespace

	int ShrinkPlan::projected_state_count_int() const
	{
		if (projected_layout.state_count > static_cast<std::size_t>(std::numeric_limits<int>::max()))
			throw std::logic_error("projected posterior is too large for MPI/int-indexed storage.");
		return static_cast<int>(projected_layout.state_count);
	}

	StateLayout make_state_layout(int active_atoms)
	{
		StateLayout layout;
		layout.active_atoms = active_atoms;
		if (active_atoms <= 8)
			layout.word_width = StateWordWidth::bits8;
		else if (active_atoms <= 16)
			layout.word_width = StateWordWidth::bits16;
		else if (active_atoms <= 32)
			layout.word_width = StateWordWidth::bits32;
		else
			layout.word_width = StateWordWidth::bits64;
		layout.state_count = checked_state_count_size(active_atoms);
		return layout;
	}

	ShrinkPlan make_shrink_plan(int original_subjects, int current_subjects, int variants,
								bgt_state_t classified_atoms, bgt_state_t already_classified_subjects)
	{
		ShrinkPlan plan;
		plan.original_subjects = original_subjects;
		plan.current_subjects = current_subjects;
		plan.variants = variants;
		plan.source_atoms = current_subjects * variants;
		plan.source_layout = make_state_layout(plan.source_atoms);
		plan.original_to_current_atom_position.assign(static_cast<std::size_t>(original_subjects * variants), -1);

		std::vector<int> original_subject_to_current(static_cast<std::size_t>(original_subjects), -1);
		int current_subject = 0;
		for (int subject = 0; subject < original_subjects; ++subject)
		{
			if (already_classified_subjects & bgt::state_bit(subject))
				continue;
			original_subject_to_current[static_cast<std::size_t>(subject)] = current_subject++;
		}

		for (int subject = 0; subject < original_subjects; ++subject)
		{
			const int current_subject_position = original_subject_to_current[static_cast<std::size_t>(subject)];
			if (current_subject_position < 0)
				continue;
			for (int variant = 0; variant < variants; ++variant)
			{
				const int original_atom = variant * original_subjects + subject;
				const int current_atom = variant * current_subjects + current_subject_position;
				plan.original_to_current_atom_position[static_cast<std::size_t>(original_atom)] = current_atom;
				if (classified_atoms & bgt::state_bit(original_atom))
					plan.current_classified_atoms |= bgt::state_bit(current_atom);
			}
		}

		plan.removable_current_atoms = curr_shrinkable_atoms(plan.current_classified_atoms, current_subjects, variants);
		plan.reduced_atom_count = bgt::state_popcount(plan.removable_current_atoms);
		plan.base_atom_count = plan.source_atoms - plan.reduced_atom_count;
		plan.base_atom_masks.reserve(static_cast<std::size_t>(plan.base_atom_count));
		plan.reduced_atom_masks.reserve(static_cast<std::size_t>(plan.reduced_atom_count));
		for (int atom = 0; atom < plan.source_atoms; ++atom)
		{
			const bgt_state_t atom_mask = bgt::state_bit(atom);
			if (plan.removable_current_atoms & atom_mask)
				plan.reduced_atom_masks.push_back(atom_mask);
			else
				plan.base_atom_masks.push_back(atom_mask);
		}

		plan.new_classified_subjects = update_clas_subj(classified_atoms, original_subjects, variants);
		plan.new_current_subjects = original_subjects - bgt::state_popcount(plan.new_classified_subjects);
		plan.projected_layout = make_state_layout(plan.base_atom_count);
		return plan;
	}

} // namespace bgt::model
