#include "bgt/detail/model/lattice.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace bgt::model
{

namespace
{
int ceil_log2_int(int value)
{
	if (value <= 1)
		return 0;
	int ret = 0;
	int capacity = 1;
	while (capacity < value)
	{
		capacity <<= 1;
		++ret;
	}
	return ret;
}

int ceil_div_int(int numerator, int denominator)
{
	return (numerator + denominator - 1) / denominator;
}

bgt_state_t clear_subject_atoms(bgt_state_t atoms, int subject, int subjects, int variants)
{
	for (int variant = 0; variant < variants; ++variant)
		atoms &= ~bgt::state_bit(variant * subjects + subject);
	return atoms;
}


bgt_state_t compact_expanded_state(bgt_state_t expanded_state, const std::vector<bgt_state_t> &atom_masks)
{
	bgt_state_t compact = 0;
	for (std::size_t i = 0; i < atom_masks.size(); ++i)
	{
		if (expanded_state & atom_masks[i])
			compact |= bgt::state_bit(static_cast<int>(i));
	}
	return compact;
}

std::vector<bgt::accumulator_t> compute_global_atom_masses(
	const bgt::posterior_t *posterior,
	int curr_atoms,
	bgt_state_t state_base,
	int local_state_count)
{
	std::vector<bgt::accumulator_t> atom_masses(static_cast<std::size_t>(curr_atoms), 0.0);
	for (int i = 0; i < local_state_count; ++i)
	{
		const bgt_state_t state = state_base + static_cast<bgt_state_t>(i);
		const bgt::posterior_t probability = posterior[i];
		for (int atom = 0; atom < curr_atoms; ++atom)
		{
			if (state & bgt::state_bit(atom))
				atom_masses[static_cast<std::size_t>(atom)] += probability;
		}
	}
	BGT_MPI_CHECK(MPI_Allreduce(
		MPI_IN_PLACE, atom_masses.data(), static_cast<int>(atom_masses.size()),
		bgt_accumulator_mpi_type(), MPI_SUM, MPI_COMM_WORLD));
	return atom_masses;
}
} // namespace

bgt::posterior_t *DistributedLattice::temp_post_prob_holder = nullptr;
MPI_Win DistributedLattice::win;
bgt::accumulator_t *DistributedLattice::partition_mass = nullptr;

namespace
{
std::vector<bgt::posterior_t> temp_post_prob_holder_storage;
std::vector<bgt::accumulator_t> partition_mass_storage;
bool temp_post_prob_window_active = false;
int temp_post_prob_holder_capacity = 0;
}

DistributedLattice::DistributedLattice(int subjs, int variants, bgt::host_probability_t *pi0)
{
	_curr_subjs = subjs;
	_orig_subjs = subjs;
	_variants = variants;
	_pi0 = pi0;
	allocate_posterior_probs(total_states_per_rank());
	prior_probs(pi0);
}

// Fetch a posterior state from its owner rank. This is intentionally a query path,
// not a hot update path.
bgt::host_probability_t DistributedLattice::posterior_prob(bgt_state_t state) const
{
	bgt::host_probability_t ret = -1.0;
	int target_rank = state_to_rank(state);
	int target_offset = state_to_offset(state);
	if (rank == target_rank)
		ret = _posterior.data()[target_offset];
	BGT_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &ret, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD));
	return ret;
}

void DistributedLattice::prior_probs(bgt::host_probability_t *pi0)
{
	bgt::posterior_t *posterior = _posterior.data();
#ifdef ENABLE_OMP
#pragma omp parallel for schedule(static)
#endif
	for (int i = 0; i < total_states_per_rank(); i++)
	{
		posterior[i] = prior_prob(offset_to_state(i), pi0);
	}
}

void DistributedLattice::update_metadata(bgt::host_probability_t thres_up, bgt::host_probability_t thres_lo)
{
	const std::vector<bgt::accumulator_t> atom_masses =
		compute_global_atom_masses(
			_posterior.data(), curr_atoms(),
			static_cast<bgt_state_t>(partition_start(total_states(), rank)),
			total_states_per_rank());
	for (int i = 0; i < curr_atoms(); i++)
	{
		bgt_state_t placement = bgt::state_bit(i);
		if ((_pos_clas_atoms | _neg_clas_atoms) & placement)
			continue; // skip checking since it's already classified as either positive or negative
		const bgt::accumulator_t prob_mass = atom_masses[static_cast<std::size_t>(i)];
		if (prob_mass < thres_lo)
			_pos_clas_atoms |= placement; // classified as positive
		else if (prob_mass > (1 - thres_up))
			_neg_clas_atoms |= placement; // classified as negative
	}
}

bool DistributedLattice::update_metadata_with_shrinking(
	bgt::host_probability_t thres_up,
	bgt::host_probability_t thres_lo)
{
	bgt_state_t clas_atoms = (_pos_clas_atoms | _neg_clas_atoms); // same size as orig layout
	bgt_state_t new_pos_clas_atoms = 0;
	bgt_state_t new_neg_clas_atoms = 0;
	ShrinkPlan old_plan = make_shrink_plan(_orig_subjs, _curr_subjs, _variants, clas_atoms, _clas_subjs);
	const std::vector<bgt::accumulator_t> atom_masses =
		compute_global_atom_masses(
			_posterior.data(), curr_atoms(),
			static_cast<bgt_state_t>(partition_start(total_states(), rank)),
			total_states_per_rank());

	for (int i = 0; i < _orig_subjs * _variants; i++)
	{
		bgt_state_t orig_index = bgt::state_bit(i);
		if ((clas_atoms & orig_index))
			continue;

		const int current_atom_position = old_plan.original_to_current_atom_position[static_cast<std::size_t>(i)];
		if (current_atom_position < 0)
			continue;

		const bgt::accumulator_t prob_mass = atom_masses[static_cast<std::size_t>(current_atom_position)];
		if (prob_mass < thres_lo)
		{
			new_pos_clas_atoms |= orig_index;
		}
		else if (prob_mass > (1 - thres_up))
		{
			new_neg_clas_atoms |= orig_index;
		}
	}
	_pos_clas_atoms |= new_pos_clas_atoms;
	_neg_clas_atoms |= new_neg_clas_atoms;
	clas_atoms = (_pos_clas_atoms | _neg_clas_atoms);
	ShrinkPlan target_plan = make_shrink_plan(_orig_subjs, _curr_subjs, _variants, clas_atoms, _clas_subjs);

	if (is_classified())
	{ // if fully classified, update variables and return;
		_clas_subjs = target_plan.new_classified_subjects;
		_curr_subjs = target_plan.new_current_subjects;
		free_posterior_probs();
		return false;
	}

	if (!target_plan.has_shrink())
		return false;

	const int target_prob_size = target_plan.projected_state_count_int();
	if (target_prob_size >= world_size)
	{
		apply_shrink_plan(target_plan);
		return false;
	}

	const bgt_state_t true_pos_clas_atoms = _pos_clas_atoms;
	const bgt_state_t true_neg_clas_atoms = _neg_clas_atoms;
	const int min_parallel_atoms = ceil_log2_int(world_size);
	const int min_parallel_subjects = std::min(_curr_subjs, ceil_div_int(min_parallel_atoms, _variants));
	const int intermediate_subjects = std::max(target_plan.new_current_subjects, min_parallel_subjects);
	if (intermediate_subjects > target_plan.new_current_subjects && intermediate_subjects < _curr_subjs)
	{
		bgt_state_t partial_classified_atoms = clas_atoms;
		const int subjects_to_withhold = intermediate_subjects - target_plan.new_current_subjects;
		int withheld = 0;
		for (int subject = 0; subject < _orig_subjs && withheld < subjects_to_withhold; ++subject)
		{
			if (subj_is_classified(partial_classified_atoms, subject, _orig_subjs, _variants) &&
				!(_clas_subjs & bgt::state_bit(subject)))
			{
				// Keep enough active subjects to partition work across ranks. For
				// multinomial lattices, withholding a subject must clear every variant
				// atom for that subject; clearing a subject-index bit alone corrupts the
				// original atom layout.
				partial_classified_atoms = clear_subject_atoms(partial_classified_atoms, subject, _orig_subjs, _variants);
				++withheld;
			}
		}
		_pos_clas_atoms = partial_classified_atoms;
		_neg_clas_atoms = partial_classified_atoms;
		ShrinkPlan intermediate_plan = make_shrink_plan(_orig_subjs, _curr_subjs, _variants,
														partial_classified_atoms, _clas_subjs);
		if (intermediate_plan.has_shrink())
			apply_shrink_plan(intermediate_plan);
		_pos_clas_atoms = true_pos_clas_atoms;
		_neg_clas_atoms = true_neg_clas_atoms;
		target_plan = make_shrink_plan(_orig_subjs, _curr_subjs, _variants, clas_atoms, _clas_subjs);
	}

	const int gathered_prob_size = total_states();
	std::vector<int> recvcounts(world_size);
	std::vector<int> displs(world_size);
	for (int r = 0; r < world_size; r++)
	{
		recvcounts[r] = partition_count(gathered_prob_size, r);
		displs[r] = partition_start(gathered_prob_size, r);
	}
	PosteriorBuffer gathered;
	if (rank == 0)
		gathered = PosteriorBuffer::allocate(gathered_prob_size);
	PosteriorBuffer projected = PosteriorBuffer::allocate(target_prob_size);
	BGT_MPI_CHECK(MPI_Gatherv(_posterior.data(), total_states_per_rank(), bgt_posterior_mpi_type(),
							  rank == 0 ? gathered.data() : nullptr, recvcounts.data(), displs.data(), bgt_posterior_mpi_type(),
							  0, MPI_COMM_WORLD));
	if (rank == 0)
	{
		project_posterior(gathered.data(), projected.data(), target_plan);
	}
	BGT_MPI_CHECK(MPI_Bcast(projected.data(), target_prob_size, bgt_posterior_mpi_type(), 0, MPI_COMM_WORLD));
	set_posterior(std::move(projected));
	_clas_subjs = target_plan.new_classified_subjects;
	_curr_subjs = target_plan.new_current_subjects;
	return true;
}

void DistributedLattice::apply_shrink_plan(const ShrinkPlan &plan)
{
	const int reduce_count = plan.reduced_atom_count;
	const int shrunk_total_states = plan.projected_state_count_int();
	const int shrunk_local_count = partition_count(shrunk_total_states, rank);
	const int reduced_state_count = static_cast<int>(make_state_layout(reduce_count).state_count);
	const int holder_size = shrunk_local_count * reduced_state_count;
	if (!temp_post_prob_window_active || temp_post_prob_holder_capacity < holder_size)
	{
		if (temp_post_prob_window_active)
		{
			BGT_MPI_CHECK(MPI_Win_free(&win));
		}
		temp_post_prob_holder_storage.assign(static_cast<std::size_t>(holder_size), 0.0);
		temp_post_prob_holder = temp_post_prob_holder_storage.data();
		temp_post_prob_holder_capacity = holder_size;
		BGT_MPI_CHECK(MPI_Win_create(temp_post_prob_holder, sizeof(bgt::posterior_t) * holder_size, sizeof(bgt::posterior_t), MPI_INFO_NULL, MPI_COMM_WORLD, &win));
		temp_post_prob_window_active = true;
	}
	std::fill(temp_post_prob_holder_storage.begin(),
			  temp_post_prob_holder_storage.begin() + holder_size, 0.0);
	BGT_MPI_CHECK(MPI_Win_fence(0, win));
	const bgt::posterior_t *posterior = _posterior.data();
	for (int i = 0; i < total_states_per_rank(); i++)
	{
		bgt_state_t state = offset_to_state(i);
		bgt_state_t shrunk_state = compact_expanded_state(state, plan.base_atom_masks);
		bgt_state_t pos = compact_expanded_state(state, plan.reduced_atom_masks);
		const int target_rank = partition_rank(shrunk_total_states, static_cast<int>(shrunk_state));
		const int target_offset = static_cast<int>(shrunk_state) - partition_start(shrunk_total_states, target_rank);
		BGT_MPI_CHECK(MPI_Put(&posterior[i], 1, bgt_posterior_mpi_type(), target_rank,
							  target_offset * reduced_state_count + static_cast<int>(pos), 1, bgt_posterior_mpi_type(), win));
	}
	BGT_MPI_CHECK(MPI_Win_fence(0, win));

	PosteriorBuffer projected = PosteriorBuffer::allocate(shrunk_local_count);
	for (int i = 0; i < shrunk_local_count; i++)
	{
		bgt::accumulator_t sum = 0.0;
		for (int j = 0; j < reduced_state_count; j++)
		{
			sum += temp_post_prob_holder[i * reduced_state_count + j];
		}
		projected.data()[i] = static_cast<bgt::posterior_t>(sum);
	}
	set_posterior(std::move(projected));
	_clas_subjs = plan.new_classified_subjects;
	_curr_subjs = plan.new_current_subjects;
}

// Exhaustive traversal is faster than active generation for atoms
bgt::accumulator_t DistributedLattice::get_atom_prob_mass(bgt_state_t atom) const
{
	bgt::accumulator_t ret = 0.0;
	const bgt::posterior_t *posterior = _posterior.data();
	for (int i = 0; i < total_states_per_rank(); i++)
	{
		if ((offset_to_state(i) & atom) == atom)
		{
			ret += posterior[i];
		}
	}
	BGT_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &ret, 1, bgt_accumulator_mpi_type(), MPI_SUM, MPI_COMM_WORLD));
	return ret;
}

void DistributedLattice::update_probs(bgt_state_t experiment, bgt_state_t response, bgt::host_probability_t **dilution)
{
	PosteriorBuffer next = PosteriorBuffer::allocate(total_states_per_rank());
	bgt::posterior_t *ret = next.data();
	const bgt::posterior_t *posterior = _posterior.data();
	bgt::accumulator_t denominator = 0.0;
	for (int i = 0; i < total_states_per_rank(); i++)
	{
		ret[i] = static_cast<bgt::posterior_t>(posterior[i] * response_prob(experiment, response, offset_to_state(i), dilution));
		denominator += ret[i];
	}
	BGT_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &denominator, 1, bgt_accumulator_mpi_type(), MPI_SUM, MPI_COMM_WORLD));
	const bgt::accumulator_t denominator_inv = bgt::accumulator_t{1.0} / denominator;
	for (int i = 0; i < total_states_per_rank(); i++)
	{
		ret[i] = static_cast<bgt::posterior_t>(ret[i] * denominator_inv);
	}
	set_posterior(std::move(next));
}

void DistributedLattice::update_probs_in_place(bgt_state_t experiment, bgt_state_t response, bgt::host_probability_t **dilution)
{
	bgt::accumulator_t denominator = 0.0;
	bgt::posterior_t *posterior = _posterior.data();
	for (int i = 0; i < total_states_per_rank(); i++)
	{
		posterior[i] = static_cast<bgt::posterior_t>(posterior[i] * response_prob(experiment, response, offset_to_state(i), dilution));
		denominator += posterior[i];
	}
	BGT_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &denominator, 1, bgt_accumulator_mpi_type(), MPI_SUM, MPI_COMM_WORLD));
	const bgt::accumulator_t denominator_inv = bgt::accumulator_t{1.0} / denominator;
	for (int i = 0; i < total_states_per_rank(); i++)
	{
		posterior[i] = static_cast<bgt::posterior_t>(posterior[i] * denominator_inv);
	}
}

void DistributedLattice::lattice_mpi_initialize(int subjs, int variants)
{
	Lattice::lattice_mpi_initialize();
	if (partition_mass != nullptr)
	{
		if (!rank)
		{
			BGT_LOG_ERROR(bgt::LogSubsystem::model,
						  "A distributed lattice model is already initialized with %d subjects and %d variants.",
						  _orig_subjs, _variants);
		}
		DistributedLattice::lattice_mpi_finalize();
		throw bgt::Error(bgt::Status::internal_error("distributed lattice model is already initialized", __FILE__, __LINE__));
	}
	partition_mass_storage.assign(static_cast<std::size_t>(1 << (subjs + variants)), 0.0);
	partition_mass = partition_mass_storage.data();
	memset(reinterpret_cast<void *>(partition_mass), 0x00, (1 << (subjs + variants)) * sizeof(bgt::accumulator_t));
}

void DistributedLattice::lattice_mpi_finalize()
{
	Lattice::lattice_mpi_finalize();
	if (temp_post_prob_window_active)
	{
		BGT_MPI_CHECK(MPI_Win_free(&win));
		temp_post_prob_holder_storage.clear();
		temp_post_prob_holder_storage.shrink_to_fit();
		temp_post_prob_holder = nullptr;
		temp_post_prob_holder_capacity = 0;
		temp_post_prob_window_active = false;
	}
	partition_mass_storage.clear();
	partition_mass_storage.shrink_to_fit();
	partition_mass = nullptr;
}

} // namespace bgt::model
