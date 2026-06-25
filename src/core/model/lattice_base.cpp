#include "bgt/detail/model/lattice.hpp"

#include <memory>
#include <stdexcept>
#include <vector>

namespace bgt::model
{

int Lattice::rank;
int Lattice::world_size;
int Lattice::_orig_subjs;
int Lattice::_variants;
bgt::host_probability_t *Lattice::_pi0;


namespace
{
std::vector<bgt_state_t> &power_set_scratch()
{
	thread_local std::vector<bgt_state_t> scratch;
	return scratch;
}

std::vector<bgt::accumulator_t> compute_atom_masses(
	const bgt::posterior_t *posterior,
	int curr_atoms,
	int total_states)
{
	std::vector<bgt::accumulator_t> atom_masses(static_cast<std::size_t>(curr_atoms), 0.0);
	for (int state = 0; state < total_states; ++state)
	{
		const bgt::posterior_t probability = posterior[state];
		for (int atom = 0; atom < curr_atoms; ++atom)
		{
			if (static_cast<bgt_state_t>(state) & bgt::state_bit(atom))
				atom_masses[static_cast<std::size_t>(atom)] += probability;
		}
	}
	return atom_masses;
}

} // namespace

int Lattice::partition_count(int total, int partition_rank)
{
	const int base = total / world_size;
	const int remainder = total % world_size;
	return base + (partition_rank < remainder ? 1 : 0);
}

int Lattice::partition_start(int total, int partition_rank)
{
	const int base = total / world_size;
	const int remainder = total % world_size;
	return partition_rank * base + (partition_rank < remainder ? partition_rank : remainder);
}

int Lattice::partition_rank(int total, int item)
{
	for (int r = 0; r < world_size; r++)
	{
		const int start = partition_start(total, r);
		const int stop = start + partition_count(total, r);
		if (item >= start && item < stop)
			return r;
	}
	return world_size - 1;
}

Lattice::Lattice(int subjs, int variants, bgt::host_probability_t *pi0)
{
	_curr_subjs = subjs;
	_orig_subjs = subjs;
	_pi0 = pi0;
	_variants = variants;
	allocate_posterior_probs(total_states());
	prior_probs(pi0);
}

Lattice::Lattice(const Lattice &other, lattice_copy_op_t op)
{
	_curr_subjs = other._curr_subjs;
	_pos_clas_atoms = other._pos_clas_atoms;
	_neg_clas_atoms = other._neg_clas_atoms;
	_clas_subjs = other._clas_subjs;
	if (op == SHALLOW_COPY_PROB_DIST)
	{
		borrow_posterior_probs(other.posterior_probs(), other.posterior_prob_count());
	}
	else if (op == DEEP_COPY_PROB_DIST)
	{
		const int count = other.posterior_prob_count();
		allocate_posterior_probs(count);
		for (int i = 0; i < count; i++)
			_posterior.data()[i] = other.posterior_probs()[i];
	}
}

Lattice::~Lattice()
{
	free_posterior_probs();
}

void Lattice::allocate_posterior_probs(int count)
{
	_posterior = PosteriorBuffer::allocate(count);
}

void Lattice::own_posterior_probs(bgt::posterior_t *post_probs, int count)
{
	if (_posterior.data() == post_probs && _posterior.owning())
		return;
	_posterior = PosteriorBuffer::own(post_probs, count);
}

void Lattice::borrow_posterior_probs(bgt::posterior_t *post_probs, int count)
{
	if (_posterior.data() == post_probs && !_posterior.owning())
		return;
	_posterior = PosteriorBuffer::borrow(post_probs, count);
}

PosteriorBuffer Lattice::take_posterior()
{
	return std::move(_posterior);
}

void Lattice::set_posterior(PosteriorBuffer buffer)
{
	_posterior = std::move(buffer);
}

void Lattice::free_posterior_probs()
{
	_posterior.reset();
}

bgt::host_probability_t Lattice::posterior_prob(bgt_state_t state) const
{
	return _posterior.data()[state];
}

bgt_state_t *Lattice::get_up_set(bgt_state_t state, bgt_state_t *ret) const
{
	int index_len = curr_atoms() - bgt::state_popcount(state);
	std::vector<bgt_state_t> &add_index = power_set_scratch();
	add_index.resize(index_len);
	int counter = 0;
	for (int i = 0; i < curr_atoms(); i++)
	{
		if ((state & bgt::state_bit(i)) == 0)
		{
			add_index[counter++] = bgt::state_bit(i);
		}
	}
	generate_power_set_adder(add_index.data(), index_len, state, ret);
	return ret;
}

void Lattice::generate_power_set_adder(bgt_state_t *add_index, int index_len, bgt_state_t state, bgt_state_t *ret) const
{
	int pow_set_size = 1 << index_len;
	int i, j;
	bgt_state_t temp;
	for (i = 0; i < pow_set_size; i++)
	{
		temp = state;
		for (j = 0; j < index_len; j++)
		{
			/*
			 * Check if j-th bit in the counter is set If set then print j-th element from
			 * set
			 */
			if ((i & (1 << j)))
			{
				temp |= add_index[j];
			}
		}
		ret[i] = temp;
	}
}

void Lattice::prior_probs(bgt::host_probability_t *pi0)
{
	int index = total_states();
	bgt::posterior_t *posterior = _posterior.data();
#ifdef ENABLE_OMP
#pragma omp parallel for schedule(static)
#endif
	for (int i = 0; i < index; i++)
	{
		posterior[i] = prior_prob(i, pi0);
	}
}

bgt::host_probability_t Lattice::prior_prob(bgt_state_t state, bgt::host_probability_t *pi0) const
{
	bgt::host_probability_t prob = 1.0;
	for (int i = 0; i < curr_atoms(); i++)
	{
		if ((state & bgt::state_bit(i)) == 0)
			prob *= pi0[i];
		else
			prob *= (1.0 - pi0[i]);
	}
	return prob;
}

void Lattice::update_probs(bgt_state_t experiment, bgt_state_t response, bgt::host_probability_t **dilution)
{
	const int total_state = total_states();
	PosteriorBuffer next = PosteriorBuffer::allocate(total_state);
	bgt::posterior_t *ret = next.data();
	const bgt::posterior_t *posterior = _posterior.data();
	bgt::accumulator_t denominator = 0.0;
	for (int i = 0; i < total_state; i++)
	{
		ret[i] = static_cast<bgt::posterior_t>(posterior[i] * response_prob(experiment, response, static_cast<bgt_state_t>(i), dilution));
		denominator += ret[i];
	}
	const bgt::accumulator_t denominator_inv = bgt::accumulator_t{1.0} / denominator;
	for (int i = 0; i < total_state; i++)
	{
		ret[i] = static_cast<bgt::posterior_t>(ret[i] * denominator_inv);
	}
	set_posterior(std::move(next));
}

void Lattice::update_probs_in_place(bgt_state_t experiment, bgt_state_t response, bgt::host_probability_t **dilution)
{
	bgt::accumulator_t denominator = 0.0;
	const int total_state = total_states();
	bgt::posterior_t *posterior = _posterior.data();
	for (int i = 0; i < total_state; i++)
	{
		posterior[i] = static_cast<bgt::posterior_t>(posterior[i] * response_prob(experiment, response, static_cast<bgt_state_t>(i), dilution));
		denominator += posterior[i];
	}
	for (int i = 0; i < total_state; i++)
	{
		posterior[i] = static_cast<bgt::posterior_t>(posterior[i] / denominator);
	}
}

void Lattice::update_metadata(bgt::host_probability_t thres_up, bgt::host_probability_t thres_lo)
{
	const std::vector<bgt::accumulator_t> atom_masses =
		compute_atom_masses(_posterior.data(), curr_atoms(), total_states());
	for (int i = 0; i < curr_atoms(); i++)
	{
		bgt_state_t placement = bgt::state_bit(i);
		if ((_pos_clas_atoms | _neg_clas_atoms) & placement)
			continue; // skip checking since it's already classified as either positive or negative
		const bgt::accumulator_t prob_mass = atom_masses[static_cast<std::size_t>(i)];

		{
			if (prob_mass < thres_lo)
				_pos_clas_atoms |= placement; // classified as positive
			else if (prob_mass > (1 - thres_up))
				_neg_clas_atoms |= placement; // classified as negative
		}
	}
}

bool Lattice::update_metadata_with_shrinking(
	bgt::host_probability_t thres_up,
	bgt::host_probability_t thres_lo)
{
	bgt_state_t clas_atoms = (_pos_clas_atoms | _neg_clas_atoms); // same size as orig layout
	bgt_state_t new_pos_clas_atoms = 0;
	bgt_state_t new_neg_clas_atoms = 0;
	ShrinkPlan old_plan = make_shrink_plan(_orig_subjs, _curr_subjs, _variants, clas_atoms, _clas_subjs);
	const std::vector<bgt::accumulator_t> atom_masses =
		compute_atom_masses(_posterior.data(), curr_atoms(), total_states());

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

	ShrinkPlan plan = make_shrink_plan(_orig_subjs, _curr_subjs, _variants,
									   _pos_clas_atoms | _neg_clas_atoms, _clas_subjs);
	if (plan.has_shrink())
		apply_shrink_plan(plan);
	return false;
}

void Lattice::apply_shrink_plan(const ShrinkPlan &plan)
{
	if (!plan.has_shrink())
		return;
	PosteriorBuffer projected = PosteriorBuffer::allocate(plan.projected_state_count_int());
	project_posterior(_posterior.data(), projected.data(), plan);
	_posterior = std::move(projected);
	_clas_subjs = plan.new_classified_subjects;
	_curr_subjs = plan.new_current_subjects;
}

// Active generation
bgt::accumulator_t Lattice::get_prob_mass(bgt_state_t state) const
{
	bgt::accumulator_t ret = 0.0;
	int n = curr_atoms() - bgt::state_popcount(state), pow_set_size = 1 << n;
	bgt_state_t temp;
	std::vector<bgt_state_t> &add_index = power_set_scratch();
	add_index.resize(n);
	int counter = 0;
	bgt_state_t index;
	for (int i = 0; i < curr_atoms(); i++)
	{
		index = bgt::state_bit(i);
		if (!(state & index))
		{
			add_index[counter++] = index;
		}
	}

	for (int i = 0; i < pow_set_size; i++)
	{
		temp = state;
		for (int j = 0; j < n; j++)
		{
			if ((i & (1 << j)))
			{
				temp |= add_index[j];
			}
		}
		ret += _posterior.data()[temp];
	}

	return ret;
}

// Exhaustive traversal is faster than active generation for atoms
bgt::accumulator_t Lattice::get_atom_prob_mass(bgt_state_t atom) const
{
	bgt::accumulator_t ret = 0.0;
	const bgt::posterior_t *posterior = _posterior.data();
	for (int i = 0; i < bgt::state_count(curr_atoms()); i++)
	{
		if ((static_cast<bgt_state_t>(i) & atom) == atom)
		{
			ret += posterior[i];
		}
	}
	return ret;
}


std::unique_ptr<Lattice> Lattice::to_local()
{
	throw std::logic_error("Only distributed lattice models can be converted to local lattices.");
}

} // namespace bgt::model
