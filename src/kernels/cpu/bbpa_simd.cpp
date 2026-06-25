#include "bgt/detail/model/lattice.hpp"

#include <cmath>
#include <cstring>

namespace bgt::model
{

namespace
{
#ifdef ENABLE_SIMD
/// Return the variant count baked into the compiled SIMD BBPA kernel.
int accelerated_bbpa_variant_count()
{
#ifdef NUM_VARIANTS
	return NUM_VARIANTS;
#else
	return 2;
#endif
}

/// Check whether the current lattice shape can use the compiled SIMD BBPA kernel.
bool simd_bbpa_supported(int curr_subjs, int variants)
{
	const int experiment_count = bgt::state_count(curr_subjs);
	return variants == accelerated_bbpa_variant_count() &&
		   experiment_count >= bgt::simd::kStateLanes &&
		   experiment_count % bgt::simd::kStateLanes == 0;
}

/// Convert per-lane nonzero values to all-ones masks for branchless SIMD selection.
///
/// This is the vector equivalent of scalar `nonzero_to_one`, except SIMD bit
/// selection needs an all-ones mask rather than integer 1. ANDing that mask with
/// `state_bit(Variant)` yields exactly the scalar result `nonzero_to_one(x) << Variant`
/// in each lane.
	inline bgt::simd::StateVector simd_nonzero_to_mask(bgt::simd::StateVector value)
	{
		return bgt::simd::not_equal_zero(value);
	}

	/// Score one reduced experiment/response bucket row against the BBPA target.
	inline bgt::accumulator_t score_partition_mass(
		const bgt::accumulator_t *partition_mass,
		int response_count,
		bgt::accumulator_t target_mass)
	{
		bgt::accumulator_t score = 0.0;
		for (int response = 0; response < response_count; response++)
		{
			score += std::abs(partition_mass[response] - target_mass);
		}
		return score;
	}

	/// Scan the globally reduced partition table and return the selected experiment.
	inline BBPAResult best_partition_candidate(
		const bgt::accumulator_t *partition_mass,
		int experiment_count,
		int response_count,
		bgt::accumulator_t target_mass)
	{
		BBPAResult best_result;
		for (int experiment = 0; experiment < experiment_count; experiment++)
		{
			const bgt::accumulator_t score =
				score_partition_mass(&partition_mass[experiment * response_count], response_count, target_mass);
			if (score < best_result.min || (score == best_result.min && experiment < static_cast<int>(best_result.candidate)))
			{
				best_result.min = score;
				best_result.candidate = static_cast<bgt_state_t>(experiment);
			}
		}
		return best_result;
	}

	/// Compute one vectorized response bit for a fixed variant across SIMD experiment lanes.
	template <int Variant>
	inline bgt::simd::StateVector simd_partition_bit(bgt::simd::StateVector experiments, bgt_state_t state, int curr_subjs)
{
	const bgt::simd::StateVector variant_state = bgt::simd::splat_state(static_cast<bgt_state_t>(state >> (Variant * curr_subjs)));
	const bgt::simd::StateVector missing = bgt::simd::bit_and(experiments, bgt::simd::bit_not(variant_state));
	return bgt::simd::bit_and(
		bgt::simd::splat_state(bgt::state_bit(Variant)),
		simd_nonzero_to_mask(missing));
}

/// Compile-time unrolled builder for vectorized response partition IDs.
template <int Variants>
struct SimdPartitionId
{
	/// Return the response bucket index for each SIMD experiment lane.
	static bgt::simd::StateVector compute(bgt::simd::StateVector experiments, bgt_state_t state, int curr_subjs)
	{
		return bgt::simd::bit_or(
			SimdPartitionId<Variants - 1>::compute(experiments, state, curr_subjs),
			simd_partition_bit<Variants - 1>(experiments, state, curr_subjs));
	}
};

template <>
struct SimdPartitionId<0>
{
	/// Terminate partition-ID recursion with zero in every SIMD lane.
	static bgt::simd::StateVector compute(bgt::simd::StateVector, bgt_state_t, int)
	{
		return bgt::simd::zero_state();
	}
};
#endif
} // namespace

#ifdef ENABLE_SIMD
/// Distributed BBPA using SIMD lanes to score batches of candidate experiments.
bgt_state_t DistributedLattice::BBPA_mpi_simd(bgt::accumulator_t prob) const
{
	if (!simd_bbpa_supported(_curr_subjs, _variants))
		return BBPA_mpi(prob);

	const int experiment_count = bgt::state_count(_curr_subjs);
	const int response_count = 1 << _variants;
	const int partition_size = experiment_count * response_count;
	const bgt::posterior_t *posterior = _posterior.data();

	for (int s_iter = 0; s_iter < total_states_per_rank(); s_iter++)
	{
		const bgt_state_t state = offset_to_state(s_iter);
		for (int experiment = 0; experiment < experiment_count; experiment += bgt::simd::kStateLanes)
		{
			// Each SIMD lane represents one candidate experiment.
			const bgt::simd::StateVector experiments = bgt::simd::add_scalar(bgt::simd::lane_offsets(), experiment);
	#ifdef NUM_VARIANTS
			const bgt::simd::StateVector partition_id = SimdPartitionId<NUM_VARIANTS>::compute(experiments, state, _curr_subjs);
	#else
			const bgt::simd::StateVector partition_id = SimdPartitionId<2>::compute(experiments, state, _curr_subjs);
	#endif
			bgt::simd::accumulate_partition(partition_mass, experiments, response_count, partition_id, posterior[s_iter]);
		}
	}

	bgt_state_t candidate = bgt::invalid_state();
	if (world_size > 1)
	{
		if (rank == 0)
		{
			BGT_MPI_CHECK(MPI_Reduce(MPI_IN_PLACE, partition_mass, partition_size, bgt_accumulator_mpi_type(), MPI_SUM, 0, MPI_COMM_WORLD));
		}
		else
		{
			BGT_MPI_CHECK(MPI_Reduce(partition_mass, partition_mass, partition_size, bgt_accumulator_mpi_type(), MPI_SUM, 0, MPI_COMM_WORLD));
		}
	}
	if (rank == 0)
		candidate = best_partition_candidate(partition_mass, experiment_count, response_count, prob).candidate;
	BGT_MPI_CHECK(MPI_Bcast(&candidate, 1, bgt_state_mpi_type(), 0, MPI_COMM_WORLD));
	memset(reinterpret_cast<void *>(partition_mass), 0x00, partition_size * sizeof(bgt::accumulator_t));
	return candidate;
}
#endif

#if defined(ENABLE_OMP) && defined(ENABLE_SIMD)
/// Distributed BBPA using OpenMP over states and SIMD over candidate experiments.
bgt_state_t DistributedLattice::BBPA_mpi_omp_simd(bgt::accumulator_t prob) const
{
	if (!simd_bbpa_supported(_curr_subjs, _variants))
		return BBPA_mpi_omp(prob);

	BBPAResult res;
	const int experiment_count = bgt::state_count(_curr_subjs);
	const int response_count = 1 << _variants;
	const int partition_size = experiment_count * response_count;
	const bgt::posterior_t *posterior = _posterior.data();

#pragma omp parallel for schedule(static) reduction(+ : partition_mass[ : partition_size])
	for (int s_iter = 0; s_iter < total_states_per_rank(); s_iter++)
	{
		const bgt_state_t state = offset_to_state(s_iter);
		for (int experiment = 0; experiment < experiment_count; experiment += bgt::simd::kStateLanes)
		{
			// Each SIMD lane represents one candidate experiment.
			const bgt::simd::StateVector experiments = bgt::simd::add_scalar(bgt::simd::lane_offsets(), experiment);
	#ifdef NUM_VARIANTS
			const bgt::simd::StateVector partition_id = SimdPartitionId<NUM_VARIANTS>::compute(experiments, state, _curr_subjs);
	#else
			const bgt::simd::StateVector partition_id = SimdPartitionId<2>::compute(experiments, state, _curr_subjs);
	#endif
			bgt::simd::accumulate_partition(partition_mass, experiments, response_count, partition_id, posterior[s_iter]);
		}
	}
	if (world_size > 1)
	{
		if (rank == 0)
		{
			BGT_MPI_CHECK(MPI_Reduce(MPI_IN_PLACE, partition_mass, partition_size, bgt_accumulator_mpi_type(), MPI_SUM, 0, MPI_COMM_WORLD));
		}
		else
		{
			BGT_MPI_CHECK(MPI_Reduce(partition_mass, partition_mass, partition_size, bgt_accumulator_mpi_type(), MPI_SUM, 0, MPI_COMM_WORLD));
		}
	}
	if (rank == 0)
		res = best_partition_candidate(partition_mass, experiment_count, response_count, prob);
	BGT_MPI_CHECK(MPI_Bcast(&res.candidate, 1, bgt_state_mpi_type(), 0, MPI_COMM_WORLD));
	memset(reinterpret_cast<void *>(partition_mass), 0x00, partition_size * sizeof(bgt::accumulator_t));
	return res.candidate;
}
#endif

} // namespace bgt::model
