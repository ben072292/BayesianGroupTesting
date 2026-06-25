#include "bgt/detail/model/lattice.hpp"
#include "bgt/detail/runtime/simulation_helpers.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

#ifdef ENABLE_OMP
#include <omp.h>
#endif

namespace bgt::model
{

MPI_Datatype BBPAResult_type;
MPI_Op BBPA_op;

namespace
{
constexpr int kMaxUnrolledScalarVariants = 16;

void subset_zeta_transform(bgt::accumulator_t *upset_mass, int subjects);
#ifdef ENABLE_OMP
void subset_zeta_transform_omp(bgt::accumulator_t *upset_mass, int subjects);
#endif

/// Thread-local scratch for serial BBPA paths.
///
/// The non-OpenMP selectors are called repeatedly during tree construction and
/// do not need to allocate a fresh partition-mass vector for every node. OpenMP
/// paths intentionally keep local vectors because their array reductions need
/// compiler-owned reduction storage.
std::vector<bgt::accumulator_t> &partition_mass_scratch()
{
	thread_local std::vector<bgt::accumulator_t> scratch;
	return scratch;
}

std::vector<bgt::accumulator_t> &upset_mass_scratch()
{
	thread_local std::vector<bgt::accumulator_t> scratch;
	return scratch;
}

/// Return a zeroed scratch buffer with room for experiment/response masses.
bgt::accumulator_t *zeroed_partition_mass_scratch(int partition_size)
{
	std::vector<bgt::accumulator_t> &scratch = partition_mass_scratch();
	scratch.assign(static_cast<std::size_t>(partition_size), bgt::accumulator_t{0.0});
	return scratch.data();
}

/// Score one candidate experiment by L1 distance from the target response mass.
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

/// Accumulate a local posterior partition by its positive subject mask.
///
/// For binary BBPA, response 0 for experiment E means E is a subset of the
/// state's positive mask. After this base histogram is zeta-transformed,
/// positive_mass[E] is exactly that response-0 mass.
///
/// Work is O(local_state_count) for this histogram plus O(n * 2^n) for the
/// zeta transform, where n is the current subject count. The dense buffer is
/// O(2^n), so this is the same exact BBPA objective without an explicit
/// O(2^n experiments * 2^n states) sweep.
inline bgt::accumulator_t build_binary_positive_mask_masses(
	int local_state_count,
	bgt_state_t state_base,
	int curr_subjs,
	const bgt::posterior_t *posterior,
	bgt::accumulator_t *positive_mass)
{
	const bgt_state_t subject_mask = static_cast<bgt_state_t>(bgt::state_count(curr_subjs) - 1);
	bgt::accumulator_t total_mass = bgt::accumulator_t{0.0};
	for (int s_iter = 0; s_iter < local_state_count; s_iter++)
	{
		const bgt_state_t state = state_base + static_cast<bgt_state_t>(s_iter);
		const bgt::accumulator_t mass = static_cast<bgt::accumulator_t>(posterior[s_iter]);
		positive_mass[static_cast<int>(state & subject_mask)] += mass;
		total_mass += mass;
	}
	return total_mass;
}

/// Specialized two-variant feature histogram.
///
/// For `k=2`, the non-empty variant subsets are `{0}`, `{1}`, and `{0,1}`.
/// Building those three masks directly avoids the generic subset-DP loop in the
/// hot state scan for the common multinomial case.
///
/// The state pass is O(local_state_count), followed by three zeta transforms:
/// O(3 * n * 2^n). The generic k-variant routine would build 2^k - 1 feature
/// tables per state, so k=2 is worth specializing because it is common and the
/// required intersections are just mask0, mask1, and mask0 & mask1.
///
/// This changes the workflow from the brute-force O(2^n experiments * 2^(2n)
/// states) sweep to one O(2^(2n)) state aggregation plus O(n * 2^n) transform
/// work. After the transform, every experiment's four response buckets are
/// recovered in constant time from F0(E), F1(E), and F01(E).
inline bgt::accumulator_t build_two_variant_intersection_masses(
	int local_state_count,
	bgt_state_t state_base,
	int curr_subjs,
	const bgt::posterior_t *posterior,
	bgt::accumulator_t *feature_mass)
{
	const int experiment_count = bgt::state_count(curr_subjs);
	const bgt_state_t subject_mask = static_cast<bgt_state_t>(experiment_count - 1);
	bgt::accumulator_t *variant0_mass = feature_mass + experiment_count;
	bgt::accumulator_t *variant1_mass = feature_mass + 2 * experiment_count;
	bgt::accumulator_t *both_variants_mass = feature_mass + 3 * experiment_count;
	bgt::accumulator_t total_mass = bgt::accumulator_t{0.0};

	for (int s_iter = 0; s_iter < local_state_count; s_iter++)
	{
		const bgt_state_t state = state_base + static_cast<bgt_state_t>(s_iter);
		const bgt_state_t variant0_mask = static_cast<bgt_state_t>(state & subject_mask);
		const bgt_state_t variant1_mask = static_cast<bgt_state_t>((state >> curr_subjs) & subject_mask);
		const bgt::accumulator_t mass = static_cast<bgt::accumulator_t>(posterior[s_iter]);
		variant0_mass[static_cast<int>(variant0_mask)] += mass;
		variant1_mass[static_cast<int>(variant1_mask)] += mass;
		both_variants_mass[static_cast<int>(variant0_mask & variant1_mask)] += mass;
		total_mass += mass;
	}

	feature_mass[0] = total_mass;
	return total_mass;
}

#ifdef ENABLE_OMP
/// OpenMP version of the two-variant feature histogram used by optimized BBPA.
inline bgt::accumulator_t build_two_variant_intersection_masses_omp(
	int local_state_count,
	bgt_state_t state_base,
	int curr_subjs,
	const bgt::posterior_t *posterior,
	bgt::accumulator_t *feature_mass)
{
	const int experiment_count = bgt::state_count(curr_subjs);
	const int table_size = 4 * experiment_count;
	const bgt_state_t subject_mask = static_cast<bgt_state_t>(experiment_count - 1);
	const int max_threads = omp_get_max_threads();
	std::vector<bgt::accumulator_t> thread_feature_mass(
		static_cast<std::size_t>(max_threads) * table_size,
		bgt::accumulator_t{0.0});

#pragma omp parallel
	{
		const int thread_id = omp_get_thread_num();
		bgt::accumulator_t *local =
			thread_feature_mass.data() + static_cast<std::size_t>(thread_id) * table_size;
		bgt::accumulator_t *variant0_mass = local + experiment_count;
		bgt::accumulator_t *variant1_mass = local + 2 * experiment_count;
		bgt::accumulator_t *both_variants_mass = local + 3 * experiment_count;
		bgt::accumulator_t local_total = bgt::accumulator_t{0.0};

#pragma omp for schedule(static)
		for (int s_iter = 0; s_iter < local_state_count; s_iter++)
		{
			const bgt_state_t state = state_base + static_cast<bgt_state_t>(s_iter);
			const bgt_state_t variant0_mask = static_cast<bgt_state_t>(state & subject_mask);
			const bgt_state_t variant1_mask = static_cast<bgt_state_t>((state >> curr_subjs) & subject_mask);
			const bgt::accumulator_t mass = static_cast<bgt::accumulator_t>(posterior[s_iter]);
			variant0_mass[static_cast<int>(variant0_mask)] += mass;
			variant1_mass[static_cast<int>(variant1_mask)] += mass;
			both_variants_mass[static_cast<int>(variant0_mask & variant1_mask)] += mass;
			local_total += mass;
		}
		local[0] = local_total;
	}

#pragma omp parallel for schedule(static)
	for (int index = 0; index < table_size; index++)
	{
		bgt::accumulator_t sum = bgt::accumulator_t{0.0};
		for (int thread_id = 0; thread_id < max_threads; thread_id++)
		{
			sum += thread_feature_mass[static_cast<std::size_t>(thread_id) * table_size + index];
		}
		feature_mass[index] = sum;
	}

	return feature_mass[0];
}
#endif

/// Pick the binary BBPA experiment from zeta-transformed positive masses.
inline BBPAResult best_binary_partition_candidate(
	const bgt::accumulator_t *positive_mass,
	int experiment_count,
	bgt::accumulator_t total_mass,
	bgt::accumulator_t target_mass)
{
	BBPAResult best_result;
	for (int experiment = 0; experiment < experiment_count; experiment++)
	{
		const bgt::accumulator_t positive = positive_mass[experiment];
		const bgt::accumulator_t negative = total_mass - positive;
		const bgt::accumulator_t score = std::abs(positive - target_mass) + std::abs(negative - target_mass);
		if (score < best_result.min || (score == best_result.min && experiment < static_cast<int>(best_result.candidate)))
		{
			best_result.min = score;
			best_result.candidate = static_cast<bgt_state_t>(experiment);
		}
	}
	return best_result;
}

/// Accumulate local posterior mass by variant-subset intersection masks.
///
/// For a variant subset B, feature_mass[B, M] stores the posterior mass of
/// states whose subject intersection across variants in B is exactly M. After a
/// subset zeta transform over M, feature_mass[B, E] becomes F_B(E): mass of
/// states where experiment E would be positive for every variant in B. Exact
/// multinomial response buckets are then recovered from the F_B values by
/// inclusion-exclusion over variants.
///
/// Let n be subjects and k be variants. The generic state pass is
/// O(local_state_count * 2^k), because each posterior state contributes to one
/// intersection mask for every non-empty variant subset. The transform phase is
/// O((2^k - 1) * n * 2^n), and the final candidate scan reconstructs response
/// buckets by inclusion-exclusion. For k=2 this dispatches to the direct
/// three-table path above.
inline bgt::accumulator_t build_variant_subset_intersection_masses(
	int variants,
	int local_state_count,
	bgt_state_t state_base,
	int curr_subjs,
	const bgt::posterior_t *posterior,
	bgt::accumulator_t *feature_mass)
{
	if (variants == 2)
	{
		return build_two_variant_intersection_masses(
			local_state_count, state_base, curr_subjs, posterior, feature_mass);
	}

	const int response_count = bgt::state_count(variants);
	const int experiment_count = bgt::state_count(curr_subjs);
	const bgt_state_t subject_mask = static_cast<bgt_state_t>(experiment_count - 1);
	std::vector<bgt_state_t> variant_masks(static_cast<std::size_t>(variants));
	std::vector<bgt_state_t> subset_intersections(static_cast<std::size_t>(response_count));
	bgt::accumulator_t total_mass = bgt::accumulator_t{0.0};

	for (int s_iter = 0; s_iter < local_state_count; s_iter++)
	{
		const bgt_state_t state = state_base + static_cast<bgt_state_t>(s_iter);
		const bgt::accumulator_t mass = static_cast<bgt::accumulator_t>(posterior[s_iter]);
		for (int variant = 0; variant < variants; variant++)
		{
			variant_masks[static_cast<std::size_t>(variant)] =
				static_cast<bgt_state_t>((state >> (variant * curr_subjs)) & subject_mask);
		}

		subset_intersections[0] = subject_mask;
		for (int subset = 1; subset < response_count; subset++)
		{
			const int low_bit = subset & -subset;
			const int variant = __builtin_ctz(static_cast<unsigned int>(low_bit));
			const int previous_subset = subset ^ low_bit;
			const bgt_state_t intersection =
				static_cast<bgt_state_t>(subset_intersections[static_cast<std::size_t>(previous_subset)] &
										 variant_masks[static_cast<std::size_t>(variant)]);
			subset_intersections[static_cast<std::size_t>(subset)] = intersection;
			feature_mass[subset * experiment_count + static_cast<int>(intersection)] += mass;
		}
		total_mass += mass;
	}

	feature_mass[0] = total_mass;
	return total_mass;
}

/// Zeta-transform every non-empty variant-subset feature table.
///
/// After this step, feature_mass[B, E] is the mass of states whose variants in
/// B are all positive on experiment E. The cost is (2^k - 1) independent subset
/// zeta transforms, each O(n * 2^n).
inline void transform_variant_subset_masses(bgt::accumulator_t *feature_mass, int variants, int curr_subjs)
{
	const int response_count = bgt::state_count(variants);
	const int experiment_count = bgt::state_count(curr_subjs);
	for (int subset = 1; subset < response_count; subset++)
	{
		subset_zeta_transform(&feature_mass[subset * experiment_count], curr_subjs);
	}
}

#ifdef ENABLE_OMP
/// OpenMP zeta-transform for every non-empty variant-subset feature table.
inline void transform_variant_subset_masses_omp(bgt::accumulator_t *feature_mass, int variants, int curr_subjs)
{
	const int response_count = bgt::state_count(variants);
	const int experiment_count = bgt::state_count(curr_subjs);
	for (int subset = 1; subset < response_count; subset++)
	{
		subset_zeta_transform_omp(&feature_mass[subset * experiment_count], curr_subjs);
	}
}
#endif

/// Read F_B(E), treating B=0 as the total posterior mass for every experiment.
inline bgt::accumulator_t variant_subset_mass(
	const bgt::accumulator_t *feature_mass,
	int subset,
	int experiment,
	int experiment_count,
	bgt::accumulator_t total_mass)
{
	return subset == 0 ? total_mass : feature_mass[subset * experiment_count + experiment];
}

/// Pick the BBPA experiment for `k=2` using closed-form inclusion-exclusion.
///
/// The three transformed tables give positive0 = F0(E), positive1 = F1(E), and
/// positive_both = F01(E). The four exact response buckets are then:
/// both, variant 1 only, variant 0 only, and neither. This keeps scoring
/// O(1) per experiment after the zeta transforms.
inline BBPAResult best_two_variant_partition_candidate(
	const bgt::accumulator_t *feature_mass,
	int curr_subjs,
	bgt::accumulator_t total_mass,
	bgt::accumulator_t target_mass)
{
	const int experiment_count = bgt::state_count(curr_subjs);
	const bgt::accumulator_t *variant0_mass = feature_mass + experiment_count;
	const bgt::accumulator_t *variant1_mass = feature_mass + 2 * experiment_count;
	const bgt::accumulator_t *both_variants_mass = feature_mass + 3 * experiment_count;
	BBPAResult best_result;

	for (int experiment = 0; experiment < experiment_count; experiment++)
	{
		const bgt::accumulator_t positive0 = variant0_mass[experiment];
		const bgt::accumulator_t positive1 = variant1_mass[experiment];
		const bgt::accumulator_t positive_both = both_variants_mass[experiment];
		const bgt::accumulator_t response0 = positive_both;
		const bgt::accumulator_t response1 = positive1 - positive_both;
		const bgt::accumulator_t response2 = positive0 - positive_both;
		const bgt::accumulator_t response3 = total_mass - positive0 - positive1 + positive_both;
		const bgt::accumulator_t score =
			std::abs(response0 - target_mass) +
			std::abs(response1 - target_mass) +
			std::abs(response2 - target_mass) +
			std::abs(response3 - target_mass);
		if (score < best_result.min || (score == best_result.min && experiment < static_cast<int>(best_result.candidate)))
		{
			best_result.min = score;
			best_result.candidate = static_cast<bgt_state_t>(experiment);
		}
	}
	return best_result;
}

#ifdef ENABLE_OMP
/// OpenMP best-candidate scan for the optimized two-variant BBPA tables.
inline BBPAResult best_two_variant_partition_candidate_omp(
	const bgt::accumulator_t *feature_mass,
	int curr_subjs,
	bgt::accumulator_t total_mass,
	bgt::accumulator_t target_mass)
{
	const int experiment_count = bgt::state_count(curr_subjs);
	const bgt::accumulator_t *variant0_mass = feature_mass + experiment_count;
	const bgt::accumulator_t *variant1_mass = feature_mass + 2 * experiment_count;
	const bgt::accumulator_t *both_variants_mass = feature_mass + 3 * experiment_count;
	BBPAResult best_result;

#pragma omp declare reduction(BBPA_Min:BBPAResult : BBPAResult::min_assign(omp_out, omp_in)) initializer(omp_priv = BBPAResult())
#pragma omp parallel for schedule(static) reduction(BBPA_Min : best_result)
	for (int experiment = 0; experiment < experiment_count; experiment++)
	{
		const bgt::accumulator_t positive0 = variant0_mass[experiment];
		const bgt::accumulator_t positive1 = variant1_mass[experiment];
		const bgt::accumulator_t positive_both = both_variants_mass[experiment];
		const bgt::accumulator_t response0 = positive_both;
		const bgt::accumulator_t response1 = positive1 - positive_both;
		const bgt::accumulator_t response2 = positive0 - positive_both;
		const bgt::accumulator_t response3 = total_mass - positive0 - positive1 + positive_both;
		const bgt::accumulator_t score =
			std::abs(response0 - target_mass) +
			std::abs(response1 - target_mass) +
			std::abs(response2 - target_mass) +
			std::abs(response3 - target_mass);
		if (score < best_result.min ||
			(score == best_result.min && experiment < static_cast<int>(best_result.candidate)))
		{
			best_result.min = score;
			best_result.candidate = static_cast<bgt_state_t>(experiment);
		}
	}
	return best_result;
}
#endif

/// Pick the multinomial BBPA experiment from transformed F_B(E) tables.
///
/// For a fixed response R, positive_variants are the variants that responded
/// positive and missing_variants are the variants that responded negative. The
/// inner submask loop applies inclusion-exclusion over missing variants to turn
/// the cumulative feature masses F_B(E) into one exact response bucket. Summed
/// over all responses this is O(3^k) work per experiment; k=2 uses the closed
/// form fast path.
inline BBPAResult best_multinomial_partition_candidate(
	const bgt::accumulator_t *feature_mass,
	int variants,
	int curr_subjs,
	bgt::accumulator_t total_mass,
	bgt::accumulator_t target_mass)
{
	if (variants == 2)
		return best_two_variant_partition_candidate(feature_mass, curr_subjs, total_mass, target_mass);

	const int response_count = bgt::state_count(variants);
	const int experiment_count = bgt::state_count(curr_subjs);
	const int all_variants = response_count - 1;
	BBPAResult best_result;

	for (int experiment = 0; experiment < experiment_count; experiment++)
	{
		bgt::accumulator_t score = bgt::accumulator_t{0.0};
		for (int response = 0; response < response_count; response++)
		{
			const int positive_variants = (~response) & all_variants;
			const int missing_variants = response;
			bgt::accumulator_t exact_mass = bgt::accumulator_t{0.0};
			for (int extra = missing_variants;; extra = (extra - 1) & missing_variants)
			{
				const int feature_subset = positive_variants | extra;
				const bgt::accumulator_t feature =
					variant_subset_mass(feature_mass, feature_subset, experiment, experiment_count, total_mass);
				if ((__builtin_popcount(static_cast<unsigned int>(extra)) & 1) == 0)
					exact_mass += feature;
				else
					exact_mass -= feature;
				if (extra == 0)
					break;
			}
			score += std::abs(exact_mass - target_mass);
		}
		if (score < best_result.min || (score == best_result.min && experiment < static_cast<int>(best_result.candidate)))
		{
			best_result.min = score;
			best_result.candidate = static_cast<bgt_state_t>(experiment);
		}
	}
	return best_result;
}

/// Scan a contiguous experiment range and return the best BBPA score/candidate.
inline BBPAResult best_partition_candidate(
	const bgt::accumulator_t *partition_mass,
	int experiment_begin,
	int experiment_end,
	int experiment_index_base,
	int response_count,
	bgt::accumulator_t target_mass)
{
	BBPAResult best_result;
	for (int experiment = experiment_begin; experiment < experiment_end; experiment++)
	{
		const int local_experiment = experiment - experiment_index_base;
		const bgt::accumulator_t score =
			score_partition_mass(&partition_mass[local_experiment * response_count], response_count, target_mass);
		if (score < best_result.min || (score == best_result.min && experiment < static_cast<int>(best_result.candidate)))
		{
			best_result.min = score;
			best_result.candidate = static_cast<bgt_state_t>(experiment);
		}
	}
	return best_result;
}

/// In-place zeta transform over the subset lattice.
///
/// On entry, upset_mass[state] is posterior(state). On exit,
/// upset_mass[experiment] is the posterior mass of all states that contain every
/// subject in experiment. This is the binary Op-BHA objective and avoids the
/// O(4^n) explicit experiment/state sweep.
///
/// Example for n=3: after the transform, upset_mass[010] equals the sum of
/// original masses for 010, 011, 110, and 111, because those are exactly the
/// states containing subject 1. Each subject sweep performs 2^(n - 1)
/// additions, so the transform does n * 2^(n - 1) additions: O(n * 2^n).
void subset_zeta_transform(bgt::accumulator_t *upset_mass, int subjects)
{
	const int state_count = bgt::state_count(subjects);
	for (int subject = 0; subject < subjects; subject++)
	{
		const int bit = 1 << subject;
		const int step = bit << 1;
		for (int base = 0; base < state_count; base += step)
		{
			for (int offset = 0; offset < bit; offset++)
			{
				upset_mass[base + offset] += upset_mass[base + offset + bit];
			}
		}
	}
}

#ifdef ENABLE_OMP
/// OpenMP zeta transform. Subject sweeps remain sequential because each sweep
/// depends on the previous one, but blocks within one sweep are independent.
void subset_zeta_transform_omp(bgt::accumulator_t *upset_mass, int subjects)
{
	const int state_count = bgt::state_count(subjects);
	for (int subject = 0; subject < subjects; subject++)
	{
		const int bit = 1 << subject;
		const int step = bit << 1;
#pragma omp parallel for schedule(static)
		for (int base = 0; base < state_count; base += step)
		{
			for (int offset = 0; offset < bit; offset++)
			{
				upset_mass[base + offset] += upset_mass[base + offset + bit];
			}
		}
	}
}
#endif

/// Pick the experiment whose upset mass is closest to one half.
inline bgt_state_t best_op_bha_candidate(const bgt::accumulator_t *upset_mass, int experiment_count)
{
	bgt_state_t candidate = 0;
	bgt::accumulator_t best_distance = bgt::accumulator_t{2.0};
	for (int experiment = 0; experiment < experiment_count; experiment++)
	{
		const bgt::accumulator_t distance = std::abs(upset_mass[experiment] - bgt::accumulator_t{0.5});
		if (distance < best_distance)
		{
			best_distance = distance;
			candidate = static_cast<bgt_state_t>(experiment);
		}
	}
	return candidate;
}

#ifdef ENABLE_OMP
/// OpenMP scan for the experiment whose upset mass is closest to one half.
inline bgt_state_t best_op_bha_candidate_omp(const bgt::accumulator_t *upset_mass, int experiment_count)
{
	BBPAResult best_result;
#pragma omp declare reduction(BBPA_Min:BBPAResult : BBPAResult::min_assign(omp_out, omp_in)) initializer(omp_priv = BBPAResult())
#pragma omp parallel for schedule(static) reduction(BBPA_Min : best_result)
	for (int experiment = 0; experiment < experiment_count; experiment++)
	{
		const bgt::accumulator_t distance = std::abs(upset_mass[experiment] - bgt::accumulator_t{0.5});
		if (distance < best_result.min ||
			(distance == best_result.min && experiment < static_cast<int>(best_result.candidate)))
		{
			best_result.min = distance;
			best_result.candidate = static_cast<bgt_state_t>(experiment);
		}
	}
	return best_result.candidate;
}
#endif

/// Translate a public selector into the response-bucket mass BBPA should target.
bgt::accumulator_t selector_target_mass(bgt::SelectorType selector, int variants)
{
	const bgt::SelectorType resolved = bgt::runtime_detail::resolve_selector(selector, variants);
	if (resolved == bgt::SelectorType::op_bha)
	{
		if (variants != 1)
			throw std::invalid_argument("Op-BHA experiment selection is only supported for binary lattices.");
		return bgt::accumulator_t{0.5};
	}
	if (resolved == bgt::SelectorType::op_bbpa)
		return bgt::accumulator_t{1.0} / static_cast<bgt::accumulator_t>(1 << variants);
	throw std::invalid_argument("unsupported selector.");
}

/// Convert a state word to 0 or 1 for branchless response-bit construction.
///
/// This is intentionally written as a comparison instead of a width-specific bit
/// trick. In optimized Clang/AArch64 builds, `value != 0` lowers to `cmp; cset`,
/// not a conditional branch, for 8/16/32/64-bit state widths. On a new compiler or
/// hardware target, verify with `c++ -O3 -S` and check this helper's assembly for a
/// conditional jump before replacing it with target-specific arithmetic.
inline int nonzero_to_one(bgt_state_t value)
{
	return static_cast<int>(value != 0);
}

/// Compute one response bit for a fixed variant of a candidate experiment/state pair.
template <int Variant>
inline int scalar_partition_bit(bgt_state_t experiment, bgt_state_t state, int curr_subjs)
{
	const bgt_state_t variant_state = state >> (Variant * curr_subjs);
	const bgt_state_t missing = experiment & ~variant_state;
	return nonzero_to_one(missing) << Variant;
}

/// Compile-time unrolled builder for the response partition ID across all variants.
template <int Variants>
struct ScalarPartitionId
{
	/// Return the response bucket index for one experiment/state pair.
	static int compute(bgt_state_t experiment, bgt_state_t state, int curr_subjs)
	{
		return ScalarPartitionId<Variants - 1>::compute(experiment, state, curr_subjs) |
			   scalar_partition_bit<Variants - 1>(experiment, state, curr_subjs);
	}
};

template <>
struct ScalarPartitionId<0>
{
	/// Terminate partition-ID recursion with the identity value for bitwise OR.
	static int compute(bgt_state_t, bgt_state_t, int) { return 0; }
};

/// Runtime-loop fallback for partition-ID construction when variants are not unrolled.
inline int scalar_partition_id_runtime(bgt_state_t experiment, bgt_state_t state, int curr_subjs, int variants)
{
	int partition_id = 0;
	for (int variant = 0; variant < variants; variant++)
	{
		const bgt_state_t variant_state = state >> (variant * curr_subjs);
		const bgt_state_t missing = experiment & ~variant_state;
		partition_id |= nonzero_to_one(missing) << variant;
	}
	return partition_id;
}

/// Accumulate posterior mass into experiment/response buckets with variant-count unrolling.
template <int Variants>
inline void accumulate_partition_mass_unrolled(
	int local_state_count,
	bgt_state_t state_base,
	int experiment_begin,
	int experiment_end,
	int experiment_index_base,
	int curr_subjs,
	int response_count,
	const bgt::posterior_t *posterior,
	bgt::accumulator_t *partition_mass)
{
	for (int s_iter = 0; s_iter < local_state_count; s_iter++)
	{
		const bgt_state_t state = state_base + static_cast<bgt_state_t>(s_iter);
		const bgt::posterior_t mass = posterior[s_iter];
		for (int experiment = experiment_begin; experiment < experiment_end; experiment++)
		{
			// The partition ID is the response bucket this state induces for this experiment.
			const int partition_id = ScalarPartitionId<Variants>::compute(static_cast<bgt_state_t>(experiment), state, curr_subjs);
			partition_mass[(experiment - experiment_index_base) * response_count + partition_id] += mass;
		}
	}
}

/// Accumulate posterior mass with a runtime variant loop for uncommon variant counts.
inline void accumulate_partition_mass_runtime(
	int local_state_count,
	bgt_state_t state_base,
	int experiment_begin,
	int experiment_end,
	int experiment_index_base,
	int curr_subjs,
	int variants,
	int response_count,
	const bgt::posterior_t *posterior,
	bgt::accumulator_t *partition_mass)
{
	for (int s_iter = 0; s_iter < local_state_count; s_iter++)
	{
		const bgt_state_t state = state_base + static_cast<bgt_state_t>(s_iter);
		const bgt::posterior_t mass = posterior[s_iter];
		for (int experiment = experiment_begin; experiment < experiment_end; experiment++)
		{
			// The partition ID is the response bucket this state induces for this experiment.
			const int partition_id = scalar_partition_id_runtime(static_cast<bgt_state_t>(experiment), state, curr_subjs, variants);
			partition_mass[(experiment - experiment_index_base) * response_count + partition_id] += mass;
		}
	}
}

#ifdef ENABLE_OMP
/// OpenMP version of the unrolled mass accumulator using an array reduction.
template <int Variants>
inline void accumulate_partition_mass_unrolled_omp(
	int local_state_count,
	bgt_state_t state_base,
	int experiment_begin,
	int experiment_end,
	int experiment_index_base,
	int curr_subjs,
	int response_count,
	const bgt::posterior_t *posterior,
	bgt::accumulator_t *partition_mass,
	int partition_size)
{
#pragma omp parallel for schedule(static) reduction(+ : partition_mass[ : partition_size])
	for (int s_iter = 0; s_iter < local_state_count; s_iter++)
	{
		const bgt_state_t state = state_base + static_cast<bgt_state_t>(s_iter);
		const bgt::posterior_t mass = posterior[s_iter];
		for (int experiment = experiment_begin; experiment < experiment_end; experiment++)
		{
			// The partition ID is the response bucket this state induces for this experiment.
			const int partition_id = ScalarPartitionId<Variants>::compute(static_cast<bgt_state_t>(experiment), state, curr_subjs);
			partition_mass[(experiment - experiment_index_base) * response_count + partition_id] += mass;
		}
	}
}

/// OpenMP version of the runtime-loop mass accumulator using an array reduction.
inline void accumulate_partition_mass_runtime_omp(
	int local_state_count,
	bgt_state_t state_base,
	int experiment_begin,
	int experiment_end,
	int experiment_index_base,
	int curr_subjs,
	int variants,
	int response_count,
	const bgt::posterior_t *posterior,
	bgt::accumulator_t *partition_mass,
	int partition_size)
{
#pragma omp parallel for schedule(static) reduction(+ : partition_mass[ : partition_size])
	for (int s_iter = 0; s_iter < local_state_count; s_iter++)
	{
		const bgt_state_t state = state_base + static_cast<bgt_state_t>(s_iter);
		const bgt::posterior_t mass = posterior[s_iter];
		for (int experiment = experiment_begin; experiment < experiment_end; experiment++)
		{
			// The partition ID is the response bucket this state induces for this experiment.
			const int partition_id = scalar_partition_id_runtime(static_cast<bgt_state_t>(experiment), state, curr_subjs, variants);
			partition_mass[(experiment - experiment_index_base) * response_count + partition_id] += mass;
		}
	}
}
#endif

/// Dispatch to a compile-time variant specialization when the runtime count matches.
template <int MaxVariants, typename Fn>
bool dispatch_unrolled_variant_count(int variants, Fn &fn)
{
	if constexpr (MaxVariants == 0)
	{
		return false;
	}
	else
	{
		if (variants == MaxVariants)
		{
			fn.template operator()<MaxVariants>();
			return true;
		}
		return dispatch_unrolled_variant_count<MaxVariants - 1>(variants, fn);
	}
}

/// Choose the unrolled scalar accumulator when possible, otherwise use the runtime loop.
inline void accumulate_partition_mass(
	int variants,
	int local_state_count,
	bgt_state_t state_base,
	int experiment_begin,
	int experiment_end,
	int experiment_index_base,
	int curr_subjs,
	int response_count,
	const bgt::posterior_t *posterior,
	bgt::accumulator_t *partition_mass)
{
	auto unrolled = [&]<int Variants>() {
		accumulate_partition_mass_unrolled<Variants>(
			local_state_count,
			state_base,
			experiment_begin,
			experiment_end,
			experiment_index_base,
			curr_subjs,
			response_count,
			posterior,
			partition_mass);
	};
	if (!dispatch_unrolled_variant_count<kMaxUnrolledScalarVariants>(variants, unrolled))
	{
		accumulate_partition_mass_runtime(
			local_state_count,
			state_base,
			experiment_begin,
			experiment_end,
			experiment_index_base,
			curr_subjs,
			variants,
			response_count,
			posterior,
			partition_mass);
	}
}

#ifdef ENABLE_OMP
/// Choose the OpenMP unrolled accumulator when possible, otherwise use the OpenMP runtime loop.
inline void accumulate_partition_mass_omp(
	int variants,
	int local_state_count,
	bgt_state_t state_base,
	int experiment_begin,
	int experiment_end,
	int experiment_index_base,
	int curr_subjs,
	int response_count,
	const bgt::posterior_t *posterior,
	bgt::accumulator_t *partition_mass,
	int partition_size)
{
	auto unrolled = [&]<int Variants>() {
		accumulate_partition_mass_unrolled_omp<Variants>(
			local_state_count,
			state_base,
			experiment_begin,
			experiment_end,
			experiment_index_base,
			curr_subjs,
			response_count,
			posterior,
			partition_mass,
			partition_size);
	};
	if (!dispatch_unrolled_variant_count<kMaxUnrolledScalarVariants>(variants, unrolled))
	{
		accumulate_partition_mass_runtime_omp(
			local_state_count,
			state_base,
			experiment_begin,
			experiment_end,
			experiment_index_base,
			curr_subjs,
			variants,
			response_count,
			posterior,
			partition_mass,
			partition_size);
	}
}
#endif

} // namespace

/// Select an experiment using the configured selector on the fastest local path.
bgt_state_t Lattice::select_experiment(bgt::SelectorType selector) const
{
	const bgt::SelectorType resolved = bgt::runtime_detail::resolve_selector(selector, _variants);
	if (resolved == bgt::SelectorType::op_bha)
	{
		if (_variants != 1)
			throw std::invalid_argument("Op-BHA experiment selection is only supported for binary lattices.");
		return op_bha();
	}
	return BBPA(selector_target_mass(resolved, _variants));
}

/// Select an experiment using the serial local BBPA path.
bgt_state_t Lattice::select_experiment_serial(bgt::SelectorType selector) const
{
	const bgt::SelectorType resolved = bgt::runtime_detail::resolve_selector(selector, _variants);
	if (resolved == bgt::SelectorType::op_bha)
	{
		if (_variants != 1)
			throw std::invalid_argument("Op-BHA experiment selection is only supported for binary lattices.");
		return op_bha_serial();
	}
	return BBPA_serial(selector_target_mass(resolved, _variants));
}

/// Select the binary halving experiment using the fastest local Op-BHA path.
bgt_state_t Lattice::op_bha() const
{
#ifdef ENABLE_OMP
	return op_bha_omp();
#else
	return op_bha_serial();
#endif
}

/// Serial binary Op-BHA using an upset-mass zeta transform.
///
/// This evaluates every candidate experiment exactly, but via the subset zeta
/// transform rather than an experiment-by-state loop. Work is O(n * 2^n) and
/// memory is O(2^n). The final scan selects the experiment with positive-test
/// probability closest to one half.
bgt_state_t Lattice::op_bha_serial() const
{
	if (_variants != 1)
		throw std::invalid_argument("Op-BHA experiment selection is only supported for binary lattices.");
	const int experiment_count = bgt::state_count(_curr_subjs);
	std::vector<bgt::accumulator_t> &upset_mass = upset_mass_scratch();
	upset_mass.resize(static_cast<std::size_t>(experiment_count));
	const bgt::posterior_t *posterior = _posterior.data();
	for (int state = 0; state < experiment_count; state++)
	{
		upset_mass[static_cast<std::size_t>(state)] = static_cast<bgt::accumulator_t>(posterior[state]);
	}
	subset_zeta_transform(upset_mass.data(), _curr_subjs);
	return best_op_bha_candidate(upset_mass.data(), experiment_count);
}

#ifdef ENABLE_OMP
/// OpenMP binary Op-BHA for large replicated lattices.
bgt_state_t Lattice::op_bha_omp() const
{
	if (_variants != 1)
		throw std::invalid_argument("Op-BHA experiment selection is only supported for binary lattices.");
	const int experiment_count = bgt::state_count(_curr_subjs);
	std::vector<bgt::accumulator_t> &upset_mass = upset_mass_scratch();
	upset_mass.resize(static_cast<std::size_t>(experiment_count));
	const bgt::posterior_t *posterior = _posterior.data();
#pragma omp parallel for schedule(static) if (experiment_count >= (1 << 16))
	for (int state = 0; state < experiment_count; state++)
	{
		upset_mass[static_cast<std::size_t>(state)] = static_cast<bgt::accumulator_t>(posterior[state]);
	}
	if (experiment_count >= (1 << 16))
		subset_zeta_transform_omp(upset_mass.data(), _curr_subjs);
	else
		subset_zeta_transform(upset_mass.data(), _curr_subjs);
	return experiment_count >= (1 << 16)
			   ? best_op_bha_candidate_omp(upset_mass.data(), experiment_count)
			   : best_op_bha_candidate(upset_mass.data(), experiment_count);
}
#endif

/// Serial BBPA selector over all candidate experiments in the current lattice.
///
/// Binary k=1 degenerates to the same positive-mask zeta transform as Op-BHA,
/// then scores the two BBPA response buckets. Multinomial k>1 builds
/// variant-subset feature tables, zeta-transforms them, and reconstructs exact
/// response buckets with inclusion-exclusion. The optimized path is exact BBPA;
/// it is not the old explicit brute-force experiment/state sweep.
bgt_state_t Lattice::BBPA_serial(bgt::accumulator_t prob) const
{
	const int experiment_count = bgt::state_count(_curr_subjs);
	const int response_count = 1 << _variants;
	const int partition_size = experiment_count * response_count;
	bgt::accumulator_t *partition_mass = zeroed_partition_mass_scratch(partition_size);
	const bgt::posterior_t *posterior = _posterior.data();

	if (_variants == 1)
	{
		const bgt::accumulator_t total_mass =
			build_binary_positive_mask_masses(total_states(), 0, _curr_subjs, posterior, partition_mass);
		subset_zeta_transform(partition_mass, _curr_subjs);
		return best_binary_partition_candidate(partition_mass, experiment_count, total_mass, prob).candidate;
	}

	const bgt::accumulator_t total_mass =
		build_variant_subset_intersection_masses(_variants, total_states(), 0, _curr_subjs, posterior, partition_mass);
	transform_variant_subset_masses(partition_mass, _variants, _curr_subjs);
	return best_multinomial_partition_candidate(partition_mass, _variants, _curr_subjs, total_mass, prob).candidate;
}

#ifdef ENABLE_OMP
/// OpenMP BBPA selector over all candidate experiments in the current lattice.
bgt_state_t Lattice::BBPA_omp(bgt::accumulator_t prob) const
{
	const int experiment_count = bgt::state_count(_curr_subjs);
	const int response_count = 1 << _variants;
	const int partition_size = experiment_count * response_count;
	std::vector<bgt::accumulator_t> partition_mass_buffer(partition_size, 0.0);
	bgt::accumulator_t *partition_mass = partition_mass_buffer.data();
	const bgt::posterior_t *posterior = _posterior.data();

	accumulate_partition_mass_omp(_variants, total_states(), 0, 0, experiment_count, 0, _curr_subjs, response_count, posterior, partition_mass, partition_size);

	BBPAResult res;
#pragma omp declare reduction(BBPA_Min:BBPAResult : BBPAResult::min_assign(omp_out, omp_in)) initializer(omp_priv = BBPAResult())
#pragma omp parallel for schedule(static) reduction(BBPA_Min : res)
	for (int experiment = 0; experiment < experiment_count; experiment++)
	{
		const bgt::accumulator_t score = score_partition_mass(&partition_mass[experiment * response_count], response_count, prob);
		if (score < res.min)
		{
			res.min = score;
			res.candidate = static_cast<bgt_state_t>(experiment);
		}
	}

	return res.candidate;
}
#endif

/// MPI BBPA selector where each rank scores a slice of candidate experiments.
bgt_state_t Lattice::BBPA_mpi(bgt::accumulator_t prob) const
{
	const int total_experiments = bgt::state_count(_curr_subjs);
	const int response_count = 1 << _variants;
	const int experiment_count = partition_count(total_experiments, rank);
	const int start_experiment = partition_start(total_experiments, rank);
	const int stop_experiment = start_experiment + experiment_count;
	const int partition_size = (experiment_count == 0 ? 1 : experiment_count) * response_count;
	bgt::accumulator_t *partition_mass = zeroed_partition_mass_scratch(partition_size);
	const bgt::posterior_t *posterior = _posterior.data();

	accumulate_partition_mass(_variants, total_states(), 0, start_experiment, stop_experiment, start_experiment, _curr_subjs, response_count, posterior, partition_mass);
	BBPAResult best_result =
		best_partition_candidate(partition_mass, start_experiment, stop_experiment, start_experiment, response_count, prob);
	BGT_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &best_result, 1, BBPAResult_type, BBPA_op, MPI_COMM_WORLD));

	return best_result.candidate;
}

#ifdef ENABLE_OMP
/// MPI plus OpenMP BBPA selector over this rank's candidate experiment slice.
bgt_state_t Lattice::BBPA_mpi_omp(bgt::accumulator_t prob) const
{
	BBPAResult res;
	const int total_experiments = bgt::state_count(_curr_subjs);
	const int response_count = 1 << _variants;
	const int experiment_count = partition_count(total_experiments, rank);
	const int start_experiment = partition_start(total_experiments, rank);
	const int stop_experiment = start_experiment + experiment_count;
	const int partition_size = (experiment_count == 0 ? 1 : experiment_count) * response_count;
	std::vector<bgt::accumulator_t> partition_mass_buffer(partition_size, 0.0);
	bgt::accumulator_t *partition_mass = partition_mass_buffer.data();
	const bgt::posterior_t *posterior = _posterior.data();

	accumulate_partition_mass_omp(_variants, total_states(), 0, start_experiment, stop_experiment, start_experiment, _curr_subjs, response_count, posterior, partition_mass, partition_size);

#pragma omp declare reduction(BBPA_Min:BBPAResult : BBPAResult::min_assign(omp_out, omp_in)) initializer(omp_priv = BBPAResult())
#pragma omp parallel for schedule(static) reduction(BBPA_Min : res)
	for (int experiment = start_experiment; experiment < stop_experiment; experiment++)
	{
		const bgt::accumulator_t score =
			score_partition_mass(&partition_mass[(experiment - start_experiment) * response_count], response_count, prob);
		if (score < res.min)
		{
			res.min = score;
			res.candidate = static_cast<bgt_state_t>(experiment);
		}
	}
	BGT_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &res, 1, BBPAResult_type, BBPA_op, MPI_COMM_WORLD));
	return res.candidate;
}
#endif

/// Dispatch local BBPA to the exact transform implementation.
///
/// The selector uses zeta transforms and inclusion-exclusion because it is
/// mathematically equivalent to explicit BBPA and avoids the O(states * experiments) sweep.
bgt_state_t Lattice::BBPA(bgt::accumulator_t prob) const
{
#ifdef ENABLE_OMP
	if (_variants == 2)
	{
		const int experiment_count = bgt::state_count(_curr_subjs);
		const int response_count = 1 << _variants;
		const int partition_size = experiment_count * response_count;
		std::vector<bgt::accumulator_t> partition_mass_buffer(partition_size, bgt::accumulator_t{0.0});
		bgt::accumulator_t *partition_mass = partition_mass_buffer.data();
		const bgt::posterior_t *posterior = _posterior.data();
		const bgt::accumulator_t total_mass =
			build_two_variant_intersection_masses_omp(total_states(), 0, _curr_subjs, posterior, partition_mass);
		transform_variant_subset_masses_omp(partition_mass, _variants, _curr_subjs);
		return best_two_variant_partition_candidate_omp(partition_mass, _curr_subjs, total_mass, prob).candidate;
	}
#endif
	return BBPA_serial(prob);
}

/// Initialize MPI rank metadata, the BBPA result datatype, and its reduction op.
void Lattice::lattice_mpi_initialize()
{
	// Get the number of processes
	BGT_MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &world_size));
	// Get the rank of the process
	BGT_MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

	BBPAResult::create_mpi_type(&BBPAResult_type);
	BGT_MPI_CHECK(MPI_Type_commit(&BBPAResult_type));
	BGT_MPI_CHECK(MPI_Op_create((MPI_User_function *)&BBPAResult::mpi_reduce, true, &BBPA_op));
}

/// Release MPI datatype and reduction resources used by BBPA.
void Lattice::lattice_mpi_finalize()
{
	// Free datatype
	BGT_MPI_CHECK(MPI_Type_free(&BBPAResult_type));
	// Free reduce op
	BGT_MPI_CHECK(MPI_Op_free(&BBPA_op));
	// Finalize the MPI environment.
}

/// Dispatch distributed BBPA through the exact transform implementation and MPI.
///
/// MPI rank scaling is the production path for distributed lattices because the
/// transform path is exact and avoids the O(states * experiments) sweep.
bgt_state_t DistributedLattice::BBPA(bgt::accumulator_t prob) const
{
	return BBPA_mpi(prob);
}

/// Partitioned-lattice Op-BHA where ranks hold disjoint posterior state partitions.
bgt_state_t DistributedLattice::op_bha() const
{
	return op_bha_mpi();
}

/// Partitioned-lattice binary Op-BHA.
///
/// This is a collective selector primitive for a posterior array already split
/// across ranks. It is not the tree-level dynamic scheduler: binary tree
/// simulation should use `parallel_dynamic_tree`, where workers receive
/// true-state tasks with wildcard P2P and call the local serial Op-BHA selector.
///
/// Each rank fills a dense O(2^n) positive-mask buffer for its local state
/// partition. Rank 0 reduces those buffers, performs the O(n * 2^n) zeta
/// transform and scan once, then broadcasts the single chosen experiment. The
/// communication payload is therefore O(2^n) accumulator values per selection.
bgt_state_t DistributedLattice::op_bha_mpi() const
{
	if (_variants != 1)
		throw std::invalid_argument("Op-BHA experiment selection is only supported for binary lattices.");
	const int experiment_count = bgt::state_count(_curr_subjs);
	std::fill(partition_mass, partition_mass + experiment_count, bgt::accumulator_t{0.0});
	const bgt_state_t state_base = static_cast<bgt_state_t>(partition_start(total_states(), rank));
	const bgt::posterior_t *posterior = _posterior.data();
	for (int state = 0; state < total_states_per_rank(); state++)
	{
		partition_mass[static_cast<int>(state_base + static_cast<bgt_state_t>(state))] =
			static_cast<bgt::accumulator_t>(posterior[state]);
	}
	if (world_size > 1)
	{
		if (rank == 0)
		{
			BGT_MPI_CHECK(MPI_Reduce(MPI_IN_PLACE, partition_mass, experiment_count, bgt_accumulator_mpi_type(), MPI_SUM, 0, MPI_COMM_WORLD));
		}
		else
		{
			BGT_MPI_CHECK(MPI_Reduce(partition_mass, partition_mass, experiment_count, bgt_accumulator_mpi_type(), MPI_SUM, 0, MPI_COMM_WORLD));
		}
	}
	bgt_state_t candidate = 0;
	if (rank == 0)
	{
#ifdef ENABLE_OMP
		if (experiment_count >= (1 << 16))
			subset_zeta_transform_omp(partition_mass, _curr_subjs);
		else
			subset_zeta_transform(partition_mass, _curr_subjs);
#else
		subset_zeta_transform(partition_mass, _curr_subjs);
#endif
		candidate = best_op_bha_candidate(partition_mass, experiment_count);
	}
	BGT_MPI_CHECK(MPI_Bcast(&candidate, 1, bgt_state_mpi_type(), 0, MPI_COMM_WORLD));
	return candidate;
}

/// Partitioned-lattice exact BBPA.
///
/// Each rank scans only its posterior state partition, but it still contributes
/// to a dense feature table of size O(2^k * 2^n). The table is reduced to rank
/// 0, which performs the zeta transforms, inclusion-exclusion candidate scan,
/// and broadcasts one experiment. This improves state-pass parallelism but the
/// reduction payload and root-side scan remain the scalability limits.
bgt_state_t DistributedLattice::BBPA_mpi(bgt::accumulator_t prob) const
{
	const int experiment_count = bgt::state_count(_curr_subjs);
	const int response_count = 1 << _variants;
	const int partition_size = experiment_count * response_count;
	const bgt::posterior_t *posterior = _posterior.data();
	const bgt_state_t state_base = static_cast<bgt_state_t>(partition_start(total_states(), rank));

	std::fill(partition_mass, partition_mass + partition_size, bgt::accumulator_t{0.0});
	if (_variants == 1)
	{
		bgt::accumulator_t total_mass =
			build_binary_positive_mask_masses(total_states_per_rank(), state_base, _curr_subjs, posterior, partition_mass);
		if (world_size > 1)
		{
			if (rank == 0)
			{
				BGT_MPI_CHECK(MPI_Reduce(MPI_IN_PLACE, partition_mass, experiment_count, bgt_accumulator_mpi_type(), MPI_SUM, 0, MPI_COMM_WORLD));
				BGT_MPI_CHECK(MPI_Reduce(MPI_IN_PLACE, &total_mass, 1, bgt_accumulator_mpi_type(), MPI_SUM, 0, MPI_COMM_WORLD));
			}
			else
			{
				BGT_MPI_CHECK(MPI_Reduce(partition_mass, partition_mass, experiment_count, bgt_accumulator_mpi_type(), MPI_SUM, 0, MPI_COMM_WORLD));
				BGT_MPI_CHECK(MPI_Reduce(&total_mass, &total_mass, 1, bgt_accumulator_mpi_type(), MPI_SUM, 0, MPI_COMM_WORLD));
			}
		}
		bgt_state_t candidate = bgt::invalid_state();
		if (rank == 0)
		{
			subset_zeta_transform(partition_mass, _curr_subjs);
			candidate = best_binary_partition_candidate(partition_mass, experiment_count, total_mass, prob).candidate;
		}
		BGT_MPI_CHECK(MPI_Bcast(&candidate, 1, bgt_state_mpi_type(), 0, MPI_COMM_WORLD));
		memset(reinterpret_cast<void *>(partition_mass), 0x00, partition_size * sizeof(bgt::accumulator_t));
		return candidate;
	}

	bgt::accumulator_t total_mass =
		build_variant_subset_intersection_masses(_variants, total_states_per_rank(), state_base, _curr_subjs, posterior, partition_mass);
	bgt_state_t candidate = bgt::invalid_state();
	if (world_size > 1)
	{
		if (rank == 0)
		{
			BGT_MPI_CHECK(MPI_Reduce(MPI_IN_PLACE, partition_mass, partition_size, bgt_accumulator_mpi_type(), MPI_SUM, 0, MPI_COMM_WORLD));
			total_mass = partition_mass[0];
		}
		else
		{
			BGT_MPI_CHECK(MPI_Reduce(partition_mass, partition_mass, partition_size, bgt_accumulator_mpi_type(), MPI_SUM, 0, MPI_COMM_WORLD));
		}
	}
	if (rank == 0)
	{
		transform_variant_subset_masses(partition_mass, _variants, _curr_subjs);
		candidate = best_multinomial_partition_candidate(partition_mass, _variants, _curr_subjs, total_mass, prob).candidate;
	}
	BGT_MPI_CHECK(MPI_Bcast(&candidate, 1, bgt_state_mpi_type(), 0, MPI_COMM_WORLD));
	memset(reinterpret_cast<void *>(partition_mass), 0x00, partition_size * sizeof(bgt::accumulator_t));
	return candidate;
}

#ifdef ENABLE_OMP
/// Distributed BBPA with OpenMP state accumulation and candidate scoring.
bgt_state_t DistributedLattice::BBPA_mpi_omp(bgt::accumulator_t prob) const
{
	BBPAResult res;
	const int experiment_count = bgt::state_count(_curr_subjs);
	const int response_count = 1 << _variants;
	const int partition_size = experiment_count * response_count;
	const bgt::posterior_t *posterior = _posterior.data();
	const bgt_state_t state_base = static_cast<bgt_state_t>(partition_start(total_states(), rank));

	accumulate_partition_mass_omp(_variants, total_states_per_rank(), state_base, 0, experiment_count, 0, _curr_subjs, response_count, posterior, partition_mass, partition_size);
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
		res = best_partition_candidate(partition_mass, 0, experiment_count, 0, response_count, prob);
	BGT_MPI_CHECK(MPI_Bcast(&res.candidate, 1, bgt_state_mpi_type(), 0, MPI_COMM_WORLD));
	memset(reinterpret_cast<void *>(partition_mass), 0x00, partition_size * sizeof(bgt::accumulator_t));
	return res.candidate;
}
#endif

} // namespace bgt::model
