#include "bgt/detail/cuda_provider.hpp"
#include "bgt/detail/cuda_checks.hpp"
#include "bgt/detail/logging_macros.hpp"
#include "bgt/detail/mpi_checks.hpp"
#include "bgt/detail/state_encoding.hpp"

#include <cuda_runtime.h>
#include <mpi.h>
#ifdef BGT_ENABLE_NCCL
#include <nccl.h>
#endif
#ifdef BGT_ENABLE_NCCL_GIN
#include <nccl_device.h>

#include <cuda/atomic>
#endif

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace
{
const bgt::host_probability_t kNegativeResponse = 0.99;
constexpr int kBlockSize = 256;
constexpr bgt::accumulator_t kBinaryHalvingTarget = bgt::accumulator_t{0.5};

template <typename T>
struct CudaBuffer
{
	T *data = nullptr;
	int count = 0;

	CudaBuffer() = default;

	explicit CudaBuffer(int count_)
	{
		ensure(count_);
	}

	void ensure(int requested_count)
	{
		if (requested_count <= count && data != nullptr)
		{
			return;
		}
		reset();
		count = std::max(1, requested_count);
		BGT_CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&data), static_cast<std::size_t>(count) * sizeof(T)));
	}

	void reset()
	{
		if (data != nullptr)
		{
			BGT_CUDA_FREE_NOEXCEPT(data);
			data = nullptr;
			count = 0;
		}
	}

	~CudaBuffer()
	{
		reset();
	}

	CudaBuffer(const CudaBuffer &) = delete;
	CudaBuffer &operator=(const CudaBuffer &) = delete;

	CudaBuffer(CudaBuffer &&other) noexcept : data(other.data), count(other.count)
	{
		other.data = nullptr;
		other.count = 0;
	}

	CudaBuffer &operator=(CudaBuffer &&other) noexcept
	{
		if (this != &other)
		{
			reset();
			data = other.data;
			count = other.count;
			other.data = nullptr;
			other.count = 0;
		}
		return *this;
	}
};

using PosteriorCudaBuffer = CudaBuffer<bgt::posterior_t>;
using AccumulatorCudaBuffer = CudaBuffer<bgt::accumulator_t>;
using HostProbabilityCudaBuffer = CudaBuffer<bgt::host_probability_t>;

struct DeviceBestCandidate
{
	bgt::accumulator_t score;
	int experiment;
};

__device__ bool better_candidate(DeviceBestCandidate candidate, DeviceBestCandidate current)
{
	return candidate.score < current.score ||
		   (candidate.score == current.score && candidate.experiment < current.experiment);
}

struct DeviceTreeNode
{
	int first_child;
	int child_count;
	int experiment;
	int response;
	int stage;
	int positive_mask;
	int negative_mask;
};

constexpr int kStatsCount = 7;
constexpr int kTraversalStackCapacity = 256;
constexpr int kMaxDeterministicResponseCount = 16;
#ifdef BGT_ENABLE_NCCL_GIN
constexpr int kGinCtaCount = 16;
constexpr int kGinThreadsPerCta = 512;
#endif

/// Reusable device memory for one CUDA simulation.
///
/// Tree construction repeatedly performs the same phases at each node: selection,
/// posterior update, classification, and eventual statistics traversal. Keeping
/// those scratch buffers in one workspace avoids repeated cudaMalloc/cudaFree in
/// recursive branches. Posterior buffers are depth-indexed so a parent posterior
/// is not overwritten while its child subtree is being expanded.
struct CudaWorkspace
{
	AccumulatorCudaBuffer accumulator_a;
	AccumulatorCudaBuffer accumulator_b;
	AccumulatorCudaBuffer block_sums;
	AccumulatorCudaBuffer scalar_a;
	AccumulatorCudaBuffer scalar_b;
	CudaBuffer<DeviceBestCandidate> best_blocks;
	CudaBuffer<DeviceBestCandidate> best_final;
	CudaBuffer<int> masks;
	CudaBuffer<int> overflow;
	CudaBuffer<DeviceTreeNode> tree_nodes;
	CudaBuffer<bgt::statistic_t> block_stats;
	CudaBuffer<bgt::statistic_t> local_stats;
	CudaBuffer<bgt::statistic_t> global_stats;
	HostProbabilityCudaBuffer prior;
	HostProbabilityCudaBuffer dilution;
	std::vector<std::unique_ptr<PosteriorCudaBuffer>> posterior_by_depth;

	PosteriorCudaBuffer &posterior_for_depth(int depth, int count)
	{
		if (depth >= static_cast<int>(posterior_by_depth.size()))
		{
			posterior_by_depth.resize(static_cast<std::size_t>(depth + 1));
		}
		if (posterior_by_depth[static_cast<std::size_t>(depth)] == nullptr)
		{
			posterior_by_depth[static_cast<std::size_t>(depth)] = std::make_unique<PosteriorCudaBuffer>();
		}
		posterior_by_depth[static_cast<std::size_t>(depth)]->ensure(count);
		return *posterior_by_depth[static_cast<std::size_t>(depth)];
	}
};

#ifdef BGT_ENABLE_NCCL
void check_nccl(ncclResult_t code, const char *expr, const char *file, int line)
{
	if (code == ncclSuccess)
	{
		return;
	}
	std::string message = "NCCL call failed: ";
	message += expr;
	message += ": ";
	message += ncclGetErrorString(code);
	throw bgt::Error(bgt::Status::backend_error(
		bgt::StatusCode::cuda_error, std::move(message), static_cast<int>(code), file, line));
}

#define BGT_NCCL_CHECK(call) check_nccl((call), #call, __FILE__, __LINE__)

ncclDataType_t nccl_accumulator_type()
{
#if BGT_ACCUMULATOR_BITS == 32
	return ncclFloat;
#elif BGT_ACCUMULATOR_BITS == 64
	return ncclDouble;
#else
#error "Unsupported BGT_ACCUMULATOR_BITS value."
#endif
}

struct NcclContext
{
	ncclComm_t comm;
	cudaStream_t stream;
	bool gin_enabled;
#ifdef BGT_ENABLE_NCCL_GIN
	ncclDevComm dev_comm;
	void *gin_buffer;
	std::size_t gin_capacity_bytes;
	ncclWindow_t gin_window;
	bool gin_window_active;
#endif

	NcclContext(int rank, int world_size, bool enable_gin)
		: comm(nullptr),
		  stream(0),
		  gin_enabled(false)
#ifdef BGT_ENABLE_NCCL_GIN
		  ,
		  dev_comm{},
		  gin_buffer(nullptr),
		  gin_capacity_bytes(0),
		  gin_window{},
		  gin_window_active(false)
#endif
	{
		ncclUniqueId id;
		if (rank == 0)
		{
			BGT_NCCL_CHECK(ncclGetUniqueId(&id));
		}
		BGT_MPI_CHECK(MPI_Bcast(&id, static_cast<int>(sizeof(id)), MPI_BYTE, 0, MPI_COMM_WORLD));
		BGT_NCCL_CHECK(ncclCommInitRank(&comm, world_size, id, rank));
#ifdef BGT_ENABLE_NCCL_GIN
		if (enable_gin && world_size > 1)
		{
			ncclDevCommRequirements requirements = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
			requirements.worldGinBarrierCount = kGinCtaCount;
			requirements.ginSignalCount = kGinCtaCount;
			requirements.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
			BGT_NCCL_CHECK(ncclDevCommCreate(comm, &requirements, &dev_comm));
			gin_enabled = true;
		}
#else
		(void)enable_gin;
#endif
	}

	~NcclContext()
	{
#ifdef BGT_ENABLE_NCCL_GIN
		if (gin_window_active)
		{
			ncclCommWindowDeregister(comm, &gin_window);
		}
		if (gin_buffer != nullptr)
		{
			ncclMemFree(gin_buffer);
		}
		if (gin_enabled)
		{
			ncclDevCommDestroy(comm, &dev_comm);
		}
#endif
		if (comm != nullptr)
		{
			ncclCommDestroy(comm);
		}
	}

	NcclContext(const NcclContext &) = delete;
	NcclContext &operator=(const NcclContext &) = delete;

	void all_reduce_sum(const void *send, void *receive, std::size_t count, ncclDataType_t type);

#ifdef BGT_ENABLE_NCCL_GIN
	void ensure_gin_workspace(std::size_t required_bytes);

	template <typename T>
	bool all_reduce_sum_gin(const T *send, T *receive, std::size_t count);
#endif
};
#else
struct NcclContext
{
};
#endif

struct GpuTreeNode
{
	int experiment;
	int response;
	int stage;
	int positive_mask;
	int negative_mask;
	std::vector<std::unique_ptr<GpuTreeNode>> children;

	GpuTreeNode(int experiment_, int response_, int stage_, int positive_mask_, int negative_mask_)
		: experiment(experiment_),
		  response(response_),
		  stage(stage_),
		  positive_mask(positive_mask_),
		  negative_mask(negative_mask_)
	{
	}

	bool classified(int atoms) const
	{
		return (positive_mask | negative_mask) == ((1 << atoms) - 1);
	}
};

#if BGT_ACCUMULATOR_BITS == 32
__device__ float atomic_add_accumulator(float *address, float val)
{
	return atomicAdd(address, val);
}
#endif

#if BGT_ACCUMULATOR_BITS == 64
#if __CUDA_ARCH__ >= 600
__device__ double atomic_add_accumulator(double *address, double val)
{
	return atomicAdd(address, val);
}
#else
__device__ double atomic_add_accumulator(double *address, double val)
{
	unsigned long long int *address_as_ull = reinterpret_cast<unsigned long long int *>(address);
	unsigned long long int old = *address_as_ull;
	unsigned long long int assumed;
	do
	{
		assumed = old;
		old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
	} while (assumed != old);
	return __longlong_as_double(old);
}
#endif
#endif

__host__ __device__ __forceinline__ int nonzero_to_one_u32(unsigned int value)
{
	// Branchless 0/1 conversion for response-bucket construction. For nonzero v,
	// either v or -v has the top bit set; for zero both are zero. This preserves
	// the old bit-twiddling intent without relying on implementation-defined
	// signed right shifts.
	return static_cast<int>((value | (0u - value)) >> 31);
}

__global__ void set_prior_kernel(bgt::posterior_t *post_probs, const bgt::host_probability_t *prior, int atoms, int total_states)
{
	const int state = blockIdx.x * blockDim.x + threadIdx.x;
	if (state >= total_states)
	{
		return;
	}

	bgt::host_probability_t prob = 1.0;
	for (int i = 0; i < atoms; i++)
	{
		prob *= (state & (1 << i)) == 0 ? prior[i] : (1.0 - prior[i]);
	}
	post_probs[state] = static_cast<bgt::posterior_t>(prob);
}

__global__ void set_prior_partition_kernel(
	bgt::posterior_t *post_probs,
	const bgt::host_probability_t *prior,
	int atoms,
	int state_base,
	int local_states)
{
	const int local_state = blockIdx.x * blockDim.x + threadIdx.x;
	if (local_state >= local_states)
	{
		return;
	}

	const int state = state_base + local_state;
	bgt::host_probability_t prob = 1.0;
	for (int i = 0; i < atoms; i++)
	{
		prob *= (state & (1 << i)) == 0 ? prior[i] : (1.0 - prior[i]);
	}
	post_probs[local_state] = static_cast<bgt::posterior_t>(prob);
}

__global__ void bbpa_mass_kernel(
	const bgt::posterior_t *post_probs,
	bgt::accumulator_t *partition_mass,
	int subjects,
	int variants,
	int total_states,
	int total_experiments)
{
	const long long pair = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
	const long long total_pairs = static_cast<long long>(total_states) * total_experiments;
	if (pair >= total_pairs)
	{
		return;
	}

	const int experiment = static_cast<int>(pair % total_experiments);
	const int state = static_cast<int>(pair / total_experiments);
	int partition_id = 0;
	for (int variant = 0; variant < variants; variant++)
	{
		const int variant_state = state >> (variant * subjects);
		// Response bit is 1 when the state does not contain all subjects in the
		// candidate experiment for this variant.
		const unsigned int missing = static_cast<unsigned int>(experiment & ~variant_state);
		partition_id |= nonzero_to_one_u32(missing) << variant;
	}
	atomic_add_accumulator(
		&partition_mass[experiment * (1 << variants) + partition_id],
		static_cast<bgt::accumulator_t>(post_probs[state]));
}

__global__ void bbpa_best_reduce_small_response_kernel(
	const bgt::posterior_t *post_probs,
	DeviceBestCandidate *experiment_best,
	int subjects,
	int variants,
	int total_states,
	int total_experiments,
	int response_count,
	int active_mask)
{
	const int experiment = blockIdx.x;
	const int tid = threadIdx.x;
	if (experiment >= total_experiments || response_count > kMaxDeterministicResponseCount)
	{
		return;
	}

	bgt::accumulator_t local[kMaxDeterministicResponseCount] = {};
	for (int state = tid; state < total_states; state += blockDim.x)
	{
		int partition_id = 0;
		for (int variant = 0; variant < variants; variant++)
		{
			const int variant_state = state >> (variant * subjects);
			const unsigned int missing = static_cast<unsigned int>(experiment & ~variant_state);
			partition_id |= nonzero_to_one_u32(missing) << variant;
		}
		local[partition_id] += static_cast<bgt::accumulator_t>(post_probs[state]);
	}

	extern __shared__ bgt::accumulator_t shared_response_mass[];
	for (int response = 0; response < response_count; response++)
	{
		shared_response_mass[response * blockDim.x + tid] = local[response];
	}
	__syncthreads();

	for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
	{
		if (tid < stride)
		{
			for (int response = 0; response < response_count; response++)
			{
				shared_response_mass[response * blockDim.x + tid] +=
					shared_response_mass[response * blockDim.x + tid + stride];
			}
		}
		__syncthreads();
	}

	if (tid == 0)
	{
		bgt::accumulator_t score = bgt::accumulator_t{1.0e30};
		if ((experiment & ~active_mask) == 0)
		{
			score = bgt::accumulator_t{0.0};
			const bgt::accumulator_t target =
				bgt::accumulator_t{1.0} / static_cast<bgt::accumulator_t>(response_count);
			for (int response = 0; response < response_count; response++)
			{
				const bgt::accumulator_t diff = shared_response_mass[response * blockDim.x] - target;
				score += diff < bgt::accumulator_t{0.0} ? -diff : diff;
			}
		}
		experiment_best[experiment] = DeviceBestCandidate{score, experiment};
	}
}

__global__ void bbpa_k2_feature_mass_kernel(
	const bgt::posterior_t *post_probs,
	bgt::accumulator_t *feature_mass,
	int subjects,
	int total_experiments)
{
	const int feature = blockIdx.x;
	const int tid = threadIdx.x;
	if (feature >= total_experiments)
	{
		return;
	}

	const int subject_mask = total_experiments - 1;
	bgt::accumulator_t variant0_sum = 0.0;
	bgt::accumulator_t variant1_sum = 0.0;
	bgt::accumulator_t both_sum = 0.0;

	for (int other = tid; other < total_experiments; other += blockDim.x)
	{
		variant0_sum += static_cast<bgt::accumulator_t>(post_probs[feature | (other << subjects)]);
		variant1_sum += static_cast<bgt::accumulator_t>(post_probs[other | (feature << subjects)]);
	}

	const int free_mask = subject_mask & ~feature;
	const int free_count = __popc(static_cast<unsigned int>(free_mask));
	int intersection_state_count = 1;
	for (int i = 0; i < free_count; i++)
	{
		intersection_state_count *= 3;
	}
	for (int code = tid; code < intersection_state_count; code += blockDim.x)
	{
		int remaining_code = code;
		int variant0_state = feature;
		int variant1_state = feature;
		for (int subject = 0; subject < subjects; subject++)
		{
			const int bit = 1 << subject;
			if ((free_mask & bit) == 0)
			{
				continue;
			}
			const int digit = remaining_code % 3;
			remaining_code /= 3;
			if (digit == 1)
			{
				variant0_state |= bit;
			}
			else if (digit == 2)
			{
				variant1_state |= bit;
			}
		}
		both_sum += static_cast<bgt::accumulator_t>(post_probs[variant0_state | (variant1_state << subjects)]);
	}

	extern __shared__ bgt::accumulator_t shared_feature_mass[];
	bgt::accumulator_t *shared_variant0 = shared_feature_mass;
	bgt::accumulator_t *shared_variant1 = shared_feature_mass + blockDim.x;
	bgt::accumulator_t *shared_both = shared_feature_mass + 2 * blockDim.x;
	shared_variant0[tid] = variant0_sum;
	shared_variant1[tid] = variant1_sum;
	shared_both[tid] = both_sum;
	__syncthreads();

	for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
	{
		if (tid < stride)
		{
			shared_variant0[tid] += shared_variant0[tid + stride];
			shared_variant1[tid] += shared_variant1[tid + stride];
			shared_both[tid] += shared_both[tid + stride];
		}
		__syncthreads();
	}

	if (tid == 0)
	{
		feature_mass[feature] = shared_variant0[0];
		feature_mass[total_experiments + feature] = shared_variant1[0];
		feature_mass[2 * total_experiments + feature] = shared_both[0];
	}
}

__global__ void bbpa_k2_feature_zeta_step_kernel(
	bgt::accumulator_t *feature_mass,
	int subject_bit,
	int total_experiments)
{
	const int experiment = blockIdx.x * blockDim.x + threadIdx.x;
	if (experiment >= total_experiments || (experiment & subject_bit) != 0)
	{
		return;
	}

	const int superset = experiment | subject_bit;
	feature_mass[experiment] += feature_mass[superset];
	feature_mass[total_experiments + experiment] += feature_mass[total_experiments + superset];
	feature_mass[2 * total_experiments + experiment] += feature_mass[2 * total_experiments + superset];
}

__global__ void bbpa_k2_best_experiment_kernel(
	const bgt::accumulator_t *feature_mass,
	int total_experiments,
	int active_mask,
	DeviceBestCandidate *block_best)
{
	__shared__ DeviceBestCandidate shared[kBlockSize];
	const int tid = threadIdx.x;
	DeviceBestCandidate best{bgt::accumulator_t{1.0e30}, 0};

	for (int experiment = blockIdx.x * blockDim.x + tid;
		 experiment < total_experiments;
		 experiment += blockDim.x * gridDim.x)
	{
		if ((experiment & ~active_mask) == 0)
		{
			const bgt::accumulator_t total_mass = feature_mass[0];
			const bgt::accumulator_t variant0_contains = feature_mass[experiment];
			const bgt::accumulator_t variant1_contains = feature_mass[total_experiments + experiment];
			const bgt::accumulator_t both_contain = feature_mass[2 * total_experiments + experiment];
			const bgt::accumulator_t target = total_mass * bgt::accumulator_t{0.25};
			const bgt::accumulator_t masses[4] = {
				both_contain,
				variant1_contains - both_contain,
				variant0_contains - both_contain,
				bgt::accumulator_t{1.0} - variant0_contains - variant1_contains + both_contain};
			bgt::accumulator_t score = 0.0;
			for (int response = 0; response < 4; response++)
			{
				const bgt::accumulator_t diff = masses[response] - target;
				score += diff < bgt::accumulator_t{0.0} ? -diff : diff;
			}
			DeviceBestCandidate candidate{score, experiment};
			if (better_candidate(candidate, best))
			{
				best = candidate;
			}
		}
	}

	shared[tid] = best;
	__syncthreads();
	for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
	{
		if (tid < stride && better_candidate(shared[tid + stride], shared[tid]))
		{
			shared[tid] = shared[tid + stride];
		}
		__syncthreads();
	}
	if (tid == 0)
	{
		block_best[blockIdx.x] = shared[0];
	}
}

__global__ void bbpa_mass_partition_kernel(
	const bgt::posterior_t *post_probs,
	bgt::accumulator_t *partition_mass,
	int subjects,
	int variants,
	int state_base,
	int local_states,
	int total_experiments)
{
	const long long pair = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
	const long long total_pairs = static_cast<long long>(local_states) * total_experiments;
	if (pair >= total_pairs)
	{
		return;
	}

	const int experiment = static_cast<int>(pair % total_experiments);
	const int local_state = static_cast<int>(pair / total_experiments);
	const int state = state_base + local_state;
	int partition_id = 0;
	for (int variant = 0; variant < variants; variant++)
	{
		const int variant_state = state >> (variant * subjects);
		// Same branchless response-bucket construction as bbpa_mass_kernel.
		const unsigned int missing = static_cast<unsigned int>(experiment & ~variant_state);
		partition_id |= nonzero_to_one_u32(missing) << variant;
	}
	atomic_add_accumulator(
		&partition_mass[experiment * (1 << variants) + partition_id],
		static_cast<bgt::accumulator_t>(post_probs[local_state]));
}

__global__ void op_bha_copy_posterior_kernel(
	const bgt::posterior_t *post_probs,
	bgt::accumulator_t *upset_mass,
	int total_states)
{
	const int state = blockIdx.x * blockDim.x + threadIdx.x;
	if (state < total_states)
	{
		upset_mass[state] = static_cast<bgt::accumulator_t>(post_probs[state]);
	}
}

__global__ void op_bha_scatter_partition_posterior_kernel(
	const bgt::posterior_t *post_probs,
	bgt::accumulator_t *upset_mass,
	int state_base,
	int local_states)
{
	const int local_state = blockIdx.x * blockDim.x + threadIdx.x;
	if (local_state < local_states)
	{
		upset_mass[state_base + local_state] = static_cast<bgt::accumulator_t>(post_probs[local_state]);
	}
}

__global__ void op_bha_zeta_step_kernel(bgt::accumulator_t *upset_mass, int subject_bit, int total_experiments)
{
	const int experiment = blockIdx.x * blockDim.x + threadIdx.x;
	if (experiment < total_experiments && (experiment & subject_bit) == 0)
	{
		// Zeta transform over the subset lattice. After all subject bits are swept,
		// upset_mass[e] is the posterior mass of all states containing experiment e.
		upset_mass[experiment] += upset_mass[experiment | subject_bit];
	}
}

__global__ void op_bha_best_experiment_kernel(
	const bgt::accumulator_t *upset_mass,
	int total_experiments,
	int active_mask,
	DeviceBestCandidate *block_best)
{
	__shared__ DeviceBestCandidate shared[kBlockSize];
	const int tid = threadIdx.x;
	DeviceBestCandidate best{bgt::accumulator_t{2.0}, 0};

	for (int experiment = blockIdx.x * blockDim.x + tid;
		 experiment < total_experiments;
		 experiment += blockDim.x * gridDim.x)
	{
		if ((experiment & ~active_mask) == 0)
		{
			const bgt::accumulator_t diff = upset_mass[experiment] - kBinaryHalvingTarget;
			const bgt::accumulator_t score = diff < bgt::accumulator_t{0.0} ? -diff : diff;
			DeviceBestCandidate candidate{score, experiment};
			if (better_candidate(candidate, best))
			{
				best = candidate;
			}
		}
	}

	shared[tid] = best;
	__syncthreads();
	for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
	{
		if (tid < stride && better_candidate(shared[tid + stride], shared[tid]))
		{
			shared[tid] = shared[tid + stride];
		}
		__syncthreads();
	}
	if (tid == 0)
	{
		block_best[blockIdx.x] = shared[0];
	}
}

__global__ void bbpa_best_experiment_kernel(
	const bgt::accumulator_t *partition_mass,
	int total_experiments,
	int response_count,
	int active_mask,
	DeviceBestCandidate *block_best)
{
	__shared__ DeviceBestCandidate shared[kBlockSize];
	const int tid = threadIdx.x;
	DeviceBestCandidate best{bgt::accumulator_t{1.0e30}, 0};
	const bgt::accumulator_t target = bgt::accumulator_t{1.0} / static_cast<bgt::accumulator_t>(response_count);

	for (int experiment = blockIdx.x * blockDim.x + tid;
		 experiment < total_experiments;
		 experiment += blockDim.x * gridDim.x)
	{
		if ((experiment & ~active_mask) == 0)
		{
			bgt::accumulator_t score = 0.0;
			for (int response = 0; response < response_count; response++)
			{
				const bgt::accumulator_t diff = partition_mass[experiment * response_count + response] - target;
				score += diff < bgt::accumulator_t{0.0} ? -diff : diff;
			}
			DeviceBestCandidate candidate{score, experiment};
			if (better_candidate(candidate, best))
			{
				best = candidate;
			}
		}
	}

	shared[tid] = best;
	__syncthreads();
	for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
	{
		if (tid < stride && better_candidate(shared[tid + stride], shared[tid]))
		{
			shared[tid] = shared[tid + stride];
		}
		__syncthreads();
	}
	if (tid == 0)
	{
		block_best[blockIdx.x] = shared[0];
	}
}

__global__ void reduce_best_candidate_kernel(
	const DeviceBestCandidate *candidates,
	int candidate_count,
	DeviceBestCandidate *result)
{
	__shared__ DeviceBestCandidate shared[kBlockSize];
	const int tid = threadIdx.x;
	DeviceBestCandidate best{bgt::accumulator_t{1.0e30}, 0};
	for (int index = tid; index < candidate_count; index += blockDim.x)
	{
		const DeviceBestCandidate candidate = candidates[index];
		if (better_candidate(candidate, best))
		{
			best = candidate;
		}
	}
	shared[tid] = best;
	__syncthreads();
	for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
	{
		if (tid < stride && better_candidate(shared[tid + stride], shared[tid]))
		{
			shared[tid] = shared[tid + stride];
		}
		__syncthreads();
	}
	if (tid == 0)
	{
		result[0] = shared[0];
	}
}

#ifdef BGT_ENABLE_NCCL_GIN
/// Exchange one local reduction vector through NCCL GIN into every rank's symmetric inbox.
///
/// The symmetric window layout is type-specific for the current call:
///   [local send vector][world_size inbox vectors].
/// Rank r writes its local send vector into inbox slot r on every peer. The follow-up
/// kernel sums those inbox slots locally, producing the same result as ncclAllReduce(sum).
template <typename T>
__global__ void gin_exchange_allreduce_kernel(
	ncclWindow_t win,
	size_t local_offset,
	size_t inbox_offset,
	size_t count,
	ncclDevComm dev_comm)
{
	const int gin_context = 0;
	const unsigned int signal_index = blockIdx.x;
	ncclGin gin{dev_comm, gin_context};
	const uint64_t signal_value = gin.readSignal(signal_index);
	ncclGinBarrierSession<ncclCoopCta> barrier{ncclCoopCta(), gin, ncclTeamTagWorld(), blockIdx.x};
	barrier.sync(ncclCoopCta(), cuda::memory_order_acquire, ncclGinFenceLevel::None);

	const int tid = threadIdx.x + blockIdx.x * blockDim.x;
	const int thread_count = blockDim.x * gridDim.x;
	const size_t bytes = count * sizeof(T);
	for (int destination = tid; destination < dev_comm.nRanks; destination += thread_count)
	{
		gin.put(
			ncclTeamWorld(dev_comm), destination,
			win, inbox_offset + static_cast<size_t>(dev_comm.rank) * bytes,
			win, local_offset,
			bytes, ncclGin_WeakSignalInc{signal_index});
	}

	const int receiving_cta = (dev_comm.rank % thread_count) / blockDim.x;
	if (blockIdx.x == receiving_cta)
	{
		gin.waitSignal(ncclCoopCta(), signal_index, signal_value + dev_comm.nRanks);
	}
	gin.flush(ncclCoopCta());
	barrier.sync(ncclCoopCta(), cuda::memory_order_release, ncclGinFenceLevel::None);
}

template <typename T>
__global__ void gin_sum_inbox_kernel(
	const T *symmetric_buffer,
	size_t inbox_offset_elements,
	int count,
	int ranks,
	T *receive)
{
	const int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= count)
	{
		return;
	}
	T sum = T{0};
	for (int rank = 0; rank < ranks; rank++)
	{
		sum += symmetric_buffer[inbox_offset_elements + static_cast<size_t>(rank) * count + index];
	}
	receive[index] = sum;
}

void NcclContext::ensure_gin_workspace(std::size_t required_bytes)
{
	if (gin_buffer != nullptr && gin_capacity_bytes >= required_bytes)
	{
		return;
	}
	// NCCL's Device API requires device-side communication buffers to be
	// allocated with ncclMemAlloc and registered as a symmetric window on every
	// rank. Re-registering only happens when a later reduction needs more space.
	if (gin_window_active)
	{
		BGT_NCCL_CHECK(ncclCommWindowDeregister(comm, &gin_window));
		gin_window_active = false;
	}
	if (gin_buffer != nullptr)
	{
		BGT_NCCL_CHECK(ncclMemFree(gin_buffer));
		gin_buffer = nullptr;
		gin_capacity_bytes = 0;
	}
	BGT_NCCL_CHECK(ncclMemAlloc(&gin_buffer, required_bytes));
	BGT_NCCL_CHECK(ncclCommWindowRegister(comm, gin_buffer, required_bytes, &gin_window, NCCL_WIN_COLL_SYMMETRIC));
	gin_capacity_bytes = required_bytes;
	gin_window_active = true;
}

template <typename T>
bool NcclContext::all_reduce_sum_gin(const T *send, T *receive, std::size_t count)
{
	if (!gin_enabled)
	{
		return false;
	}
	const std::size_t vector_bytes = count * sizeof(T);
	const std::size_t inbox_offset = vector_bytes;
	const std::size_t required_bytes = vector_bytes * static_cast<std::size_t>(dev_comm.nRanks + 1);
	ensure_gin_workspace(required_bytes);
	BGT_CUDA_CHECK(cudaMemcpy(gin_buffer, send, vector_bytes, cudaMemcpyDeviceToDevice));

	gin_exchange_allreduce_kernel<T><<<kGinCtaCount, kGinThreadsPerCta>>>(
		gin_window, 0, inbox_offset, count, dev_comm);
	check_cuda_kernel("NCCL GIN allreduce exchange kernel");

	const int grid = std::max(1, static_cast<int>((count + kBlockSize - 1) / kBlockSize));
	gin_sum_inbox_kernel<T><<<grid, kBlockSize>>>(
		static_cast<const T *>(gin_buffer), count, static_cast<int>(count), dev_comm.nRanks, receive);
	check_cuda_kernel("NCCL GIN allreduce sum kernel");
	return true;
}
#endif

#ifdef BGT_ENABLE_NCCL
void NcclContext::all_reduce_sum(const void *send, void *receive, std::size_t count, ncclDataType_t type)
{
#ifdef BGT_ENABLE_NCCL_GIN
	if (type == ncclFloat &&
		all_reduce_sum_gin(static_cast<const float *>(send), static_cast<float *>(receive), count))
	{
		return;
	}
	if (type == ncclDouble &&
		all_reduce_sum_gin(static_cast<const double *>(send), static_cast<double *>(receive), count))
	{
		return;
	}
#endif
	// All CUDA kernels in this provider launch on the default stream. Keeping NCCL on
	// the same stream preserves producer/consumer ordering without explicit events.
	BGT_NCCL_CHECK(ncclAllReduce(send, receive, count, type, ncclSum, comm, stream));
	BGT_CUDA_CHECK(cudaStreamSynchronize(stream));
}
#endif

__global__ void classify_atoms_reduce_kernel(
	const bgt::posterior_t *post_probs,
	int atoms,
	int total_states,
	bgt::host_probability_t threshold_up,
	bgt::host_probability_t threshold_lo,
	int prior_positive_mask,
	int prior_negative_mask,
	int *masks)
{
	const int atom = blockIdx.x;
	const int tid = threadIdx.x;
	if (atom >= atoms)
	{
		return;
	}

	const int bit = 1 << atom;
	const int classified = prior_positive_mask | prior_negative_mask;
	if ((classified & bit) != 0)
	{
		if (tid == 0)
		{
			if ((prior_positive_mask & bit) != 0)
			{
				atomicOr(&masks[0], bit);
			}
			if ((prior_negative_mask & bit) != 0)
			{
				atomicOr(&masks[1], bit);
			}
		}
		return;
	}

	bgt::accumulator_t local = 0.0;
	for (int state = tid; state < total_states; state += blockDim.x)
	{
		if ((state & bit) != 0)
		{
			local += static_cast<bgt::accumulator_t>(post_probs[state]);
		}
	}

	extern __shared__ bgt::accumulator_t shared_mass[];
	shared_mass[tid] = local;
	__syncthreads();

	for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
	{
		if (tid < stride)
		{
			shared_mass[tid] += shared_mass[tid + stride];
		}
		__syncthreads();
	}

	if (tid == 0)
	{
		if (shared_mass[0] < threshold_lo)
		{
			atomicOr(&masks[0], bit);
		}
		else if (shared_mass[0] > bgt::host_probability_t{1.0} - threshold_up)
		{
			atomicOr(&masks[1], bit);
		}
	}
}

__global__ void atom_mass_partition_kernel(
	const bgt::posterior_t *post_probs,
	bgt::accumulator_t *atom_mass,
	int atoms,
	int state_base,
	int local_states)
{
	const long long pair = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
	const long long total_pairs = static_cast<long long>(local_states) * atoms;
	if (pair >= total_pairs)
	{
		return;
	}

	const int atom = static_cast<int>(pair % atoms);
	const int local_state = static_cast<int>(pair / atoms);
	const int state = state_base + local_state;
	if ((state & (1 << atom)) != 0)
	{
		atomic_add_accumulator(&atom_mass[atom], static_cast<bgt::accumulator_t>(post_probs[local_state]));
	}
}

__device__ bgt::host_probability_t response_likelihood(
	int use_dilution,
	int subjects,
	int variants,
	int experiment,
	int response,
	int state,
	const bgt::host_probability_t *dilution)
{
	bgt::host_probability_t ret = 1.0;
	const int experiment_length = __popc(static_cast<unsigned int>(experiment));
	for (int variant = 0; variant < variants; variant++)
	{
		const int variant_state = state >> (variant * subjects);
		if (!use_dilution)
		{
			// Non-dilution likelihood is a noisy equality check between the observed
			// response bit and whether this state contains every tested subject for
			// this variant. The XOR form keeps the hot binary path branchless.
			const int contains_experiment = 1 ^ nonzero_to_one_u32(static_cast<unsigned int>(experiment & ~variant_state));
			const int positive_response = nonzero_to_one_u32(static_cast<unsigned int>(response & (1 << variant)));
			const int high_probability = 1 ^ (positive_response ^ contains_experiment);
			const bgt::host_probability_t lo = 1.0 - kNegativeResponse;
			ret *= lo + static_cast<bgt::host_probability_t>(high_probability) * (kNegativeResponse - lo);
		}
		else
		{
			// Dilution depends on the number of negative subjects in the tested pool;
			// this path is table driven and intentionally separate from the branchless
			// non-dilution shortcut above.
			if (experiment_length == 0)
			{
				ret *= (response & (1 << variant)) == 0 ? 1.0 : 0.0;
				continue;
			}
			const int negative_count = experiment_length - __popc(static_cast<unsigned int>(experiment & variant_state));
			const bgt::host_probability_t positive_response = dilution[(experiment_length - 1) * (subjects + 1) + negative_count];
			ret *= (response & (1 << variant)) != 0 ? positive_response : 1.0 - positive_response;
		}
	}
	return ret;
}

__global__ void posterior_update_kernel(
	const bgt::posterior_t *post_probs,
	bgt::posterior_t *next_probs,
	bgt::accumulator_t *block_sums,
	int use_dilution,
	int subjects,
	int variants,
	int total_states,
	int experiment,
	int response,
	const bgt::host_probability_t *dilution)
{
	extern __shared__ bgt::accumulator_t shared_sums[];
	const int state = blockIdx.x * blockDim.x + threadIdx.x;
	const int local = threadIdx.x;
	bgt::accumulator_t value = 0.0;
	if (state < total_states)
	{
		value = static_cast<bgt::accumulator_t>(
			post_probs[state] * response_likelihood(use_dilution, subjects, variants, experiment, response, state, dilution));
		next_probs[state] = static_cast<bgt::posterior_t>(value);
	}
	shared_sums[local] = value;
	__syncthreads();

	for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
	{
		if (local < stride)
		{
			shared_sums[local] += shared_sums[local + stride];
		}
		__syncthreads();
	}

	if (local == 0)
	{
		block_sums[blockIdx.x] = shared_sums[0];
	}
}

__global__ void posterior_update_partition_kernel(
	const bgt::posterior_t *post_probs,
	bgt::posterior_t *next_probs,
	bgt::accumulator_t *block_sums,
	int use_dilution,
	int subjects,
	int variants,
	int state_base,
	int local_states,
	int experiment,
	int response,
	const bgt::host_probability_t *dilution)
{
	extern __shared__ bgt::accumulator_t shared_sums[];
	const int local_state = blockIdx.x * blockDim.x + threadIdx.x;
	const int local = threadIdx.x;
	bgt::accumulator_t value = 0.0;
	if (local_state < local_states)
	{
		const int state = state_base + local_state;
		value = static_cast<bgt::accumulator_t>(
			post_probs[local_state] * response_likelihood(use_dilution, subjects, variants, experiment, response, state, dilution));
		next_probs[local_state] = static_cast<bgt::posterior_t>(value);
	}
	shared_sums[local] = value;
	__syncthreads();

	for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
	{
		if (local < stride)
		{
			shared_sums[local] += shared_sums[local + stride];
		}
		__syncthreads();
	}

	if (local == 0)
	{
		block_sums[blockIdx.x] = shared_sums[0];
	}
}

__global__ void sum_array_kernel(const bgt::accumulator_t *values, int count, bgt::accumulator_t *sum)
{
	const int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < count)
	{
		atomic_add_accumulator(sum, values[index]);
	}
}

__global__ void sum_array_single_block_kernel(const bgt::accumulator_t *values, int count, bgt::accumulator_t *sum)
{
	const int tid = threadIdx.x;
	bgt::accumulator_t local = 0.0;
	for (int index = tid; index < count; index += blockDim.x)
	{
		local += values[index];
	}

	extern __shared__ bgt::accumulator_t shared_sum[];
	shared_sum[tid] = local;
	__syncthreads();

	for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
	{
		if (tid < stride)
		{
			shared_sum[tid] += shared_sum[tid + stride];
		}
		__syncthreads();
	}

	if (tid == 0)
	{
		sum[0] = shared_sum[0];
	}
}

__global__ void normalize_with_device_denominator_kernel(
	bgt::posterior_t *post_probs,
	int total_states,
	const bgt::accumulator_t *denominator)
{
	const int state = blockIdx.x * blockDim.x + threadIdx.x;
	if (state < total_states)
	{
		post_probs[state] = static_cast<bgt::posterior_t>(post_probs[state] / denominator[0]);
	}
}

__global__ void classify_atoms_from_mass_kernel(
	const bgt::accumulator_t *atom_mass,
	int atoms,
	bgt::host_probability_t threshold_up,
	bgt::host_probability_t threshold_lo,
	int prior_positive_mask,
	int prior_negative_mask,
	int *masks)
{
	if (threadIdx.x != 0 || blockIdx.x != 0)
	{
		return;
	}

	int pos = prior_positive_mask;
	int neg = prior_negative_mask;
	const int classified = pos | neg;
	for (int atom = 0; atom < atoms; atom++)
	{
		const int bit = 1 << atom;
		if ((classified & bit) != 0)
		{
			continue;
		}
		if (atom_mass[atom] < threshold_lo)
		{
			pos |= bit;
		}
		else if (atom_mass[atom] > 1.0 - threshold_up)
		{
			neg |= bit;
		}
	}

	masks[0] = pos;
	masks[1] = neg;
}

__device__ bgt::host_probability_t device_prior_prob(
	int state,
	const bgt::host_probability_t *prior,
	int atoms)
{
	bgt::host_probability_t prob = 1.0;
	for (int atom = 0; atom < atoms; atom++)
	{
		prob *= (state & (1 << atom)) == 0 ? prior[atom] : (1.0 - prior[atom]);
	}
	return prob;
}

__global__ void tree_stats_kernel(
	const DeviceTreeNode *nodes,
	int root_node,
	int total_states,
	int rank,
	int world_size,
	const bgt::host_probability_t *prior,
	int atoms,
	int use_dilution,
	int subjects,
	int variants,
	const bgt::host_probability_t *dilution,
	bgt::branch_probability_t branch_threshold,
	bgt::statistic_t *block_stats,
	int *overflow)
{
	const int local_index = blockIdx.x * blockDim.x + threadIdx.x;
	const int true_state = rank + local_index * world_size;
	int node_stack[kTraversalStackCapacity];
	bgt::branch_probability_t probability_stack[kTraversalStackCapacity];
	int top = 0;

	bgt::statistic_t correct = 0.0;
	bgt::statistic_t incorrect = 0.0;
	bgt::statistic_t false_positive = 0.0;
	bgt::statistic_t false_negative = 0.0;
	bgt::statistic_t unclassified = 0.0;
	bgt::statistic_t expected_stages = 0.0;
	bgt::statistic_t expected_tests = 0.0;
	const bgt::host_probability_t prior_weight =
		true_state < total_states ? device_prior_prob(true_state, prior, atoms) : bgt::host_probability_t{0.0};
	const int atom_mask = (1 << atoms) - 1;

	if (true_state < total_states)
	{
		node_stack[top] = root_node;
		probability_stack[top] = bgt::branch_probability_t{1.0};
		top++;
	}

	// Each CUDA thread evaluates one true state against the flattened decision
	// tree. The explicit stack avoids recursive device calls; overflow is reported
	// through a device flag so the host can fail the provider cleanly.
	bool stack_overflow = false;
	while (top > 0)
	{
		top--;
		const int node_index = node_stack[top];
		const bgt::branch_probability_t branch_prob = probability_stack[top];
		const DeviceTreeNode node = nodes[node_index];
		if (node.child_count == 0)
		{
			const int classified_mask = node.positive_mask | node.negative_mask;
			const bool classified = classified_mask == atom_mask;
			if (classified && node.negative_mask == true_state)
			{
				correct += branch_prob * prior_weight;
			}
			else if (classified)
			{
				incorrect += branch_prob * prior_weight;
				const int wrong = node.negative_mask ^ true_state;
				const int total_positive = __popc(static_cast<unsigned int>(node.positive_mask));
				const int total_negative = __popc(static_cast<unsigned int>(node.negative_mask));
				if (total_positive != 0)
				{
					false_positive +=
						static_cast<bgt::statistic_t>(__popc(static_cast<unsigned int>(wrong & true_state))) /
						static_cast<bgt::statistic_t>(total_positive) * branch_prob * prior_weight;
				}
				if (total_negative != 0)
				{
					false_negative +=
						static_cast<bgt::statistic_t>(__popc(static_cast<unsigned int>(wrong & (~true_state) & atom_mask))) /
						static_cast<bgt::statistic_t>(total_negative) * branch_prob * prior_weight;
				}
			}
			else
			{
				unclassified += branch_prob * prior_weight;
			}

			expected_stages += static_cast<bgt::statistic_t>(node.stage) * branch_prob * prior_weight;
			expected_tests += static_cast<bgt::statistic_t>(node.stage) * branch_prob * prior_weight;
			continue;
		}

		for (int child_offset = 0; child_offset < node.child_count; child_offset++)
		{
			const int child_index = node.first_child + child_offset;
			const DeviceTreeNode child = nodes[child_index];
			const bgt::branch_probability_t child_prob = static_cast<bgt::branch_probability_t>(
				branch_prob * response_likelihood(
								  use_dilution, subjects, variants,
								  child.experiment, child.response, true_state, dilution));
			if (child_prob <= branch_threshold)
			{
				continue;
			}
			if (top >= kTraversalStackCapacity)
			{
				// Supported search depths should stay below this fixed stack. If a
				// future tree mode goes deeper, increase the stack or switch to a
				// dynamically managed traversal frontier before trusting results.
				atomicExch(overflow, 1);
				stack_overflow = true;
				top = 0;
				break;
			}
			node_stack[top] = child_index;
			probability_stack[top] = child_prob;
			top++;
		}
	}

	if (stack_overflow)
	{
		correct = incorrect = false_positive = false_negative = unclassified = expected_stages = expected_tests = 0.0;
	}

	extern __shared__ bgt::statistic_t shared_stats[];
	const int tid = threadIdx.x;
	shared_stats[0 * blockDim.x + tid] = correct;
	shared_stats[1 * blockDim.x + tid] = incorrect;
	shared_stats[2 * blockDim.x + tid] = false_positive;
	shared_stats[3 * blockDim.x + tid] = false_negative;
	shared_stats[4 * blockDim.x + tid] = unclassified;
	shared_stats[5 * blockDim.x + tid] = expected_stages;
	shared_stats[6 * blockDim.x + tid] = expected_tests;
	__syncthreads();

	for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
	{
		if (tid < stride)
		{
			for (int stat = 0; stat < kStatsCount; stat++)
			{
				shared_stats[stat * blockDim.x + tid] += shared_stats[stat * blockDim.x + tid + stride];
			}
		}
		__syncthreads();
	}

	if (tid == 0)
	{
		for (int stat = 0; stat < kStatsCount; stat++)
		{
			block_stats[blockIdx.x * kStatsCount + stat] = shared_stats[stat * blockDim.x];
		}
	}
}

void check_cuda_kernel(const char *label)
{
	BGT_CUDA_CHECK_LAST(label);
}

int all_subject_mask(int subjects)
{
	return (1 << subjects) - 1;
}

int partition_start(int total_items, int rank, int world_size)
{
	const int base = total_items / world_size;
	const int remainder = total_items % world_size;
	return rank * base + std::min(rank, remainder);
}

int partition_count(int total_items, int rank, int world_size)
{
	const int base = total_items / world_size;
	const int remainder = total_items % world_size;
	return base + (rank < remainder ? 1 : 0);
}

int classified_subject_mask(int classified_atoms, int subjects, int variants)
{
	int mask = 0;
	for (int subject = 0; subject < subjects; subject++)
	{
		bool classified = true;
		for (int variant = 0; variant < variants; variant++)
		{
			if ((classified_atoms & (1 << (variant * subjects + subject))) == 0)
			{
				classified = false;
				break;
			}
		}
		if (classified)
		{
			mask |= (1 << subject);
		}
	}
	return mask;
}

bool initialize_prior(
	const bgt::host_probability_t *prior,
	int atoms,
	PosteriorCudaBuffer *posterior,
	CudaWorkspace *workspace)
{
	workspace->prior.ensure(atoms);
	BGT_CUDA_CHECK(cudaMemcpy(
		workspace->prior.data, prior,
		static_cast<std::size_t>(atoms) * sizeof(bgt::host_probability_t),
		cudaMemcpyHostToDevice));

	const int total_states = 1 << atoms;
	const int grid = (total_states + kBlockSize - 1) / kBlockSize;
	set_prior_kernel<<<grid, kBlockSize>>>(posterior->data, workspace->prior.data, atoms, total_states);
	check_cuda_kernel("set prior kernel");
	return true;
}

bool initialize_prior_partition(
	const bgt::host_probability_t *prior,
	int atoms,
	int state_base,
	int local_states,
	PosteriorCudaBuffer *posterior,
	CudaWorkspace *workspace)
{
	workspace->prior.ensure(atoms);
	BGT_CUDA_CHECK(cudaMemcpy(
		workspace->prior.data, prior,
		static_cast<std::size_t>(atoms) * sizeof(bgt::host_probability_t),
		cudaMemcpyHostToDevice));

	const int grid = std::max(1, (local_states + kBlockSize - 1) / kBlockSize);
	set_prior_partition_kernel<<<grid, kBlockSize>>>(posterior->data, workspace->prior.data, atoms, state_base, local_states);
	check_cuda_kernel("set prior partition kernel");
	return true;
}

bool copy_dilution_to_device(
	bgt::host_probability_t **dilution,
	int subjects,
	HostProbabilityCudaBuffer *d_dilution)
{
	if (dilution == nullptr)
	{
		return false;
	}

	std::vector<bgt::host_probability_t> flat(static_cast<std::size_t>(subjects) * (subjects + 1), 0.0);
	for (int row = 0; row < subjects; row++)
	{
		for (int col = 0; col <= row + 1; col++)
		{
			flat[row * (subjects + 1) + col] = dilution[row][col];
		}
	}

	d_dilution->ensure(static_cast<int>(flat.size()));
	BGT_CUDA_CHECK(cudaMemcpy(
		d_dilution->data, flat.data(),
		flat.size() * sizeof(bgt::host_probability_t),
		cudaMemcpyHostToDevice));
	return true;
}

bgt::SelectorType resolve_cuda_selector(int selector, int variants)
{
	const auto selector_type = static_cast<bgt::SelectorType>(selector);
	if (selector_type == bgt::SelectorType::auto_select)
	{
		return variants == 1 ? bgt::SelectorType::op_bha : bgt::SelectorType::op_bbpa;
	}
	return selector_type;
}

int best_experiment_from_device_upset_mass(
	const AccumulatorCudaBuffer &d_upset_mass,
	int total_experiments,
	int active_mask,
	CudaWorkspace *workspace)
{
	const int candidate_grid = (total_experiments + kBlockSize - 1) / kBlockSize;
	const int reduce_blocks = std::max(1, std::min(candidate_grid, 1024));
	workspace->best_blocks.ensure(reduce_blocks);
	workspace->best_final.ensure(1);
	op_bha_best_experiment_kernel<<<reduce_blocks, kBlockSize>>>(
		d_upset_mass.data, total_experiments, active_mask, workspace->best_blocks.data);
	check_cuda_kernel("Op-BHA best experiment kernel");
	reduce_best_candidate_kernel<<<1, kBlockSize>>>(workspace->best_blocks.data, reduce_blocks, workspace->best_final.data);
	check_cuda_kernel("Op-BHA final candidate reduction kernel");

	DeviceBestCandidate best{};
	BGT_CUDA_CHECK(cudaMemcpy(&best, workspace->best_final.data, sizeof(best), cudaMemcpyDeviceToHost));
	return best.experiment;
}

int reduce_best_candidates_from_workspace(int candidate_count, const char *label, CudaWorkspace *workspace)
{
	workspace->best_final.ensure(1);
	reduce_best_candidate_kernel<<<1, kBlockSize>>>(workspace->best_blocks.data, candidate_count, workspace->best_final.data);
	check_cuda_kernel(label);

	DeviceBestCandidate best{};
	BGT_CUDA_CHECK(cudaMemcpy(&best, workspace->best_final.data, sizeof(best), cudaMemcpyDeviceToHost));
	return best.experiment;
}

int best_experiment_from_device_partition_mass(
	const AccumulatorCudaBuffer &d_mass,
	int total_experiments,
	int response_count,
	int active_mask,
	CudaWorkspace *workspace)
{
	const int candidate_grid = (total_experiments + kBlockSize - 1) / kBlockSize;
	const int reduce_blocks = std::max(1, std::min(candidate_grid, 1024));
	workspace->best_blocks.ensure(reduce_blocks);
	workspace->best_final.ensure(1);
	bbpa_best_experiment_kernel<<<reduce_blocks, kBlockSize>>>(
		d_mass.data, total_experiments, response_count, active_mask, workspace->best_blocks.data);
	check_cuda_kernel("BBPA best experiment kernel");
	reduce_best_candidate_kernel<<<1, kBlockSize>>>(workspace->best_blocks.data, reduce_blocks, workspace->best_final.data);
	check_cuda_kernel("BBPA final candidate reduction kernel");

	DeviceBestCandidate best{};
	BGT_CUDA_CHECK(cudaMemcpy(&best, workspace->best_final.data, sizeof(best), cudaMemcpyDeviceToHost));
	return best.experiment;
}

int best_experiment_bbpa_small_response(
	const PosteriorCudaBuffer &posterior,
	int subjects,
	int variants,
	int total_states,
	int total_experiments,
	int response_count,
	int active_mask,
	CudaWorkspace *workspace)
{
	workspace->best_blocks.ensure(total_experiments);
	const std::size_t shared_bytes =
		static_cast<std::size_t>(response_count) * kBlockSize * sizeof(bgt::accumulator_t);
	bbpa_best_reduce_small_response_kernel<<<total_experiments, kBlockSize, shared_bytes>>>(
		posterior.data, workspace->best_blocks.data, subjects, variants, total_states,
		total_experiments, response_count, active_mask);
	check_cuda_kernel("fused deterministic BBPA selection kernel");
	return reduce_best_candidates_from_workspace(
		total_experiments, "fused BBPA final candidate reduction kernel", workspace);
}

int best_experiment_bbpa_k2_features(
	const PosteriorCudaBuffer &posterior,
	int subjects,
	int total_experiments,
	int active_mask,
	CudaWorkspace *workspace)
{
	workspace->accumulator_a.ensure(3 * total_experiments);
	AccumulatorCudaBuffer &feature_mass = workspace->accumulator_a;
	const std::size_t shared_bytes = 3 * kBlockSize * sizeof(bgt::accumulator_t);
	bbpa_k2_feature_mass_kernel<<<total_experiments, kBlockSize, shared_bytes>>>(
		posterior.data, feature_mass.data, subjects, total_experiments);
	check_cuda_kernel("BBPA k=2 feature mass kernel");

	const int experiment_grid = (total_experiments + kBlockSize - 1) / kBlockSize;
	for (int subject = 0; subject < subjects; subject++)
	{
		const int subject_bit = 1 << subject;
		bbpa_k2_feature_zeta_step_kernel<<<experiment_grid, kBlockSize>>>(
			feature_mass.data, subject_bit, total_experiments);
		check_cuda_kernel("BBPA k=2 feature zeta kernel");
	}

	const int reduce_blocks = std::max(1, std::min(experiment_grid, 1024));
	workspace->best_blocks.ensure(reduce_blocks);
	bbpa_k2_best_experiment_kernel<<<reduce_blocks, kBlockSize>>>(
		feature_mass.data, total_experiments, active_mask, workspace->best_blocks.data);
	check_cuda_kernel("BBPA k=2 best experiment kernel");
	return reduce_best_candidates_from_workspace(
		reduce_blocks, "BBPA k=2 final candidate reduction kernel", workspace);
}

#ifdef BGT_ENABLE_NCCL
void allreduce_accumulator_device(
	NcclContext *nccl,
	const bgt::accumulator_t *send,
	bgt::accumulator_t *receive,
	int count)
{
	if (nccl == nullptr)
	{
		if (send != receive)
		{
			BGT_CUDA_CHECK(cudaMemcpy(
				receive, send,
				static_cast<std::size_t>(count) * sizeof(bgt::accumulator_t),
				cudaMemcpyDeviceToDevice));
		}
		return;
	}
	nccl->all_reduce_sum(send, receive, static_cast<std::size_t>(count), nccl_accumulator_type());
}

void allreduce_statistic_device(
	NcclContext *nccl,
	const bgt::statistic_t *send,
	bgt::statistic_t *receive,
	int count)
{
	if (nccl == nullptr)
	{
		if (send != receive)
		{
			BGT_CUDA_CHECK(cudaMemcpy(
				receive, send,
				static_cast<std::size_t>(count) * sizeof(bgt::statistic_t),
				cudaMemcpyDeviceToDevice));
		}
		return;
	}
	nccl->all_reduce_sum(send, receive, static_cast<std::size_t>(count), ncclDouble);
}
#else
void allreduce_accumulator_device(
	NcclContext *,
	const bgt::accumulator_t *send,
	bgt::accumulator_t *receive,
	int count)
{
	if (send != receive)
	{
		BGT_CUDA_CHECK(cudaMemcpy(
			receive, send,
			static_cast<std::size_t>(count) * sizeof(bgt::accumulator_t),
			cudaMemcpyDeviceToDevice));
	}
}

void allreduce_statistic_device(
	NcclContext *,
	const bgt::statistic_t *send,
	bgt::statistic_t *receive,
	int count)
{
	if (send != receive)
	{
		BGT_CUDA_CHECK(cudaMemcpy(
			receive, send,
			static_cast<std::size_t>(count) * sizeof(bgt::statistic_t),
			cudaMemcpyDeviceToDevice));
	}
}
#endif

bool select_experiment_op_bha(
	const PosteriorCudaBuffer &posterior,
	int subjects,
	int active_mask,
	CudaWorkspace *workspace,
	int *experiment)
{
	const int total_experiments = 1 << subjects;
	workspace->accumulator_a.ensure(total_experiments);
	AccumulatorCudaBuffer &d_upset_mass = workspace->accumulator_a;

	const int copy_grid = (total_experiments + kBlockSize - 1) / kBlockSize;
	op_bha_copy_posterior_kernel<<<copy_grid, kBlockSize>>>(posterior.data, d_upset_mass.data, total_experiments);
	check_cuda_kernel("Op-BHA posterior copy kernel");

	for (int subject = 0; subject < subjects; subject++)
	{
		op_bha_zeta_step_kernel<<<copy_grid, kBlockSize>>>(d_upset_mass.data, 1 << subject, total_experiments);
		check_cuda_kernel("Op-BHA zeta transform kernel");
	}

	*experiment = best_experiment_from_device_upset_mass(d_upset_mass, total_experiments, active_mask, workspace);
	return true;
}

bool select_experiment_op_bha_distributed(
	const PosteriorCudaBuffer &posterior,
	int subjects,
	int state_base,
	int local_states,
	int active_mask,
	NcclContext *nccl,
	CudaWorkspace *workspace,
	int *experiment)
{
	const int total_experiments = 1 << subjects;
	workspace->accumulator_a.ensure(total_experiments);
	AccumulatorCudaBuffer &d_upset_mass = workspace->accumulator_a;
	BGT_CUDA_CHECK(cudaMemset(
		d_upset_mass.data, 0,
		static_cast<std::size_t>(total_experiments) * sizeof(bgt::accumulator_t)));

	const int scatter_grid = std::max(1, (local_states + kBlockSize - 1) / kBlockSize);
	op_bha_scatter_partition_posterior_kernel<<<scatter_grid, kBlockSize>>>(
		posterior.data, d_upset_mass.data, state_base, local_states);
	check_cuda_kernel("distributed Op-BHA posterior scatter kernel");

	const int transform_grid = (total_experiments + kBlockSize - 1) / kBlockSize;
	for (int subject = 0; subject < subjects; subject++)
	{
		op_bha_zeta_step_kernel<<<transform_grid, kBlockSize>>>(d_upset_mass.data, 1 << subject, total_experiments);
		check_cuda_kernel("distributed Op-BHA zeta transform kernel");
	}

	workspace->accumulator_b.ensure(total_experiments);
	AccumulatorCudaBuffer &d_global_upset_mass = workspace->accumulator_b;
	allreduce_accumulator_device(nccl, d_upset_mass.data, d_global_upset_mass.data, total_experiments);

	*experiment = best_experiment_from_device_upset_mass(d_global_upset_mass, total_experiments, active_mask, workspace);
	return true;
}

bool select_experiment_bbpa(
	const PosteriorCudaBuffer &posterior,
	int subjects,
	int variants,
	int active_mask,
	CudaWorkspace *workspace,
	int *experiment)
{
	const int atoms = subjects * variants;
	const int total_states = 1 << atoms;
	const int total_experiments = 1 << subjects;
	const int response_count = 1 << variants;

	if (variants == 2 && subjects >= 8)
	{
		*experiment = best_experiment_bbpa_k2_features(
			posterior, subjects, total_experiments, active_mask, workspace);
		return true;
	}

	if (response_count <= kMaxDeterministicResponseCount)
	{
		*experiment = best_experiment_bbpa_small_response(
			posterior, subjects, variants, total_states, total_experiments,
			response_count, active_mask, workspace);
		return true;
	}

	const int partition_size = total_experiments * response_count;
	workspace->accumulator_a.ensure(partition_size);
	AccumulatorCudaBuffer &d_mass = workspace->accumulator_a;
	BGT_CUDA_CHECK(cudaMemset(
		d_mass.data, 0,
		static_cast<std::size_t>(partition_size) * sizeof(bgt::accumulator_t)));
	const long long total_pairs = static_cast<long long>(total_states) * total_experiments;
	const int grid = static_cast<int>((total_pairs + kBlockSize - 1) / kBlockSize);
	bbpa_mass_kernel<<<grid, kBlockSize>>>(posterior.data, d_mass.data, subjects, variants, total_states, total_experiments);
	check_cuda_kernel("BBPA mass kernel");
	*experiment = best_experiment_from_device_partition_mass(d_mass, total_experiments, response_count, active_mask, workspace);
	return true;
}

bool select_experiment_bbpa_distributed(
	const PosteriorCudaBuffer &posterior,
	int subjects,
	int variants,
	int state_base,
	int local_states,
	int active_mask,
	NcclContext *nccl,
	CudaWorkspace *workspace,
	int *experiment)
{
	const int total_experiments = 1 << subjects;
	const int response_count = 1 << variants;
	const int partition_size = total_experiments * response_count;
	workspace->accumulator_a.ensure(partition_size);
	AccumulatorCudaBuffer &d_mass = workspace->accumulator_a;
	BGT_CUDA_CHECK(cudaMemset(
		d_mass.data, 0,
		static_cast<std::size_t>(partition_size) * sizeof(bgt::accumulator_t)));

	const long long total_pairs = static_cast<long long>(local_states) * total_experiments;
	const int grid = std::max(1, static_cast<int>((total_pairs + kBlockSize - 1) / kBlockSize));
	bbpa_mass_partition_kernel<<<grid, kBlockSize>>>(
		posterior.data, d_mass.data, subjects, variants, state_base, local_states, total_experiments);
	check_cuda_kernel("distributed BBPA mass kernel");

	workspace->accumulator_b.ensure(partition_size);
	AccumulatorCudaBuffer &d_global_mass = workspace->accumulator_b;
	allreduce_accumulator_device(nccl, d_mass.data, d_global_mass.data, partition_size);

	*experiment = best_experiment_from_device_partition_mass(d_global_mass, total_experiments, response_count, active_mask, workspace);
	return true;
}

bool select_experiment(
	const PosteriorCudaBuffer &posterior,
	int subjects,
	int variants,
	int active_mask,
	int selector,
	CudaWorkspace *workspace,
	int *experiment)
{
	const auto selector_type = resolve_cuda_selector(selector, variants);
	if (selector_type == bgt::SelectorType::op_bha)
	{
		return variants == 1 && select_experiment_op_bha(posterior, subjects, active_mask, workspace, experiment);
	}
	return select_experiment_bbpa(posterior, subjects, variants, active_mask, workspace, experiment);
}

bool select_experiment_distributed(
	const PosteriorCudaBuffer &posterior,
	int subjects,
	int variants,
	int state_base,
	int local_states,
	int active_mask,
	int selector,
	NcclContext *nccl,
	CudaWorkspace *workspace,
	int *experiment)
{
	const auto selector_type = resolve_cuda_selector(selector, variants);
	if (selector_type == bgt::SelectorType::op_bha)
	{
		return variants == 1 &&
			   select_experiment_op_bha_distributed(posterior, subjects, state_base, local_states, active_mask, nccl, workspace, experiment);
	}
	return select_experiment_bbpa_distributed(posterior, subjects, variants, state_base, local_states, active_mask, nccl, workspace, experiment);
}

bool update_posterior(
	const PosteriorCudaBuffer &posterior,
	PosteriorCudaBuffer *next,
	int use_dilution,
	int subjects,
	int variants,
	int experiment,
	int response,
	const HostProbabilityCudaBuffer *d_dilution,
	CudaWorkspace *workspace)
{
	const int total_states = 1 << (subjects * variants);
	const int grid = (total_states + kBlockSize - 1) / kBlockSize;
	workspace->block_sums.ensure(grid);
	AccumulatorCudaBuffer &d_block_sums = workspace->block_sums;

	posterior_update_kernel<<<grid, kBlockSize, kBlockSize * sizeof(bgt::accumulator_t)>>>(
		posterior.data, next->data, d_block_sums.data, use_dilution, subjects, variants, total_states,
		experiment, response, d_dilution == nullptr ? nullptr : d_dilution->data);
	check_cuda_kernel("posterior update kernel");

	workspace->scalar_a.ensure(1);
	AccumulatorCudaBuffer &d_denominator = workspace->scalar_a;
	BGT_CUDA_CHECK(cudaMemset(d_denominator.data, 0, sizeof(bgt::accumulator_t)));
	sum_array_single_block_kernel<<<1, kBlockSize, kBlockSize * sizeof(bgt::accumulator_t)>>>(
		d_block_sums.data, grid, d_denominator.data);
	check_cuda_kernel("posterior denominator sum kernel");

	bgt::accumulator_t denominator = 0.0;
	BGT_CUDA_CHECK(cudaMemcpy(&denominator, d_denominator.data, sizeof(bgt::accumulator_t), cudaMemcpyDeviceToHost));
	if (denominator == bgt::accumulator_t{0.0})
	{
		return false;
	}

	normalize_with_device_denominator_kernel<<<grid, kBlockSize>>>(next->data, total_states, d_denominator.data);
	check_cuda_kernel("posterior normalize kernel");
	return true;
}

bool update_posterior_distributed(
	const PosteriorCudaBuffer &posterior,
	PosteriorCudaBuffer *next,
	int use_dilution,
	int subjects,
	int variants,
	int state_base,
	int local_states,
	int experiment,
	int response,
	const HostProbabilityCudaBuffer *d_dilution,
	NcclContext *nccl,
	CudaWorkspace *workspace)
{
	const int grid = std::max(1, (local_states + kBlockSize - 1) / kBlockSize);
	workspace->block_sums.ensure(grid);
	AccumulatorCudaBuffer &d_block_sums = workspace->block_sums;

	posterior_update_partition_kernel<<<grid, kBlockSize, kBlockSize * sizeof(bgt::accumulator_t)>>>(
		posterior.data, next->data, d_block_sums.data, use_dilution, subjects, variants,
		state_base, local_states, experiment, response,
		d_dilution == nullptr ? nullptr : d_dilution->data);
	check_cuda_kernel("distributed posterior update kernel");

	workspace->scalar_a.ensure(1);
	workspace->scalar_b.ensure(1);
	AccumulatorCudaBuffer &d_local_denominator = workspace->scalar_a;
	AccumulatorCudaBuffer &d_global_denominator = workspace->scalar_b;
	BGT_CUDA_CHECK(cudaMemset(d_local_denominator.data, 0, sizeof(bgt::accumulator_t)));
	sum_array_kernel<<<std::max(1, (grid + kBlockSize - 1) / kBlockSize), kBlockSize>>>(
		d_block_sums.data, grid, d_local_denominator.data);
	check_cuda_kernel("distributed posterior denominator sum kernel");
	allreduce_accumulator_device(nccl, d_local_denominator.data, d_global_denominator.data, 1);

	bgt::accumulator_t global_denominator = 0.0;
	BGT_CUDA_CHECK(cudaMemcpy(&global_denominator, d_global_denominator.data, sizeof(bgt::accumulator_t), cudaMemcpyDeviceToHost));
	if (global_denominator == bgt::accumulator_t{0.0})
	{
		return false;
	}

	normalize_with_device_denominator_kernel<<<grid, kBlockSize>>>(next->data, local_states, d_global_denominator.data);
	check_cuda_kernel("distributed posterior normalize kernel");
	return true;
}

bool classify_atoms(
	const PosteriorCudaBuffer &posterior,
	int atoms,
	bgt::host_probability_t threshold_up,
	bgt::host_probability_t threshold_lo,
	int prior_positive_mask,
	int prior_negative_mask,
	CudaWorkspace *workspace,
	int *positive_mask,
	int *negative_mask)
{
	const int total_states = 1 << atoms;
	workspace->masks.ensure(2);
	BGT_CUDA_CHECK(cudaMemset(workspace->masks.data, 0, 2 * sizeof(int)));
	classify_atoms_reduce_kernel<<<atoms, kBlockSize, kBlockSize * sizeof(bgt::accumulator_t)>>>(
		posterior.data, atoms, total_states, threshold_up, threshold_lo,
		prior_positive_mask, prior_negative_mask, workspace->masks.data);
	check_cuda_kernel("fused atom classification kernel");

	int masks[2] = {};
	BGT_CUDA_CHECK(cudaMemcpy(masks, workspace->masks.data, sizeof(masks), cudaMemcpyDeviceToHost));
	*positive_mask = masks[0];
	*negative_mask = masks[1];
	return true;
}

bool classify_atoms_distributed(
	const PosteriorCudaBuffer &posterior,
	int atoms,
	int state_base,
	int local_states,
	bgt::host_probability_t threshold_up,
	bgt::host_probability_t threshold_lo,
	int prior_positive_mask,
	int prior_negative_mask,
	NcclContext *nccl,
	CudaWorkspace *workspace,
	int *positive_mask,
	int *negative_mask)
{
	workspace->accumulator_a.ensure(atoms);
	AccumulatorCudaBuffer &d_atom_mass = workspace->accumulator_a;
	BGT_CUDA_CHECK(cudaMemset(d_atom_mass.data, 0, static_cast<std::size_t>(atoms) * sizeof(bgt::accumulator_t)));

	const long long total_pairs = static_cast<long long>(local_states) * atoms;
	const int grid = std::max(1, static_cast<int>((total_pairs + kBlockSize - 1) / kBlockSize));
	atom_mass_partition_kernel<<<grid, kBlockSize>>>(posterior.data, d_atom_mass.data, atoms, state_base, local_states);
	check_cuda_kernel("distributed atom mass kernel");

	workspace->accumulator_b.ensure(atoms);
	AccumulatorCudaBuffer &d_global_atom_mass = workspace->accumulator_b;
	allreduce_accumulator_device(nccl, d_atom_mass.data, d_global_atom_mass.data, atoms);

	workspace->masks.ensure(2);
	classify_atoms_from_mass_kernel<<<1, 1>>>(
		d_global_atom_mass.data, atoms, threshold_up, threshold_lo,
		prior_positive_mask, prior_negative_mask, workspace->masks.data);
	check_cuda_kernel("distributed atom classification kernel");

	int masks[2] = {};
	BGT_CUDA_CHECK(cudaMemcpy(masks, workspace->masks.data, sizeof(masks), cudaMemcpyDeviceToHost));
	*positive_mask = masks[0];
	*negative_mask = masks[1];
	return true;
}

std::unique_ptr<GpuTreeNode> build_tree(
	const PosteriorCudaBuffer &posterior,
	int use_dilution,
	int subjects,
	int variants,
	int search_depth,
	int selector,
	bgt::host_probability_t threshold_up,
	bgt::host_probability_t threshold_lo,
	const HostProbabilityCudaBuffer *d_dilution,
	int stage,
	int edge_experiment,
	int edge_response,
	int positive_mask,
	int negative_mask,
	CudaWorkspace *workspace,
	bool *ok)
{
	const int atoms = subjects * variants;
	auto node = std::make_unique<GpuTreeNode>(edge_experiment, edge_response, stage, positive_mask, negative_mask);
	if (node->classified(atoms) || stage >= search_depth)
	{
		return node;
	}

	const int active_mask = all_subject_mask(subjects) & ~classified_subject_mask(positive_mask | negative_mask, subjects, variants);
	int experiment = 0;
	if (!select_experiment(posterior, subjects, variants, active_mask, selector, workspace, &experiment))
	{
		*ok = false;
		return node;
	}

	const int response_count = 1 << variants;
	node->children.reserve(response_count);
	for (int response = 0; response < response_count; response++)
	{
		PosteriorCudaBuffer &child_posterior = workspace->posterior_for_depth(stage + 1, 1 << atoms);
		if (!update_posterior(
				posterior, &child_posterior, use_dilution, subjects, variants,
				experiment, response, d_dilution, workspace))
		{
			*ok = false;
			return node;
		}

		int child_positive = positive_mask;
		int child_negative = negative_mask;
		if (!classify_atoms(
				child_posterior, atoms, threshold_up, threshold_lo,
				positive_mask, negative_mask, workspace, &child_positive, &child_negative))
		{
			*ok = false;
			return node;
		}

		node->children.push_back(build_tree(
			child_posterior, use_dilution, subjects, variants, search_depth,
			selector, threshold_up, threshold_lo, d_dilution,
			stage + 1, experiment, response,
			child_positive, child_negative, workspace, ok));
		if (!*ok)
		{
			return node;
		}
	}

	return node;
}

std::unique_ptr<GpuTreeNode> build_tree_distributed(
	const PosteriorCudaBuffer &posterior,
	int use_dilution,
	int subjects,
	int variants,
	int state_base,
	int local_states,
	int search_depth,
	int selector,
	bgt::host_probability_t threshold_up,
	bgt::host_probability_t threshold_lo,
	const HostProbabilityCudaBuffer *d_dilution,
	NcclContext *nccl,
	int stage,
	int edge_experiment,
	int edge_response,
	int positive_mask,
	int negative_mask,
	CudaWorkspace *workspace,
	bool *ok)
{
	const int atoms = subjects * variants;
	auto node = std::make_unique<GpuTreeNode>(edge_experiment, edge_response, stage, positive_mask, negative_mask);
	if (node->classified(atoms) || stage >= search_depth)
	{
		return node;
	}

	const int active_mask = all_subject_mask(subjects) & ~classified_subject_mask(positive_mask | negative_mask, subjects, variants);
	int experiment = 0;
	if (!select_experiment_distributed(
			posterior, subjects, variants, state_base, local_states, active_mask, selector, nccl, workspace, &experiment))
	{
		*ok = false;
		return node;
	}

	const int response_count = 1 << variants;
	node->children.reserve(response_count);
	for (int response = 0; response < response_count; response++)
	{
		PosteriorCudaBuffer &child_posterior = workspace->posterior_for_depth(stage + 1, local_states);
		if (!update_posterior_distributed(
				posterior, &child_posterior, use_dilution, subjects, variants,
				state_base, local_states, experiment, response, d_dilution, nccl, workspace))
		{
			*ok = false;
			return node;
		}

		int child_positive = positive_mask;
		int child_negative = negative_mask;
		if (!classify_atoms_distributed(
				child_posterior, atoms, state_base, local_states, threshold_up, threshold_lo,
				positive_mask, negative_mask, nccl, workspace, &child_positive, &child_negative))
		{
			*ok = false;
			return node;
		}

		node->children.push_back(build_tree_distributed(
			child_posterior, use_dilution, subjects, variants, state_base, local_states, search_depth,
			selector, threshold_up, threshold_lo, d_dilution, nccl,
			stage + 1, experiment, response,
			child_positive, child_negative, workspace, ok));
		if (!*ok)
		{
			return node;
		}
	}

	return node;
}

int count_leaves(const GpuTreeNode *node)
{
	if (node == nullptr)
	{
		return 0;
	}
	if (node->children.empty())
	{
		return 1;
	}

	int leaves = 0;
	for (std::size_t i = 0; i < node->children.size(); i++)
	{
		leaves += count_leaves(node->children[i].get());
	}
	return leaves;
}

void flatten_tree_at(const GpuTreeNode *node, std::vector<DeviceTreeNode> *nodes, int index)
{
	(*nodes)[index] = DeviceTreeNode{
		-1,
		static_cast<int>(node->children.size()),
		node->experiment,
		node->response,
		node->stage,
		node->positive_mask,
		node->negative_mask};

	if (!node->children.empty())
	{
		const int first_child = static_cast<int>(nodes->size());
		// Children are stored contiguously so the device traversal can address a
		// node's response branches as [first_child, first_child + child_count).
		nodes->resize(nodes->size() + node->children.size());
		for (std::size_t child = 0; child < node->children.size(); child++)
		{
			flatten_tree_at(node->children[child].get(), nodes, first_child + static_cast<int>(child));
		}
		(*nodes)[index].first_child = first_child;
	}
}

int flatten_tree(const GpuTreeNode *node, std::vector<DeviceTreeNode> *nodes)
{
	nodes->resize(1);
	flatten_tree_at(node, nodes, 0);
	return 0;
}

bool copy_prior_to_device(
	const bgt::host_probability_t *prior,
	int atoms,
	HostProbabilityCudaBuffer *d_prior)
{
	d_prior->ensure(atoms);
	BGT_CUDA_CHECK(cudaMemcpy(
		d_prior->data, prior,
		static_cast<std::size_t>(atoms) * sizeof(bgt::host_probability_t),
		cudaMemcpyHostToDevice));
	return true;
}

bool evaluate_tree_stats_on_device(
	const GpuTreeNode *root,
	int use_dilution,
	int subjects,
	int variants,
	const bgt::host_probability_t *prior,
	bgt::host_probability_t **,
	const HostProbabilityCudaBuffer *d_dilution,
	bgt::branch_probability_t branch_threshold,
	int rank,
	int world_size,
	NcclContext *nccl,
	CudaWorkspace *workspace,
	bgt::TreeStats *stats)
{
	const int atoms = subjects * variants;
	const int total_states = 1 << atoms;
	std::vector<DeviceTreeNode> flat_nodes;
	flat_nodes.reserve(static_cast<std::size_t>(count_leaves(root)) * 2);
	const int root_index = flatten_tree(root, &flat_nodes);

	workspace->tree_nodes.ensure(static_cast<int>(flat_nodes.size()));
	BGT_CUDA_CHECK(cudaMemcpy(
		workspace->tree_nodes.data, flat_nodes.data(),
		flat_nodes.size() * sizeof(DeviceTreeNode),
		cudaMemcpyHostToDevice));

	if (!copy_prior_to_device(prior, atoms, &workspace->prior))
	{
		return false;
	}

	workspace->overflow.ensure(1);
	BGT_CUDA_CHECK(cudaMemset(workspace->overflow.data, 0, sizeof(int)));

	const int local_true_states = rank >= total_states ? 0 : 1 + (total_states - 1 - rank) / world_size;
	const int grid = std::max(1, (local_true_states + kBlockSize - 1) / kBlockSize);
	workspace->block_stats.ensure(grid * kStatsCount);
	tree_stats_kernel<<<grid, kBlockSize, kStatsCount * kBlockSize * sizeof(bgt::statistic_t)>>>(
		workspace->tree_nodes.data, root_index, total_states, rank, world_size,
		workspace->prior.data, atoms, use_dilution, subjects, variants,
		d_dilution == nullptr ? nullptr : d_dilution->data,
		branch_threshold, workspace->block_stats.data, workspace->overflow.data);
	check_cuda_kernel("tree statistics kernel");

	int overflow = 0;
	BGT_CUDA_CHECK(cudaMemcpy(&overflow, workspace->overflow.data, sizeof(int), cudaMemcpyDeviceToHost));
	if (overflow != 0)
	{
		return false;
	}

	bgt::statistic_t values[kStatsCount] = {};
	std::vector<bgt::statistic_t> block_values(static_cast<std::size_t>(grid) * kStatsCount);
	BGT_CUDA_CHECK(cudaMemcpy(
		block_values.data(), workspace->block_stats.data,
		block_values.size() * sizeof(bgt::statistic_t),
		cudaMemcpyDeviceToHost));
	for (int block = 0; block < grid; block++)
	{
		for (int stat = 0; stat < kStatsCount; stat++)
		{
			values[stat] += block_values[static_cast<std::size_t>(block) * kStatsCount + stat];
		}
	}
	if (nccl != nullptr)
	{
		workspace->local_stats.ensure(kStatsCount);
		workspace->global_stats.ensure(kStatsCount);
		BGT_CUDA_CHECK(cudaMemcpy(
			workspace->local_stats.data, values, sizeof(values), cudaMemcpyHostToDevice));
		allreduce_statistic_device(nccl, workspace->local_stats.data, workspace->global_stats.data, kStatsCount);
		BGT_CUDA_CHECK(cudaMemcpy(values, workspace->global_stats.data, sizeof(values), cudaMemcpyDeviceToHost));
	}

	stats->total_leaves = count_leaves(root);
	stats->correct_probability = values[0];
	stats->incorrect_probability = values[1];
	stats->false_positive_probability = values[2];
	stats->false_negative_probability = values[3];
	stats->unclassified_probability = values[4];
	stats->expected_stages = values[5];
	stats->expected_tests = values[6];
	return true;
}

} // namespace

int bgt_cuda_provider_available()
{
	return bgt::detail::cuda_has_device() ? 1 : 0;
}

int bgt_cuda_provider_run(
	int use_dilution,
	int subjects,
	int variants,
	const bgt::host_probability_t *prior,
	int search_depth,
	bgt::host_probability_t threshold_up,
	bgt::host_probability_t threshold_lo,
	bgt::branch_probability_t branch_threshold,
	int selector,
	bgt::host_probability_t **dilution,
	bgt::TreeStats *stats)
{
	*stats = bgt::TreeStats{};
	const int atoms = subjects * variants;
	if (use_dilution && dilution == nullptr)
	{
		return 1;
	}
	if (subjects <= 0 || variants <= 0 || atoms <= 0 || atoms > bgt::kStateBits ||
		atoms >= static_cast<int>(sizeof(int) * 8))
	{
		return 2;
	}
	if (atoms > 30 || subjects > 20 || variants > 10)
	{
		return 2;
	}
	if (resolve_cuda_selector(selector, variants) == bgt::SelectorType::op_bha && variants != 1)
	{
		return 2;
	}
	if (use_dilution && subjects <= 0)
	{
		return 2;
	}
	if (!bgt_cuda_provider_available())
	{
		return 2;
	}
	BGT_CUDA_CHECK(cudaSetDevice(0));

	CudaWorkspace workspace;
	PosteriorCudaBuffer &posterior = workspace.posterior_for_depth(0, 1 << atoms);
	if (!initialize_prior(prior, atoms, &posterior, &workspace))
	{
		return 4;
	}

	HostProbabilityCudaBuffer *d_dilution = nullptr;
	if (use_dilution)
	{
		if (!copy_dilution_to_device(dilution, subjects, &workspace.dilution))
		{
			return 4;
		}
		d_dilution = &workspace.dilution;
	}

	bool ok = true;
	std::unique_ptr<GpuTreeNode> root = build_tree(
		posterior, use_dilution, subjects, variants, search_depth, selector, threshold_up, threshold_lo,
		d_dilution, 0, 0, 0, 0, 0, &workspace, &ok);
	if (!ok)
	{
		return 4;
	}

	if (!evaluate_tree_stats_on_device(
			root.get(), use_dilution, subjects, variants, prior, dilution, d_dilution,
			branch_threshold, 0, 1, nullptr, &workspace, stats))
	{
		return 4;
	}

	return 0;
}

int bgt_cuda_provider_benchmark_select(
	int subjects,
	int variants,
	const bgt::host_probability_t *prior,
	int selector,
	int iterations,
	int warmup,
	bgt::state_t *candidate,
	double *init_seconds,
	double *mean_seconds,
	double *min_seconds,
	double *max_seconds,
	double *stddev_seconds)
{
	using Clock = std::chrono::steady_clock;
	if (candidate == nullptr || init_seconds == nullptr || mean_seconds == nullptr ||
		min_seconds == nullptr || max_seconds == nullptr || stddev_seconds == nullptr)
	{
		return 1;
	}
	*candidate = bgt::state_t{0};
	*init_seconds = 0.0;
	*mean_seconds = 0.0;
	*min_seconds = 0.0;
	*max_seconds = 0.0;
	*stddev_seconds = 0.0;

	const int atoms = subjects * variants;
	if (subjects <= 0 || variants <= 0 || iterations <= 0 || warmup < 0 || prior == nullptr ||
		atoms <= 0 || atoms > bgt::kStateBits || atoms >= static_cast<int>(sizeof(int) * 8))
	{
		return 2;
	}
	if (atoms > 30 || variants > 10)
	{
		return 2;
	}
	if (resolve_cuda_selector(selector, variants) == bgt::SelectorType::op_bha && variants != 1)
	{
		return 2;
	}
	if (!bgt_cuda_provider_available())
	{
		return 2;
	}
	BGT_CUDA_CHECK(cudaSetDevice(0));

	const auto init_start = Clock::now();
	CudaWorkspace workspace;
	PosteriorCudaBuffer &posterior = workspace.posterior_for_depth(0, 1 << atoms);
	if (!initialize_prior(prior, atoms, &posterior, &workspace))
	{
		return 4;
	}
	BGT_CUDA_CHECK(cudaDeviceSynchronize());
	const auto init_stop = Clock::now();
	*init_seconds = std::chrono::duration<double>(init_stop - init_start).count();

	const int active_mask = all_subject_mask(subjects);
	int selected = 0;
	for (int i = 0; i < warmup; i++)
	{
		if (!select_experiment(posterior, subjects, variants, active_mask, selector, &workspace, &selected))
		{
			return 4;
		}
		BGT_CUDA_CHECK(cudaDeviceSynchronize());
	}

	std::vector<double> timings;
	timings.reserve(static_cast<std::size_t>(iterations));
	for (int i = 0; i < iterations; i++)
	{
		const auto start = Clock::now();
		if (!select_experiment(posterior, subjects, variants, active_mask, selector, &workspace, &selected))
		{
			return 4;
		}
		BGT_CUDA_CHECK(cudaDeviceSynchronize());
		const auto stop = Clock::now();
		timings.push_back(std::chrono::duration<double>(stop - start).count());
	}

	double sum = 0.0;
	*min_seconds = std::numeric_limits<double>::max();
	*max_seconds = 0.0;
	for (const double value : timings)
	{
		sum += value;
		*min_seconds = std::min(*min_seconds, value);
		*max_seconds = std::max(*max_seconds, value);
	}
	*mean_seconds = sum / static_cast<double>(timings.size());
	double variance = 0.0;
	for (const double value : timings)
	{
		const double diff = value - *mean_seconds;
		variance += diff * diff;
	}
	*stddev_seconds = std::sqrt(variance / static_cast<double>(timings.size()));
	*candidate = static_cast<bgt::state_t>(selected);
	return 0;
}

int bgt_cuda_provider_run_distributed(
	int use_dilution,
	int subjects,
	int variants,
	const bgt::host_probability_t *prior,
	int search_depth,
	bgt::host_probability_t threshold_up,
	bgt::host_probability_t threshold_lo,
	bgt::branch_probability_t branch_threshold,
	int selector,
	int enable_nccl_gin,
	bgt::host_probability_t **dilution,
	bgt::TreeStats *stats)
{
	*stats = bgt::TreeStats{};
#ifndef BGT_ENABLE_NCCL
	(void)enable_nccl_gin;
#endif
	int mpi_initialized = 0;
	BGT_MPI_CHECK(MPI_Initialized(&mpi_initialized));
	if (!mpi_initialized)
	{
		return 3;
	}

	int rank = 0;
	int world_size = 1;
	BGT_MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
	BGT_MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &world_size));

	const int atoms = subjects * variants;
	if (use_dilution && dilution == nullptr)
	{
		return 1;
	}
	if (subjects <= 0 || variants <= 0 || atoms <= 0 || atoms > bgt::kStateBits ||
		atoms >= static_cast<int>(sizeof(int) * 8))
	{
		return 2;
	}
	if (atoms > 30 || subjects > 20 || variants > 10)
	{
		return 2;
	}
	if (resolve_cuda_selector(selector, variants) == bgt::SelectorType::op_bha && variants != 1)
	{
		return 2;
	}
	if (!bgt_cuda_provider_available())
	{
		return 2;
	}
	int device_count = 0;
	BGT_CUDA_CHECK(cudaGetDeviceCount(&device_count));
	if (device_count <= 0)
	{
		return 2;
	}
	if (world_size > device_count)
	{
		BGT_LOG_WARN(
			bgt::LogSubsystem::cuda,
			"CUDA parallel simulation requires at least one distinct GPU per MPI rank for NCCL; world_size=%d, visible_gpus=%d",
			world_size, device_count);
		return 2;
	}
	BGT_CUDA_CHECK(cudaSetDevice(rank % device_count));
	std::unique_ptr<NcclContext> nccl;
	if (world_size > 1)
	{
#ifdef BGT_ENABLE_NCCL
		nccl = std::make_unique<NcclContext>(rank, world_size, enable_nccl_gin != 0);
#else
		(void)enable_nccl_gin;
		return 2;
#endif
	}

	const int total_states = 1 << atoms;
	const int state_base = partition_start(total_states, rank, world_size);
	const int local_states = partition_count(total_states, rank, world_size);

	CudaWorkspace workspace;
	PosteriorCudaBuffer &posterior = workspace.posterior_for_depth(0, local_states);
	if (!initialize_prior_partition(prior, atoms, state_base, local_states, &posterior, &workspace))
	{
		return 4;
	}

	HostProbabilityCudaBuffer *d_dilution = nullptr;
	if (use_dilution)
	{
		if (!copy_dilution_to_device(dilution, subjects, &workspace.dilution))
		{
			return 4;
		}
		d_dilution = &workspace.dilution;
	}

	bool ok = true;
	std::unique_ptr<GpuTreeNode> root = build_tree_distributed(
		posterior, use_dilution, subjects, variants, state_base, local_states,
		search_depth, selector, threshold_up, threshold_lo, d_dilution, nccl.get(),
		0, 0, 0, 0, 0, &workspace, &ok);
	if (!ok)
	{
		return 4;
	}

	if (!evaluate_tree_stats_on_device(
			root.get(), use_dilution, subjects, variants, prior, dilution, d_dilution,
			branch_threshold, rank, world_size, nccl.get(), &workspace, stats))
	{
		return 4;
	}

	return 0;
}
