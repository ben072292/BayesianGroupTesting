#ifndef BGT_SIMD_H_
#define BGT_SIMD_H_

#include <cstddef>
#include <cstdlib>
#include <limits>
#include <new>

#include "bgt/types.hpp"
#include "bgt/detail/state_encoding.hpp"

#if defined(ENABLE_SIMD) && (defined(__AVX2__) || defined(__AVX512F__))
#include <immintrin.h>
#endif

namespace bgt
{
namespace simd
{

#ifdef ENABLE_SIMD

#if defined(__AVX512F__)
static const int kAlignment = 64;
static const int kStateLanes = 64 / sizeof(bgt_state_t);
static const char *kArchitectureName = "AVX-512";
#if BGT_STATE_BITS == 32
typedef __m512i StateVector;
#else
typedef bgt_state_t StateVector __attribute__((vector_size(64)));
#endif

#elif defined(__AVX2__)
static const int kAlignment = 32;
static const int kStateLanes = 32 / sizeof(bgt_state_t);
static const char *kArchitectureName = "AVX2";
#if BGT_STATE_BITS == 32
typedef __m256i StateVector;
#else
typedef bgt_state_t StateVector __attribute__((vector_size(32)));
#endif

#elif defined(__SSE2__) || defined(__ARM_NEON) || defined(__ARM_NEON__)
static const int kAlignment = 16;
static const int kStateLanes = 16 / sizeof(bgt_state_t);
#if defined(__SSE2__)
static const char *kArchitectureName = "SSE2";
#else
static const char *kArchitectureName = "ARM NEON";
#endif
typedef bgt_state_t StateVector __attribute__((vector_size(16)));

#else
#error "Unsupported SIMD instruction set."
#endif

inline void *aligned_alloc_bytes(std::size_t byte_count)
{
	void *ptr = nullptr;
	if (posix_memalign(&ptr, kAlignment, byte_count))
	{
		throw std::bad_alloc();
	}
	return ptr;
}

inline const char *architecture_name()
{
	return kArchitectureName;
}

#if BGT_STATE_BITS == 32 && (defined(__AVX2__) || defined(__AVX512F__))

#if defined(__AVX512F__)
inline StateVector zero_state() { return _mm512_setzero_si512(); }
inline StateVector splat_state(bgt_state_t value) { return _mm512_set1_epi32(static_cast<int>(value)); }
inline StateVector lane_offsets() { return _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0); }
inline StateVector bit_and(StateVector lhs, StateVector rhs) { return _mm512_and_si512(lhs, rhs); }
inline StateVector bit_or(StateVector lhs, StateVector rhs) { return _mm512_or_si512(lhs, rhs); }
inline StateVector bit_xor(StateVector lhs, StateVector rhs) { return _mm512_xor_si512(lhs, rhs); }
inline StateVector add(StateVector lhs, StateVector rhs) { return _mm512_add_epi32(lhs, rhs); }
inline StateVector all_ones_state() { return _mm512_set1_epi32(-1); }
inline StateVector bit_not(StateVector value) { return bit_xor(value, all_ones_state()); }
inline StateVector not_equal_zero(StateVector value)
{
	const __mmask16 equal_zero = _mm512_cmpeq_epi32_mask(value, zero_state());
	return _mm512_mask_mov_epi32(all_ones_state(), equal_zero, zero_state());
}
inline bgt_state_t lane(StateVector value, int index)
{
	alignas(64) bgt_state_t lanes[kStateLanes];
	_mm512_store_si512(reinterpret_cast<__m512i *>(lanes), value);
	return lanes[index];
}
#elif defined(__AVX2__)
inline StateVector zero_state() { return _mm256_setzero_si256(); }
inline StateVector splat_state(bgt_state_t value) { return _mm256_set1_epi32(static_cast<int>(value)); }
inline StateVector lane_offsets() { return _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0); }
inline StateVector bit_and(StateVector lhs, StateVector rhs) { return _mm256_and_si256(lhs, rhs); }
inline StateVector bit_or(StateVector lhs, StateVector rhs) { return _mm256_or_si256(lhs, rhs); }
inline StateVector bit_xor(StateVector lhs, StateVector rhs) { return _mm256_xor_si256(lhs, rhs); }
inline StateVector add(StateVector lhs, StateVector rhs) { return _mm256_add_epi32(lhs, rhs); }
inline StateVector all_ones_state() { return _mm256_set1_epi32(-1); }
inline StateVector bit_not(StateVector value) { return bit_xor(value, all_ones_state()); }
inline StateVector not_equal_zero(StateVector value)
{
	const StateVector equal_zero = _mm256_cmpeq_epi32(value, zero_state());
	return _mm256_andnot_si256(equal_zero, all_ones_state());
}
inline bgt_state_t lane(StateVector value, int index)
{
	alignas(32) bgt_state_t lanes[kStateLanes];
	_mm256_store_si256(reinterpret_cast<__m256i *>(lanes), value);
	return lanes[index];
}
#endif

#else

inline StateVector zero_state()
{
	StateVector result = {};
	return result;
}

inline StateVector splat_state(bgt_state_t value)
{
	StateVector result = {};
	for (int i = 0; i < kStateLanes; i++)
	{
		result[i] = value;
	}
	return result;
}

inline StateVector lane_offsets()
{
	StateVector result = {};
	for (int i = 0; i < kStateLanes; i++)
	{
		result[i] = static_cast<bgt_state_t>(i);
	}
	return result;
}

inline StateVector bit_and(StateVector lhs, StateVector rhs) { return lhs & rhs; }
inline StateVector bit_or(StateVector lhs, StateVector rhs) { return lhs | rhs; }
inline StateVector add(StateVector lhs, StateVector rhs) { return lhs + rhs; }
inline StateVector bit_not(StateVector value) { return ~value; }
inline StateVector not_equal_zero(StateVector value)
{
	StateVector result = {};
	for (int i = 0; i < kStateLanes; i++)
	{
		result[i] = value[i] == 0 ? 0 : std::numeric_limits<bgt_state_t>::max();
	}
	return result;
}
inline bgt_state_t lane(StateVector value, int index) { return value[index]; }

#endif

inline StateVector add_scalar(StateVector value, int scalar)
{
	return add(value, splat_state(static_cast<bgt_state_t>(scalar)));
}

inline void accumulate_partition(bgt::accumulator_t *values, StateVector experiments, int response_count, StateVector partition_ids, bgt::posterior_t value)
{
	for (int i = 0; i < kStateLanes; i++)
	{
		values[static_cast<int>(lane(experiments, i)) * response_count + static_cast<int>(lane(partition_ids, i))] += value;
	}
}

#endif

} // namespace simd
} // namespace bgt

#endif
