#pragma once

#include "bgt/config.hpp"

#include <functional>
#include <string>
#include <vector>

namespace bgt
{

enum class OptimizationLevel
{
	debug,
	release,
	aggressive
};

enum class Precision
{
	float32,
	float64
};

enum class ProfileBackend
{
	none,
	caliper
};

inline constexpr Precision compiled_posterior_precision =
#if BGT_POSTERIOR_BITS == 32
	Precision::float32;
#elif BGT_POSTERIOR_BITS == 64
	Precision::float64;
#else
#error "Unsupported BGT_POSTERIOR_BITS value."
#endif

inline constexpr Precision compiled_accumulator_precision =
#if BGT_ACCUMULATOR_BITS == 32
	Precision::float32;
#elif BGT_ACCUMULATOR_BITS == 64
	Precision::float64;
#else
#error "Unsupported BGT_ACCUMULATOR_BITS value."
#endif

inline constexpr ProfileBackend compiled_profile_backend =
#if BGT_PROFILE_BACKEND_CODE == 0
	ProfileBackend::none;
#elif BGT_PROFILE_BACKEND_CODE == 1
	ProfileBackend::caliper;
#else
#error "Unsupported BGT_PROFILE_BACKEND_CODE value."
#endif

struct CompileOptions
{
	bool enable_simd = simd_compiled;
	bool enable_openmp = openmp_compiled;
	bool enable_cuda = cuda_compiled;
	bool enable_nccl = nccl_compiled;
	bool enable_nccl_gin = false;
	bool fast_math = fast_math_compiled;
	bool native_cpu = native_cpu_compiled;
	Precision posterior_precision = compiled_posterior_precision;
	Precision accumulator_precision = compiled_accumulator_precision;
	OptimizationLevel optimization = static_cast<OptimizationLevel>(optimization_level_code);
	ProfileBackend profile_backend = compiled_profile_backend;
	std::vector<std::string> extra_cxx_flags;
	std::vector<std::string> extra_cuda_flags;
	std::vector<std::string> definitions;

	bool operator==(const CompileOptions &other) const
	{
		return enable_simd == other.enable_simd &&
			   enable_openmp == other.enable_openmp &&
			   enable_cuda == other.enable_cuda &&
			   enable_nccl == other.enable_nccl &&
			   enable_nccl_gin == other.enable_nccl_gin &&
			   fast_math == other.fast_math &&
			   native_cpu == other.native_cpu &&
			   posterior_precision == other.posterior_precision &&
			   accumulator_precision == other.accumulator_precision &&
			   optimization == other.optimization &&
			   profile_backend == other.profile_backend &&
			   extra_cxx_flags == other.extra_cxx_flags &&
			   extra_cuda_flags == other.extra_cuda_flags &&
			   definitions == other.definitions;
	}
};

struct CompileOptionsHash
{
	std::size_t operator()(const CompileOptions &options) const
	{
		std::size_t seed = 0;
		auto combine = [&seed](auto value) {
			std::hash<decltype(value)> hasher;
			seed ^= hasher(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
		};
		auto combine_string_vector = [&combine](const std::vector<std::string> &values) {
			for (const auto &value : values)
				combine(std::hash<std::string>{}(value));
			combine(values.size());
		};
		combine(options.enable_simd);
		combine(options.enable_openmp);
		combine(options.enable_cuda);
		combine(options.enable_nccl);
		combine(options.enable_nccl_gin);
		combine(options.fast_math);
		combine(options.native_cpu);
		combine(static_cast<int>(options.posterior_precision));
		combine(static_cast<int>(options.accumulator_precision));
		combine(static_cast<int>(options.optimization));
		combine(static_cast<int>(options.profile_backend));
		combine_string_vector(options.extra_cxx_flags);
		combine_string_vector(options.extra_cuda_flags);
		combine_string_vector(options.definitions);
		return seed;
	}
};

} // namespace bgt
