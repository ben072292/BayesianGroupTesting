#pragma once

#include "bgt/compile_options.hpp"
#include "bgt/config.hpp"

#include <cstdint>
#include <functional>
#include <string>

namespace bgt
{

template <int Bits>
struct state_encoding;

template <>
struct state_encoding<8>
{
	using type = std::uint8_t;
};

template <>
struct state_encoding<16>
{
	using type = std::uint16_t;
};

template <>
struct state_encoding<32>
{
	using type = std::uint32_t;
};

template <>
struct state_encoding<64>
{
	using type = std::uint64_t;
};

using state_t = typename state_encoding<BGT_STATE_BITS>::type;

template <int Bits>
struct posterior_encoding;

template <>
struct posterior_encoding<32>
{
	using type = float;
};

template <>
struct posterior_encoding<64>
{
	using type = double;
};

using posterior_t = typename posterior_encoding<BGT_POSTERIOR_BITS>::type;

template <int Bits>
struct accumulator_encoding;

template <>
struct accumulator_encoding<32>
{
	using type = float;
};

template <>
struct accumulator_encoding<64>
{
	using type = double;
};

using accumulator_t = typename accumulator_encoding<BGT_ACCUMULATOR_BITS>::type;
using branch_probability_t = accumulator_t;
using score_t = accumulator_t;
using host_probability_t = double;
using statistic_t = double;
using seconds_t = double;

enum class LatticeType
{
	distributed_non_dilution,
	distributed_dilution,
	replicated_non_dilution,
	replicated_dilution
};

enum class SelectorType
{
	auto_select,
	op_bha,
	op_bbpa
};

enum class Provider
{
	auto_select,
	cpu,
	cuda
};

enum class DilutionMode
{
	non_dilution,
	dilution
};

struct TreeStats
{
	int total_leaves = 0;
	statistic_t correct_probability = 0.0;
	statistic_t incorrect_probability = 0.0;
	statistic_t false_positive_probability = 0.0;
	statistic_t false_negative_probability = 0.0;
	statistic_t unclassified_probability = 0.0;
	statistic_t expected_stages = 0.0;
	statistic_t expected_tests = 0.0;
};

struct SimulationOptions
{
	int search_depth = 1;
	host_probability_t threshold_up = 0.01;
	host_probability_t threshold_lo = 0.01;
	branch_probability_t branch_threshold = 0.0;
	SelectorType selector = SelectorType::auto_select;
	Provider provider = Provider::auto_select;
	CompileOptions compile_options;
};

struct KernelSpec
{
	Provider provider = Provider::auto_select;
	int subjects = 0;
	int variants = 1;
	int state_bits = BGT_STATE_BITS;
	DilutionMode dilution = DilutionMode::non_dilution;
	SelectorType selector = SelectorType::auto_select;
	Precision precision = compiled_posterior_precision;
	Precision accumulator_precision = compiled_accumulator_precision;
	std::string cuda_arch;
	CompileOptions compile_options;

	bool operator==(const KernelSpec &other) const
	{
		return provider == other.provider &&
			   subjects == other.subjects &&
			   variants == other.variants &&
			   state_bits == other.state_bits &&
			   dilution == other.dilution &&
			   selector == other.selector &&
			   precision == other.precision &&
			   accumulator_precision == other.accumulator_precision &&
			   cuda_arch == other.cuda_arch &&
			   compile_options == other.compile_options;
	}
};

struct KernelSpecHash
{
	std::size_t operator()(const KernelSpec &spec) const
	{
		std::size_t seed = 0;
		auto combine = [&seed](auto value) {
			std::hash<decltype(value)> hasher;
			seed ^= hasher(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
		};
		combine(static_cast<int>(spec.provider));
		combine(spec.subjects);
		combine(spec.variants);
		combine(spec.state_bits);
		combine(static_cast<int>(spec.dilution));
		combine(static_cast<int>(spec.selector));
		combine(static_cast<int>(spec.precision));
		combine(static_cast<int>(spec.accumulator_precision));
		combine(std::hash<std::string>{}(spec.cuda_arch));
		combine(CompileOptionsHash{}(spec.compile_options));
		return seed;
	}
};

struct JitCacheInfo
{
	std::size_t memory_entries = 0;
	std::size_t disk_entries = 0;
	std::string disk_cache_directory;
	bool jit_enabled = false;
};

} // namespace bgt
