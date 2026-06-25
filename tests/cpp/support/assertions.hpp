#pragma once

#include "bgt/bgt.hpp"

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

namespace bgt::test
{

inline constexpr double kTolerance =
#if BGT_POSTERIOR_BITS == 32 || BGT_ACCUMULATOR_BITS == 32
	1e-5;
#else
	1e-12;
#endif

[[noreturn]] inline void fail(const std::string &label, const std::string &message)
{
	std::cerr << label << ": " << message << std::endl;
	std::exit(1);
}

inline void expect_near(double actual, double expected, const std::string &label, double tolerance = kTolerance)
{
	if (std::fabs(actual - expected) > tolerance)
		fail(label, "expected " + std::to_string(expected) + ", got " + std::to_string(actual));
}

template <typename T>
void expect_equal(const T &actual, const T &expected, const std::string &label)
{
	if (actual != expected)
		fail(label, "values differ");
}

inline void expect_true(bool value, const std::string &label)
{
	if (!value)
		fail(label, "expected true");
}

inline void expect_same_stats(const TreeStats &actual, const TreeStats &expected, const std::string &label)
{
	expect_equal(actual.total_leaves, expected.total_leaves, label + " total leaves");
	expect_near(actual.correct_probability, expected.correct_probability, label + " correct");
	expect_near(actual.incorrect_probability, expected.incorrect_probability, label + " incorrect");
	expect_near(actual.false_positive_probability, expected.false_positive_probability, label + " false positive");
	expect_near(actual.false_negative_probability, expected.false_negative_probability, label + " false negative");
	expect_near(actual.unclassified_probability, expected.unclassified_probability, label + " unclassified");
	expect_near(actual.expected_stages, expected.expected_stages, label + " expected stages");
	expect_near(actual.expected_tests, expected.expected_tests, label + " expected tests");
}

inline SimulationOptions cpu_options()
{
	SimulationOptions options;
	options.search_depth = 1;
	options.threshold_up = 0.01;
	options.threshold_lo = 0.01;
	options.branch_threshold = 0.0;
	options.provider = Provider::cpu;
	return options;
}

inline SimulationConfig binary_single_subject_config()
{
	SimulationConfig config;
	config.lattice_type = LatticeType::replicated_non_dilution;
	config.subjects = 1;
	config.variants = 1;
	config.prior = {0.2};
	config.options = cpu_options();
	return config;
}

inline TreeStats binary_single_subject_expected_stats()
{
	TreeStats stats;
	stats.total_leaves = 2;
	stats.correct_probability = 0.792;
	stats.incorrect_probability = 0.002;
	stats.false_positive_probability = 0.0;
	stats.false_negative_probability = 0.002;
	stats.unclassified_probability = 0.206;
	stats.expected_stages = 1.0;
	stats.expected_tests = 1.0;
	return stats;
}

inline std::filesystem::path temporary_path(const std::string &name)
{
	return std::filesystem::temp_directory_path() / name;
}

} // namespace bgt::test
