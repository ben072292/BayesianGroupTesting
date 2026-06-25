#include "bgt/bgt.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace
{
void expect_near(double actual, double expected, const char *label)
{
	const double tolerance = (bgt::posterior_bits == 32 || bgt::accumulator_bits == 32) ? 1e-5 : 1e-12;
	if (std::fabs(actual - expected) > tolerance)
	{
		std::cerr << label << ": expected " << expected << ", got " << actual << std::endl;
		std::exit(1);
	}
}

template <typename T>
void expect_equal(T actual, T expected, const char *label)
{
	if (actual != expected)
	{
		std::cerr << label << ": expected " << expected << ", got " << actual << std::endl;
		std::exit(1);
	}
}

void expect_same_stats(const bgt::TreeStats &actual, const bgt::TreeStats &expected, const char *label)
{
	expect_equal(actual.total_leaves, expected.total_leaves, label);
	expect_near(actual.correct_probability, expected.correct_probability, label);
	expect_near(actual.incorrect_probability, expected.incorrect_probability, label);
	expect_near(actual.false_positive_probability, expected.false_positive_probability, label);
	expect_near(actual.false_negative_probability, expected.false_negative_probability, label);
	expect_near(actual.unclassified_probability, expected.unclassified_probability, label);
	expect_near(actual.expected_stages, expected.expected_stages, label);
	expect_near(actual.expected_tests, expected.expected_tests, label);
}

void expect_provider_parity(bgt::SimulationConfig config, const char *label)
{
	config.options.provider = bgt::Provider::cpu;
	const bgt::TreeStats cpu_stats = bgt::run_simulation(config).stats;
	config.options.provider = bgt::Provider::cuda;
	const bgt::TreeStats gpu_stats = bgt::run_simulation(config).stats;
	expect_same_stats(gpu_stats, cpu_stats, label);
	config.options.provider = bgt::Provider::auto_select;
	const bgt::TreeStats auto_stats = bgt::run_simulation(config).stats;
	expect_same_stats(auto_stats, gpu_stats, label);
}

bgt::SimulationConfig make_config(bgt::LatticeType type, int subjects, int variants,
								  std::vector<double> prior, const bgt::SimulationOptions &options)
{
	bgt::SimulationConfig config;
	config.lattice_type = type;
	config.subjects = subjects;
	config.variants = variants;
	config.prior = std::move(prior);
	config.options = options;
	return config;
}
}

int main()
{
	if (!bgt::cuda_available())
	{
		std::cout << "No CUDA device available; skipping CUDA tree parity test." << std::endl;
		return 0;
	}

	std::vector<double> prior{0.2, 0.4};
	bgt::SimulationOptions options;
	options.search_depth = 1;
	options.threshold_up = 0.01;
	options.threshold_lo = 0.01;
	options.branch_threshold = 0.0;

	options.provider = bgt::Provider::cpu;
	bgt::TreeStats cpu_stats = bgt::run_simulation(
		make_config(bgt::LatticeType::replicated_non_dilution, 2, 1, prior, options)).stats;
	options.provider = bgt::Provider::cuda;
	bgt::TreeStats gpu_stats = bgt::run_simulation(
		make_config(bgt::LatticeType::replicated_non_dilution, 2, 1, prior, options)).stats;
	expect_same_stats(gpu_stats, cpu_stats, "GPU non-dilution tree stats");

	options.provider = bgt::Provider::auto_select;
	bgt::TreeStats default_stats = bgt::run_simulation(
		make_config(bgt::LatticeType::replicated_non_dilution, 2, 1, prior, options)).stats;
	expect_same_stats(default_stats, gpu_stats, "default non-dilution tree stats");

	options.provider = bgt::Provider::cpu;
	cpu_stats = bgt::run_simulation(
		make_config(bgt::LatticeType::replicated_dilution, 2, 1, prior, options)).stats;
	options.provider = bgt::Provider::cuda;
	gpu_stats = bgt::run_simulation(
		make_config(bgt::LatticeType::replicated_dilution, 2, 1, prior, options)).stats;
	expect_same_stats(gpu_stats, cpu_stats, "GPU dilution tree stats");

	options.provider = bgt::Provider::auto_select;
	default_stats = bgt::run_simulation(
		make_config(bgt::LatticeType::replicated_dilution, 2, 1, prior, options)).stats;
	expect_same_stats(default_stats, gpu_stats, "default dilution tree stats");

	std::vector<double> multinomial_prior{0.2, 0.4};
	options.provider = bgt::Provider::cpu;
	cpu_stats = bgt::run_simulation(
		make_config(bgt::LatticeType::replicated_non_dilution, 1, 2, multinomial_prior, options)).stats;
	options.provider = bgt::Provider::cuda;
	gpu_stats = bgt::run_simulation(
		make_config(bgt::LatticeType::replicated_non_dilution, 1, 2, multinomial_prior, options)).stats;
	expect_same_stats(gpu_stats, cpu_stats, "GPU multinomial tree stats");

	options.provider = bgt::Provider::auto_select;
	default_stats = bgt::run_simulation(
		make_config(bgt::LatticeType::replicated_non_dilution, 1, 2, multinomial_prior, options)).stats;
	expect_same_stats(default_stats, gpu_stats, "default multinomial tree stats");

	options.provider = bgt::Provider::cpu;
	options.search_depth = 3;
	options.branch_threshold = 0.001;
	options.selector = bgt::SelectorType::op_bha;
	expect_provider_parity(
		make_config(bgt::LatticeType::replicated_non_dilution, 3, 1, {0.02, 0.05, 0.1}, options),
		"GPU deeper binary Op-BHA tree stats");

	options.selector = bgt::SelectorType::op_bbpa;
	expect_provider_parity(
		make_config(bgt::LatticeType::replicated_non_dilution, 3, 1, {0.02, 0.05, 0.1}, options),
		"GPU deeper binary Op-BBPA tree stats");

	options.selector = bgt::SelectorType::op_bha;
	expect_provider_parity(
		make_config(bgt::LatticeType::replicated_dilution, 3, 1, {0.02, 0.05, 0.1}, options),
		"GPU deeper binary dilution Op-BHA tree stats");

	options.selector = bgt::SelectorType::op_bbpa;
	expect_provider_parity(
		make_config(bgt::LatticeType::replicated_dilution, 3, 1, {0.02, 0.05, 0.1}, options),
		"GPU deeper binary dilution Op-BBPA tree stats");

	options.search_depth = 2;
	options.branch_threshold = 0.001;
	options.selector = bgt::SelectorType::op_bbpa;
	expect_provider_parity(
		make_config(bgt::LatticeType::replicated_non_dilution, 2, 2, {0.02, 0.04, 0.06, 0.08}, options),
		"GPU deeper multinomial Op-BBPA tree stats");

	return 0;
}
