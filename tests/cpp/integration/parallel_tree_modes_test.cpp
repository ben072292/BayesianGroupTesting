#include "bgt/bgt.hpp"
#include "bgt/detail/model/lattice.hpp"

#include <cmath>
#include <iostream>
#include <mpi.h>
#include <vector>

namespace
{
void expect_near(double actual, double expected, const char *label)
{
	const double tolerance = 1e-12;
	if (std::fabs(actual - expected) > tolerance)
	{
		std::cerr << label << ": expected " << expected << ", got " << actual << std::endl;
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
}

template <typename T>
void expect_equal(T actual, T expected, const char *label)
{
	if (actual != expected)
	{
		std::cerr << label << ": expected " << expected << ", got " << actual << std::endl;
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
}

bgt::SimulationConfig make_config(bgt::LatticeType type, int subjects, int variants,
								  std::vector<double> prior, bgt::SimulationMode mode,
								  const bgt::SimulationOptions &options)
{
	bgt::SimulationConfig config;
	config.lattice_type = type;
	config.subjects = subjects;
	config.variants = variants;
	config.prior = std::move(prior);
	config.options = options;
	config.mode = mode;
	config.workload_granularity = 1;
	return config;
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
}

int main(int argc, char **argv)
{
	MPI_Init(&argc, &argv);

	int rank = 0;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	bgt::SimulationOptions options;
	options.search_depth = 1;
	options.threshold_up = 0.01;
	options.threshold_lo = 0.01;
	options.branch_threshold = 0.0;

	std::vector<double> prior{0.2};
	bgt::SimulationConfig config = make_config(
		bgt::LatticeType::replicated_non_dilution, 1, 1, prior,
		bgt::SimulationMode::parallel_global_tree, options);
	bgt::SimulationResult result = bgt::run_simulation(config);
	bgt::TreeStats stats = result.stats;

	if (rank == 0)
	{
		expect_equal(stats.total_leaves, 2, "BGT parallel replicated global tree leaves");
		expect_near(stats.correct_probability, 0.792, "BGT parallel replicated global tree correct");
		expect_near(stats.incorrect_probability, 0.002, "BGT parallel replicated global tree incorrect");
		expect_near(stats.false_positive_probability, 0.0, "BGT parallel replicated global tree false positive");
		expect_near(stats.false_negative_probability, 0.002, "BGT parallel replicated global tree false negative");
		expect_near(stats.unclassified_probability, 0.206, "BGT parallel replicated global tree unclassified");
		expect_near(stats.expected_stages, 1.0, "BGT parallel replicated global tree expected stages");
		expect_near(stats.expected_tests, 1.0, "BGT parallel replicated global tree expected tests");
	}

	std::vector<double> prior_two_subjects{0.2, 0.4};
	bgt::TreeStats replicated_stats = bgt::run_simulation(
		make_config(bgt::LatticeType::replicated_non_dilution, 2, 1, prior_two_subjects,
					bgt::SimulationMode::parallel_global_tree, options)).stats;
	bgt::TreeStats distributed_stats = bgt::run_simulation(
		make_config(bgt::LatticeType::distributed_non_dilution, 2, 1, prior_two_subjects,
					bgt::SimulationMode::parallel_global_tree, options)).stats;
	if (rank == 0)
	{
		expect_same_stats(distributed_stats, replicated_stats, "BGT parallel distributed global tree stats");
	}

	replicated_stats = bgt::run_simulation(
		make_config(bgt::LatticeType::replicated_dilution, 2, 1, prior_two_subjects,
					bgt::SimulationMode::parallel_global_tree, options)).stats;
	distributed_stats = bgt::run_simulation(
		make_config(bgt::LatticeType::distributed_dilution, 2, 1, prior_two_subjects,
					bgt::SimulationMode::parallel_global_tree, options)).stats;
	if (rank == 0)
	{
		expect_same_stats(distributed_stats, replicated_stats, "BGT parallel distributed dilution global tree stats");
	}

	{
		bgt::model::lattice_mpi_initialize(DIST_NON_DILUTION, 2, 2);
		double pi0[] = {0.95, 0.5, 0.95, 0.5};
		auto lattice = bgt::model::create_lattice(DIST_NON_DILUTION, 2, 2, pi0);
		const bool converted = lattice->update_metadata_with_shrinking(0.1, 0.1);
		if (converted)
			lattice = lattice->to_local();
		expect_equal(lattice->curr_subjs(), 1, "multinomial distributed shrink subject count");
		double local_mass = 0.0;
		for (int i = 0; i < lattice->posterior_prob_count(); ++i)
			local_mass += lattice->posterior_probs()[i];
		if (lattice->parallelism() == DIST_MODEL)
			MPI_Allreduce(MPI_IN_PLACE, &local_mass, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		expect_near(local_mass, 1.0, "multinomial distributed shrink mass");
		bgt::model::lattice_mpi_finalize(DIST_NON_DILUTION);
	}

	bgt::SimulationConfig dynamic_config = make_config(
		bgt::LatticeType::replicated_non_dilution, 1, 1, prior,
		bgt::SimulationMode::parallel_dynamic_tree, options);
	bgt::SimulationConfig hybrid_config = dynamic_config;
	hybrid_config.mode = bgt::SimulationMode::parallel_hybrid_tree;
	hybrid_config.global_tree_depth = 1;
	bgt::TreeStats dynamic_stats = bgt::run_simulation(dynamic_config).stats;
	bgt::TreeStats hybrid_stats = bgt::run_simulation(hybrid_config).stats;
	if (rank == 0)
	{
		expect_equal(dynamic_stats.total_leaves, 2, "BGT parallel dynamic tree leaves");
		expect_equal(hybrid_stats.total_leaves, 2, "BGT parallel hybrid tree leaves");
		expect_near(dynamic_stats.correct_probability, 0.792, "BGT parallel dynamic tree correct");
		expect_near(hybrid_stats.correct_probability, 0.792, "BGT parallel hybrid tree correct");
	}

	bgt::SimulationConfig trimmed_config = make_config(
		bgt::LatticeType::distributed_non_dilution, 1, 1, prior,
		bgt::SimulationMode::parallel_partial_tree, options);
	trimmed_config.true_state_policy = bgt::TrueStatePolicy::trimmed;
	trimmed_config.trim_percent = 1.0;
	bgt::SimulationConfig symmetric_config = trimmed_config;
	symmetric_config.true_state_policy = bgt::TrueStatePolicy::symmetric;
	bgt::SimulationResult trimmed_result = bgt::run_simulation(trimmed_config);
	bgt::SimulationResult symmetric_result = bgt::run_simulation(symmetric_config);
	if (rank == 0)
	{
		expect_equal(trimmed_result.evaluated_states, 2, "BGT parallel partial trimmed evaluated states");
		expect_equal(symmetric_result.evaluated_states, 2, "BGT parallel partial symmetric evaluated states");
		expect_equal(trimmed_result.stats.total_leaves > 0, true, "BGT parallel partial trimmed leaves");
		expect_equal(symmetric_result.stats.total_leaves > 0, true, "BGT parallel partial symmetric leaves");
	}

	MPI_Finalize();
	return 0;
}
