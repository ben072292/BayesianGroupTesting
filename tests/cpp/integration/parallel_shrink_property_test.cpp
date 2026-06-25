#include "bgt/detail/model/lattice.hpp"

#include <cmath>
#include <iostream>
#include <mpi.h>
#include <vector>

namespace
{
void expect_near(double actual, double expected, const char *label)
{
	const double tolerance = 1e-11;
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

void compare_local_and_distributed_after_shrink(int subjects, int variants, std::vector<double> prior, double threshold)
{
	using namespace bgt::model;

	lattice_mpi_initialize(DIST_NON_DILUTION, subjects, variants);
	auto distributed = create_lattice(DIST_NON_DILUTION, subjects, variants, prior.data());
	auto replicated = create_lattice(REPL_NON_DILUTION, subjects, variants, prior.data());

	const bool converted = distributed->update_metadata_with_shrinking(threshold, threshold);
	if (converted)
		distributed = distributed->to_local();
	replicated->update_metadata_with_shrinking(threshold, threshold);

	expect_equal(distributed->curr_subjs(), replicated->curr_subjs(), "shrink parity current subjects");
	expect_equal(distributed->pos_clas_atoms(), replicated->pos_clas_atoms(), "shrink parity positive mask");
	expect_equal(distributed->neg_clas_atoms(), replicated->neg_clas_atoms(), "shrink parity negative mask");
	expect_equal(distributed->total_states(), replicated->total_states(), "shrink parity total states");
	for (int state = 0; state < replicated->total_states(); ++state)
	{
		expect_near(distributed->posterior_prob(static_cast<bgt_state_t>(state)),
					replicated->posterior_prob(static_cast<bgt_state_t>(state)),
					"distributed posterior equals replicated posterior");
	}

	lattice_mpi_finalize(DIST_NON_DILUTION);
}
}

int main(int argc, char **argv)
{
	MPI_Init(&argc, &argv);

	compare_local_and_distributed_after_shrink(3, 1, {0.95, 0.5, 0.95}, 0.1);
	compare_local_and_distributed_after_shrink(2, 2, {0.95, 0.5, 0.95, 0.5}, 0.1);
	compare_local_and_distributed_after_shrink(3, 2, {0.95, 0.5, 0.95, 0.95, 0.5, 0.95}, 0.1);

	MPI_Finalize();
	return 0;
}
