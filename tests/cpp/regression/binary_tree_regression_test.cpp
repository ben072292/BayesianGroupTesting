#include "support/assertions.hpp"

int main()
{
	bgt::SimulationConfig config = bgt::test::binary_single_subject_config();
	bgt::SimulationResult result = bgt::run_simulation(config);

	bgt::test::expect_equal(result.mode, bgt::SimulationMode::local_tree, "regression mode");
	bgt::test::expect_equal(result.provider, bgt::Provider::cpu, "regression provider");
	bgt::test::expect_equal(result.total_states, 2, "regression total states");
	bgt::test::expect_equal(result.evaluated_states, 2, "regression evaluated states");
	bgt::test::expect_same_stats(result.stats, bgt::test::binary_single_subject_expected_stats(), "binary tree regression");

	return 0;
}
