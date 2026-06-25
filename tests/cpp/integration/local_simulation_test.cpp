#include "support/assertions.hpp"

int main()
{
	bgt::SimulationConfig config = bgt::test::binary_single_subject_config();
	bgt::TreeStats cpu_stats = bgt::run_simulation(config).stats;
	bgt::test::expect_same_stats(cpu_stats, bgt::test::binary_single_subject_expected_stats(), "CPU binary simulation");

	config.options.provider = bgt::Provider::auto_select;
	bgt::TreeStats default_stats = bgt::run_simulation(config).stats;
	bgt::test::expect_same_stats(default_stats, cpu_stats, "default provider simulation");

	config.options.provider = bgt::Provider::cpu;
	config.variants = 2;
	config.prior = {0.2, 0.4};
	bgt::TreeStats multinomial_stats = bgt::run_simulation(config).stats;
	bgt::test::expect_equal(multinomial_stats.total_leaves, 4, "multinomial CPU leaves");

	return 0;
}
