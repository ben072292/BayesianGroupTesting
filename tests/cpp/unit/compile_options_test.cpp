#include "support/assertions.hpp"

int main()
{
	bgt::CompileOptions defaults;
	bgt::CompileOptions specialized = defaults;
	specialized.enable_simd = !defaults.enable_simd;
	specialized.posterior_precision = defaults.posterior_precision == bgt::Precision::float64 ? bgt::Precision::float32 : bgt::Precision::float64;
	specialized.accumulator_precision = defaults.accumulator_precision == bgt::Precision::float64 ? bgt::Precision::float32 : bgt::Precision::float64;
	specialized.profile_backend = defaults.profile_backend == bgt::ProfileBackend::caliper ? bgt::ProfileBackend::none : bgt::ProfileBackend::caliper;
	specialized.extra_cxx_flags = {"-DBGT_TEST_FLAG"};
	specialized.definitions = {"BGT_TEST_DEFINITION"};

	bgt::test::expect_true(defaults == defaults, "compile options self equality");
	bgt::test::expect_true(!(defaults == specialized), "compile options inequality");
	bgt::test::expect_true(bgt::CompileOptionsHash{}(defaults) == bgt::CompileOptionsHash{}(defaults), "compile options stable hash");

	bgt::KernelSpec first;
	first.subjects = 2;
	first.variants = 1;
	first.compile_options = defaults;
	bgt::KernelSpec second = first;
	second.compile_options = specialized;
	second.precision = specialized.posterior_precision;
	second.accumulator_precision = specialized.accumulator_precision;

	bgt::test::expect_true(first == first, "kernel spec self equality");
	bgt::test::expect_true(!(first == second), "kernel spec compile options inequality");
	bgt::test::expect_true(first.precision == defaults.posterior_precision, "kernel spec default precision matches compile options");
	bgt::test::expect_true(first.accumulator_precision == defaults.accumulator_precision, "kernel spec default accumulator precision matches compile options");
	bgt::test::expect_true(bgt::KernelSpecHash{}(first) == bgt::KernelSpecHash{}(first), "kernel spec stable hash");

	return 0;
}
