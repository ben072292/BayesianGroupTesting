#include "support/assertions.hpp"

#include <filesystem>

int main()
{
	bgt::SimulationConfig config = bgt::test::binary_single_subject_config();
	bgt::Runtime runtime;
	bgt::KernelSpec spec = runtime.make_kernel_spec(config);
	bgt::test::expect_equal(spec.subjects, 1, "KernelSpec subjects");
	bgt::test::expect_equal(spec.variants, 1, "KernelSpec variants");
	bgt::test::expect_equal(runtime.jit_cache_info().jit_enabled, true, "JIT cache enabled flag");

	const std::filesystem::path cache_dir = bgt::test::temporary_path("bgt_runtime_jit_test_cache");
	bgt::Runtime cached_runtime(cache_dir);
	cached_runtime.clear_jit_cache();
	bgt::TreeStats cached_stats = cached_runtime.run_simulation(config).stats;
	bgt::test::expect_same_stats(cached_stats, bgt::test::binary_single_subject_expected_stats(), "cached runtime stats");
	bgt::JitCacheInfo cache_info = cached_runtime.jit_cache_info();
	bgt::test::expect_equal(cache_info.memory_entries, static_cast<std::size_t>(1), "JIT memory cache entries");
	bgt::test::expect_equal(cache_info.disk_entries, static_cast<std::size_t>(1), "JIT disk cache entries");
	cached_runtime.clear_jit_cache();

	return 0;
}
