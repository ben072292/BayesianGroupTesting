#include "support/assertions.hpp"

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace
{

std::string read_file(const std::filesystem::path &path)
{
	std::ifstream in(path);
	return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

bool is_source_file(const std::filesystem::path &path)
{
	const std::string ext = path.extension().string();
	return ext == ".cpp" || ext == ".hpp" || ext == ".cu" || ext == ".h" || path.filename() == "CMakeLists.txt";
}

bool allowed_stderr_file(const std::filesystem::path &path)
{
	const std::string name = path.generic_string();
	return name.find("src/runtime/logging.cpp") != std::string::npos ||
		   name.find("src/runtime/progress.cpp") != std::string::npos;
}

void scan_path(const std::filesystem::path &root)
{
	for (const auto &entry : std::filesystem::recursive_directory_iterator(root))
	{
		if (!entry.is_regular_file() || !is_source_file(entry.path()))
			continue;
		const std::string text = read_file(entry.path());
		const std::string name = entry.path().generic_string();
		bgt::test::expect_true(text.find("log_debug(") == std::string::npos, name + " has no old log_debug");
		bgt::test::expect_true(text.find("log_error(") == std::string::npos, name + " has no old log_error");
		bgt::test::expect_true(text.find("ENABLE_PERF") == std::string::npos, name + " has no ENABLE_PERF");
		bgt::test::expect_true(text.find(std::string("BGT_ENABLE_") + "BUILTIN_PROFILING") == std::string::npos,
							   name + " has no built-in profiling backend");
		bgt::test::expect_true(text.find(std::string("Performance") + "Collector") == std::string::npos,
							   name + " has no built-in performance collector");
		bgt::test::expect_true(text.find("cuda_ok(") == std::string::npos, name + " has no cuda_ok");
		bgt::test::expect_true(text.find("check_last_kernel") == std::string::npos, name + " has no check_last_kernel");
		bgt::test::expect_true(text.find("third_party/log") == std::string::npos, name + " has no third-party logger");
		bgt::test::expect_true(text.find("exit(1)") == std::string::npos, name + " has no library exit(1)");
		if (!allowed_stderr_file(entry.path()))
			bgt::test::expect_true(text.find("std::cerr") == std::string::npos, name + " has no raw std::cerr");
	}
}

void check_dynamic_scheduler_contract(const std::filesystem::path &source_root)
{
	const std::string scheduler = read_file(source_root / "src/runtime/parallel_simulation.cpp");
	const std::string tree = read_file(source_root / "src/core/tree/distributed_tree.cpp");
	bgt::test::expect_true(scheduler.find("MPI_ANY_SOURCE") != std::string::npos,
						   "parallel dynamic tree uses wildcard worker readiness receives");
	bgt::test::expect_true(scheduler.find("master_dispatch_contiguous_work") != std::string::npos,
						   "parallel dynamic tree keeps a master work dispatcher");
	bgt::test::expect_true(tree.find("select_experiment_serial") != std::string::npos,
						   "parallel dynamic tree workers use local serial selector calls");
	bgt::test::expect_true(scheduler.find("MPI_Allreduce") == std::string::npos,
						   "parallel simulation scheduler has no allreduce selector path");
}

} // namespace

int main()
{
	const std::filesystem::path source_root = BGT_SOURCE_DIR;
	scan_path(source_root / "include");
	scan_path(source_root / "src");
	check_dynamic_scheduler_contract(source_root);
	bgt::test::expect_true(read_file(source_root / "CMakeLists.txt").find("ENABLE_PERF") == std::string::npos,
						   "CMake has no ENABLE_PERF");
	return 0;
}
