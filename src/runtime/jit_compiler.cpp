#include "bgt/runtime.hpp"

#include "bgt/detail/performance.hpp"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <utility>
#include <vector>

namespace
{
std::string kernel_spec_id(const bgt::KernelSpec &spec)
{
	std::ostringstream out;
	out << "kernel-"
		<< bgt::KernelSpecHash{}(spec)
		<< "-p" << static_cast<int>(spec.provider)
		<< "-n" << spec.subjects
		<< "-v" << spec.variants
		<< "-s" << spec.state_bits
		<< "-d" << static_cast<int>(spec.dilution)
		<< "-q" << static_cast<int>(spec.selector)
		<< "-r" << static_cast<int>(spec.precision)
		<< "-a" << static_cast<int>(spec.accumulator_precision)
		<< "-simd" << static_cast<int>(spec.compile_options.enable_simd)
		<< "-omp" << static_cast<int>(spec.compile_options.enable_openmp)
		<< "-cuda" << static_cast<int>(spec.compile_options.enable_cuda)
		<< "-nccl" << static_cast<int>(spec.compile_options.enable_nccl)
		<< "-ncclgin" << static_cast<int>(spec.compile_options.enable_nccl_gin)
		<< "-posterior" << static_cast<int>(spec.compile_options.posterior_precision)
		<< "-accumulator" << static_cast<int>(spec.compile_options.accumulator_precision)
		<< "-opt" << static_cast<int>(spec.compile_options.optimization)
		<< "-profile" << static_cast<int>(spec.compile_options.profile_backend);
	if (!spec.cuda_arch.empty())
		out << "-" << spec.cuda_arch;
	return out.str();
}

void write_compile_options(std::ostream &out, const bgt::CompileOptions &options)
{
	auto write_vector = [&out](const char *name, const std::vector<std::string> &values) {
		out << name << '=';
		for (std::size_t i = 0; i < values.size(); i++)
		{
			if (i != 0)
				out << ';';
			out << values[i];
		}
		out << '\n';
	};
	out << "compile.enable_simd=" << options.enable_simd << '\n'
		<< "compile.enable_openmp=" << options.enable_openmp << '\n'
		<< "compile.enable_cuda=" << options.enable_cuda << '\n'
		<< "compile.enable_nccl=" << options.enable_nccl << '\n'
		<< "compile.enable_nccl_gin=" << options.enable_nccl_gin << '\n'
		<< "compile.fast_math=" << options.fast_math << '\n'
		<< "compile.native_cpu=" << options.native_cpu << '\n'
		<< "compile.posterior_precision=" << static_cast<int>(options.posterior_precision) << '\n'
		<< "compile.accumulator_precision=" << static_cast<int>(options.accumulator_precision) << '\n'
		<< "compile.optimization=" << static_cast<int>(options.optimization) << '\n'
		<< "compile.profile_backend=" << static_cast<int>(options.profile_backend) << '\n';
	write_vector("compile.extra_cxx_flags", options.extra_cxx_flags);
	write_vector("compile.extra_cuda_flags", options.extra_cuda_flags);
	write_vector("compile.definitions", options.definitions);
}

std::filesystem::path kernel_cache_path(const std::filesystem::path &cache_directory, const bgt::KernelSpec &spec)
{
	return cache_directory / (kernel_spec_id(spec) + ".bgtjit");
}

} // namespace

namespace bgt
{

JitCompiler::JitCompiler() : JitCompiler(std::filesystem::temp_directory_path() / "bayesian_group_testing_jit")
{
}

JitCompiler::JitCompiler(std::filesystem::path cache_directory) : cache_directory_(std::move(cache_directory))
{
	enabled_ = true;
	std::filesystem::create_directories(cache_directory_);
}

bool JitCompiler::compile(const KernelSpec &spec)
{
	BGT_PERF_SCOPE_FOR(LogSubsystem::jit, "jit.compile");
	std::filesystem::create_directories(cache_directory_);
	const std::filesystem::path artifact = kernel_cache_path(cache_directory_, spec);
	if (std::filesystem::exists(artifact))
		return true;

	std::ofstream out(artifact, std::ios::trunc);
	out << "provider=" << static_cast<int>(spec.provider) << '\n'
		<< "subjects=" << spec.subjects << '\n'
		<< "variants=" << spec.variants << '\n'
		<< "state_bits=" << spec.state_bits << '\n'
		<< "dilution=" << static_cast<int>(spec.dilution) << '\n'
		<< "selector=" << static_cast<int>(spec.selector) << '\n'
		<< "precision=" << static_cast<int>(spec.precision) << '\n'
		<< "cuda_arch=" << spec.cuda_arch << '\n';
	write_compile_options(out, spec.compile_options);
	out << "backend=generic-provider-fallback\n";
	if (!out)
		return false;
	return true;
}

std::size_t JitCompiler::cache_entry_count() const
{
	if (!std::filesystem::exists(cache_directory_))
		return 0;
	std::size_t count = 0;
	for (const auto &entry : std::filesystem::directory_iterator(cache_directory_))
	{
		if (entry.is_regular_file() && entry.path().extension() == ".bgtjit")
			count++;
	}
	return count;
}

void JitCompiler::clear_cache()
{
	if (std::filesystem::exists(cache_directory_))
		std::filesystem::remove_all(cache_directory_);
	std::filesystem::create_directories(cache_directory_);
}

} // namespace bgt
