#pragma once

#include "bgt/lattice.hpp"
#include "bgt/simulation.hpp"

#include <filesystem>
#include <span>
#include <unordered_map>
#include <vector>

namespace bgt
{

class KernelRegistry
{
public:
	bool contains(const KernelSpec &spec) const;
	void insert(const KernelSpec &spec);
	void clear();
	std::size_t size() const;

private:
	std::unordered_map<KernelSpec, bool, KernelSpecHash> entries_;
};

class JitCompiler
{
public:
	JitCompiler();
	explicit JitCompiler(std::filesystem::path cache_directory);

	const std::filesystem::path &cache_directory() const { return cache_directory_; }
	bool enabled() const { return enabled_; }
	bool compile(const KernelSpec &spec);
	std::size_t cache_entry_count() const;
	void clear_cache();

private:
	std::filesystem::path cache_directory_;
	bool enabled_ = false;
};

class CpuProvider
{
public:
	TreeStats run(LatticeType type, int subjects, int variants, std::span<const host_probability_t> prior,
				  const SimulationOptions &options, const DilutionTable *dilution) const;
};

class CudaProvider
{
public:
	bool available() const;
	TreeStats run(LatticeType type, int subjects, int variants, std::span<const host_probability_t> prior,
				  const SimulationOptions &options, const DilutionTable *dilution) const;
};

class Runtime
{
public:
	Runtime();
	explicit Runtime(std::filesystem::path cache_directory);

	std::vector<Provider> available_providers() const;
	JitCacheInfo jit_cache_info() const;
	void clear_jit_cache();
	KernelSpec make_kernel_spec(const SimulationConfig &config) const;
	SimulationResult run_simulation(const SimulationConfig &config);

private:
	KernelRegistry registry_;
	JitCompiler jit_;
	CudaProvider cuda_;
};

bool cuda_available();
std::vector<Provider> available_providers();
SimulationResult run_simulation(const SimulationConfig &config);

} // namespace bgt
