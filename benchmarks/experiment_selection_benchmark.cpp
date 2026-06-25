#include "bgt/detail/cuda_provider.hpp"
#include "bgt/detail/model/lattice.hpp"

#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

using Clock = std::chrono::steady_clock;

enum class Provider
{
	cpu,
	cuda
};

enum class CpuPath
{
	fastest,
	serial
};

enum class ReferenceMode
{
	none,
	brute_force
};

struct Options
{
	Provider provider = Provider::cpu;
	CpuPath cpu_path = CpuPath::fastest;
	ReferenceMode reference = ReferenceMode::none;
	bgt::SelectorType selector = bgt::SelectorType::op_bha;
	int subjects = 30;
	int variants = 1;
	int iterations = 3;
	int warmup = 1;
	double prior = 0.02;
};

std::string value_after_equals(const std::string &arg)
{
	const std::size_t pos = arg.find('=');
	if (pos == std::string::npos)
		throw std::invalid_argument("expected --key=value argument: " + arg);
	return arg.substr(pos + 1);
}

Provider parse_provider(const std::string &value)
{
	if (value == "cpu")
		return Provider::cpu;
	if (value == "cuda")
		return Provider::cuda;
	throw std::invalid_argument("unknown provider: " + value);
}

CpuPath parse_cpu_path(const std::string &value)
{
	if (value == "fastest")
		return CpuPath::fastest;
	if (value == "serial")
		return CpuPath::serial;
	throw std::invalid_argument("unknown cpu path: " + value);
}

bgt::SelectorType parse_selector(const std::string &value)
{
	if (value == "op_bha")
		return bgt::SelectorType::op_bha;
	if (value == "op_bbpa")
		return bgt::SelectorType::op_bbpa;
	throw std::invalid_argument("unknown selector: " + value);
}

ReferenceMode parse_reference(const std::string &value)
{
	if (value == "none")
		return ReferenceMode::none;
	if (value == "brute_force")
		return ReferenceMode::brute_force;
	throw std::invalid_argument("unknown reference mode: " + value);
}

const char *provider_name(Provider provider)
{
	return provider == Provider::cpu ? "cpu" : "cuda";
}

const char *cpu_path_name(CpuPath path)
{
	return path == CpuPath::fastest ? "fastest" : "serial";
}

const char *reference_name(ReferenceMode reference)
{
	return reference == ReferenceMode::none ? "none" : "brute_force";
}

const char *selector_name(bgt::SelectorType selector)
{
	switch (selector)
	{
	case bgt::SelectorType::op_bha:
		return "op_bha";
	case bgt::SelectorType::op_bbpa:
		return "op_bbpa";
	case bgt::SelectorType::auto_select:
		return "auto_select";
	}
	return "unknown";
}

Options parse_options(int argc, char **argv)
{
	Options options;
	for (int i = 1; i < argc; i++)
	{
		const std::string arg = argv[i];
		if (arg == "--help")
		{
			throw std::invalid_argument(
				"usage: experiment_selection_benchmark --provider=cpu|cuda "
				"--subjects=N --variants=K --selector=op_bha|op_bbpa "
				"[--iterations=N] [--warmup=N] [--prior=P] [--cpu-path=fastest|serial] "
				"[--reference=none|brute_force]");
		}
		if (arg.rfind("--provider=", 0) == 0)
			options.provider = parse_provider(value_after_equals(arg));
		else if (arg.rfind("--cpu-path=", 0) == 0)
			options.cpu_path = parse_cpu_path(value_after_equals(arg));
		else if (arg.rfind("--reference=", 0) == 0)
			options.reference = parse_reference(value_after_equals(arg));
		else if (arg.rfind("--selector=", 0) == 0)
			options.selector = parse_selector(value_after_equals(arg));
		else if (arg.rfind("--subjects=", 0) == 0)
			options.subjects = std::stoi(value_after_equals(arg));
		else if (arg.rfind("--variants=", 0) == 0)
			options.variants = std::stoi(value_after_equals(arg));
		else if (arg.rfind("--iterations=", 0) == 0)
			options.iterations = std::stoi(value_after_equals(arg));
		else if (arg.rfind("--warmup=", 0) == 0)
			options.warmup = std::stoi(value_after_equals(arg));
		else if (arg.rfind("--prior=", 0) == 0)
			options.prior = std::stod(value_after_equals(arg));
		else
			throw std::invalid_argument("unknown argument: " + arg);
	}

	const int atoms = options.subjects * options.variants;
	if (options.subjects <= 0 || options.variants <= 0 || atoms <= 0 || atoms > bgt::kStateBits)
		throw std::invalid_argument("invalid lattice dimensions.");
	if (options.iterations <= 0 || options.warmup < 0)
		throw std::invalid_argument("iterations must be positive and warmup must be nonnegative.");
	if (options.selector == bgt::SelectorType::op_bha && options.variants != 1)
		throw std::invalid_argument("op_bha requires variants=1.");
	if (options.reference == ReferenceMode::brute_force && options.provider != Provider::cpu)
		throw std::invalid_argument("brute-force reference is CPU-only.");
	return options;
}

double mean(const std::vector<double> &values)
{
	return std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
}

double standard_deviation(const std::vector<double> &values, double average)
{
	double sum = 0.0;
	for (const double value : values)
	{
		const double diff = value - average;
		sum += diff * diff;
	}
	return std::sqrt(sum / static_cast<double>(values.size()));
}

struct BenchmarkResult
{
	unsigned long long candidate = 0;
	double init_seconds = 0.0;
	double mean_seconds = 0.0;
	double min_seconds = 0.0;
	double max_seconds = 0.0;
	double stddev_seconds = 0.0;
};

int response_bucket(bgt_state_t experiment, bgt_state_t state, int subjects, int variants)
{
	int response = 0;
	const bgt_state_t subject_mask = static_cast<bgt_state_t>(bgt::state_count(subjects) - 1);
	for (int variant = 0; variant < variants; variant++)
	{
		const bgt_state_t variant_state =
			static_cast<bgt_state_t>((state >> (variant * subjects)) & subject_mask);
		const bgt_state_t missing = static_cast<bgt_state_t>(experiment & ~variant_state);
		response |= static_cast<int>(missing != 0) << variant;
	}
	return response;
}

bgt_state_t brute_force_reference_candidate(const bgt::model::Lattice &lattice, int subjects, int variants)
{
	const int experiment_count = bgt::state_count(subjects);
	const int response_count = bgt::state_count(variants);
	const int state_count = bgt::state_count(subjects * variants);
	const bgt::accumulator_t target_mass =
		bgt::accumulator_t{1.0} / static_cast<bgt::accumulator_t>(response_count);
	const bgt::posterior_t *posterior = lattice.posterior_probs();
	bgt_state_t best_candidate = 0;
	bgt::accumulator_t best_score = std::numeric_limits<bgt::accumulator_t>::infinity();
	std::vector<bgt::accumulator_t> response_mass(static_cast<std::size_t>(response_count));

	for (int experiment = 0; experiment < experiment_count; experiment++)
	{
		std::fill(response_mass.begin(), response_mass.end(), bgt::accumulator_t{0.0});
		for (int state = 0; state < state_count; state++)
		{
			const int response =
				response_bucket(static_cast<bgt_state_t>(experiment), static_cast<bgt_state_t>(state), subjects, variants);
			response_mass[static_cast<std::size_t>(response)] +=
				static_cast<bgt::accumulator_t>(posterior[state]);
		}

		bgt::accumulator_t score = bgt::accumulator_t{0.0};
		for (const bgt::accumulator_t mass : response_mass)
			score += std::abs(mass - target_mass);
		if (score < best_score)
		{
			best_score = score;
			best_candidate = static_cast<bgt_state_t>(experiment);
		}
	}
	return best_candidate;
}

BenchmarkResult run_cpu(const Options &options, const std::vector<bgt::host_probability_t> &prior)
{
	BenchmarkResult result;
	const auto init_start = Clock::now();
	bgt::model::lattice_mpi_initialize(REPL_NON_DILUTION, options.subjects, options.variants);
	std::unique_ptr<bgt::model::Lattice> lattice =
		bgt::model::create_lattice(
			REPL_NON_DILUTION, options.subjects, options.variants,
			const_cast<bgt::host_probability_t *>(prior.data()));
	const auto init_stop = Clock::now();
	result.init_seconds = std::chrono::duration<double>(init_stop - init_start).count();

	auto select_once = [&]() {
		if (options.reference == ReferenceMode::brute_force)
			return brute_force_reference_candidate(*lattice, options.subjects, options.variants);
		if (options.cpu_path == CpuPath::serial)
			return lattice->select_experiment_serial(options.selector);
		return lattice->select_experiment(options.selector);
	};

	for (int i = 0; i < options.warmup; i++)
		(void)select_once();

	std::vector<double> timings;
	timings.reserve(static_cast<std::size_t>(options.iterations));
	bgt_state_t candidate = 0;
	for (int i = 0; i < options.iterations; i++)
	{
		const auto start = Clock::now();
		candidate = select_once();
		const auto stop = Clock::now();
		timings.push_back(std::chrono::duration<double>(stop - start).count());
	}

	result.candidate = static_cast<unsigned long long>(candidate);
	result.mean_seconds = mean(timings);
	const auto [min_it, max_it] = std::minmax_element(timings.begin(), timings.end());
	result.min_seconds = *min_it;
	result.max_seconds = *max_it;
	result.stddev_seconds = standard_deviation(timings, result.mean_seconds);
	lattice.reset();
	bgt::model::lattice_mpi_finalize(REPL_NON_DILUTION);
	return result;
}

BenchmarkResult run_cuda(const Options &options, const std::vector<bgt::host_probability_t> &prior)
{
#ifdef BGT_ENABLE_CUDA
	BenchmarkResult result;
	bgt::state_t candidate = 0;
	const int status = bgt_cuda_provider_benchmark_select(
		options.subjects, options.variants, prior.data(), static_cast<int>(options.selector),
		options.iterations, options.warmup, &candidate, &result.init_seconds,
		&result.mean_seconds, &result.min_seconds, &result.max_seconds, &result.stddev_seconds);
	if (status != 0)
		throw std::runtime_error("CUDA selection benchmark failed with status " + std::to_string(status));
	result.candidate = static_cast<unsigned long long>(candidate);
	return result;
#else
	(void)options;
	(void)prior;
	throw std::runtime_error("CUDA provider was not compiled.");
#endif
}

} // namespace

int main(int argc, char **argv)
{
	MPI_Init(&argc, &argv);
	int rank = 0;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	try
	{
		const Options options = parse_options(argc, argv);
		std::vector<bgt::host_probability_t> prior(
			static_cast<std::size_t>(options.subjects * options.variants), options.prior);
		const BenchmarkResult result =
			options.provider == Provider::cpu ? run_cpu(options, prior) : run_cuda(options, prior);

		if (rank == 0)
		{
			const int atoms = options.subjects * options.variants;
			const double posterior_gib =
				static_cast<double>(bgt::state_count(atoms)) * sizeof(bgt::posterior_t) / (1024.0 * 1024.0 * 1024.0);
			std::cout << std::setprecision(10)
					  << "provider,cpu_path,reference,selector,subjects,variants,atoms,iterations,warmup,prior,"
					  << "candidate,init_seconds,mean_seconds,min_seconds,max_seconds,stddev_seconds,posterior_gib\n"
					  << provider_name(options.provider) << ','
					  << cpu_path_name(options.cpu_path) << ','
					  << reference_name(options.reference) << ','
					  << selector_name(options.selector) << ','
					  << options.subjects << ','
					  << options.variants << ','
					  << atoms << ','
					  << options.iterations << ','
					  << options.warmup << ','
					  << options.prior << ','
					  << result.candidate << ','
					  << result.init_seconds << ','
					  << result.mean_seconds << ','
					  << result.min_seconds << ','
					  << result.max_seconds << ','
					  << result.stddev_seconds << ','
					  << posterior_gib << '\n';
		}
	}
	catch (const std::exception &ex)
	{
		if (rank == 0)
			std::cerr << ex.what() << '\n';
		MPI_Finalize();
		return 1;
	}
	MPI_Finalize();
	return 0;
}
