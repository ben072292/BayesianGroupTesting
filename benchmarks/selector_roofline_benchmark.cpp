#include "bgt/detail/model/lattice.hpp"
#include "bgt/detail/runtime/simulation_helpers.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

using Clock = std::chrono::steady_clock;

enum class ModelKind
{
	replicated,
	distributed
};

struct Options
{
	ModelKind model = ModelKind::replicated;
	bgt::SelectorType selector = bgt::SelectorType::op_bha;
	int subjects = 12;
	int variants = 1;
	int iterations = 20;
	int warmup = 5;
	double prior = 0.03;
};

struct WorkEstimate
{
	double operations = 0.0;
	double memory_bytes = 0.0;
	double collective_bytes = 0.0;
};

std::string value_after_equals(const std::string &arg)
{
	const std::size_t pos = arg.find('=');
	if (pos == std::string::npos)
		throw std::invalid_argument("expected --key=value argument: " + arg);
	return arg.substr(pos + 1);
}

ModelKind parse_model(const std::string &value)
{
	if (value == "replicated")
		return ModelKind::replicated;
	if (value == "distributed")
		return ModelKind::distributed;
	throw std::invalid_argument("unknown model: " + value);
}

bgt::SelectorType parse_selector(const std::string &value)
{
	if (value == "op_bha")
		return bgt::SelectorType::op_bha;
	if (value == "op_bbpa")
		return bgt::SelectorType::op_bbpa;
	throw std::invalid_argument("unknown selector: " + value);
}

const char *model_name(ModelKind model)
{
	return model == ModelKind::replicated ? "replicated" : "distributed";
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
				"usage: selector_roofline_benchmark --model=replicated|distributed "
				"--selector=op_bha|op_bbpa --subjects=N --variants=K "
				"[--iterations=N] [--warmup=N] [--prior=P]");
		}
		if (arg.rfind("--model=", 0) == 0)
			options.model = parse_model(value_after_equals(arg));
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
	if (options.subjects <= 0 || options.variants <= 0)
		throw std::invalid_argument("subjects and variants must be positive.");
	if (options.iterations <= 0 || options.warmup < 0)
		throw std::invalid_argument("iterations must be positive and warmup must be nonnegative.");
	if (options.selector == bgt::SelectorType::op_bha && options.variants != 1)
		throw std::invalid_argument("op_bha benchmark requires variants=1.");
	return options;
}

WorkEstimate estimate_work(const Options &options, int local_states, int world_size)
{
	const double states = static_cast<double>(bgt::state_count(options.subjects * options.variants));
	const double experiments = static_cast<double>(bgt::state_count(options.subjects));
	const double responses = static_cast<double>(bgt::state_count(options.variants));
	const double local = options.model == ModelKind::distributed ? static_cast<double>(local_states) : states;
	const double posterior_bytes = sizeof(bgt::posterior_t);
	const double accumulator_bytes = sizeof(bgt::accumulator_t);
	WorkEstimate estimate;
	if (options.selector == bgt::SelectorType::op_bha)
	{
		const double zeta_adds = states * static_cast<double>(options.subjects) / 2.0;
		estimate.operations = zeta_adds + states * 2.0;
		estimate.memory_bytes =
			local * (posterior_bytes + accumulator_bytes) +
			zeta_adds * accumulator_bytes * 3.0 +
			states * accumulator_bytes;
		estimate.collective_bytes = options.model == ModelKind::distributed
										? states * accumulator_bytes * static_cast<double>(std::max(1, world_size - 1))
										: 0.0;
		return estimate;
	}

	const double feature_tables = responses - 1.0;
	const double feature_entries = responses * experiments;
	const double histogram_updates = local * feature_tables;
	const double zeta_adds = feature_tables * experiments * static_cast<double>(options.subjects) / 2.0;
	const double inclusion_terms = experiments * std::pow(3.0, static_cast<double>(options.variants));
	estimate.operations =
		histogram_updates * static_cast<double>(options.variants + 1) +
		zeta_adds +
		inclusion_terms * 2.0;
	estimate.memory_bytes =
		local * posterior_bytes +
		histogram_updates * accumulator_bytes * 2.0 +
		zeta_adds * accumulator_bytes * 3.0 +
		inclusion_terms * accumulator_bytes;
	estimate.collective_bytes = options.model == ModelKind::distributed
									? feature_entries * accumulator_bytes * static_cast<double>(std::max(1, world_size - 1))
									: 0.0;
	return estimate;
}

double mean(const std::vector<double> &values)
{
	return std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
}

double standard_deviation(const std::vector<double> &values, double average)
{
	double sum = 0.0;
	for (double value : values)
	{
		const double diff = value - average;
		sum += diff * diff;
	}
	return std::sqrt(sum / static_cast<double>(values.size()));
}

} // namespace

int main(int argc, char **argv)
{
	MPI_Init(&argc, &argv);
	int rank = 0;
	int world_size = 1;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	try
	{
		const Options options = parse_options(argc, argv);
		const auto lattice_type = options.model == ModelKind::distributed
									  ? DIST_NON_DILUTION
									  : REPL_NON_DILUTION;
		std::vector<bgt::host_probability_t> prior(
			static_cast<std::size_t>(options.subjects * options.variants), options.prior);

		bgt::model::lattice_mpi_initialize(lattice_type, options.subjects, options.variants);
		std::unique_ptr<bgt::model::Lattice> lattice =
			bgt::model::create_lattice(lattice_type, options.subjects, options.variants, prior.data());

		for (int i = 0; i < options.warmup; i++)
		{
			(void)lattice->select_experiment(options.selector);
		}

		std::vector<double> seconds;
		seconds.reserve(static_cast<std::size_t>(options.iterations));
		bgt_state_t candidate = 0;
		for (int i = 0; i < options.iterations; i++)
		{
			MPI_Barrier(MPI_COMM_WORLD);
			const auto start = Clock::now();
			candidate = lattice->select_experiment(options.selector);
			MPI_Barrier(MPI_COMM_WORLD);
			const auto stop = Clock::now();
			double local_seconds = std::chrono::duration<double>(stop - start).count();
			double max_seconds = 0.0;
			MPI_Reduce(&local_seconds, &max_seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
			if (rank == 0)
				seconds.push_back(max_seconds);
		}

		const int local_states = lattice->posterior_prob_count();
		const WorkEstimate estimate = estimate_work(options, local_states, world_size);
		if (rank == 0)
		{
			const double average = mean(seconds);
			const auto [min_it, max_it] = std::minmax_element(seconds.begin(), seconds.end());
			const double sd = standard_deviation(seconds, average);
			const double operations_per_second = estimate.operations / average;
			const double bytes_per_second = estimate.memory_bytes / average;
			const double arithmetic_intensity = estimate.memory_bytes > 0.0 ? estimate.operations / estimate.memory_bytes : 0.0;
			std::cout << std::setprecision(10)
					  << "model,selector,subjects,variants,ranks,iterations,warmup,candidate,"
					  << "mean_seconds,min_seconds,max_seconds,stddev_seconds,"
					  << "estimated_ops,estimated_memory_bytes,estimated_collective_bytes,"
					  << "effective_gops,effective_gbps,arithmetic_intensity_ops_per_byte\n"
					  << model_name(options.model) << ','
					  << selector_name(options.selector) << ','
					  << options.subjects << ','
					  << options.variants << ','
					  << world_size << ','
					  << options.iterations << ','
					  << options.warmup << ','
					  << static_cast<unsigned long long>(candidate) << ','
					  << average << ','
					  << *min_it << ','
					  << *max_it << ','
					  << sd << ','
					  << estimate.operations << ','
					  << estimate.memory_bytes << ','
					  << estimate.collective_bytes << ','
					  << operations_per_second / 1.0e9 << ','
					  << bytes_per_second / 1.0e9 << ','
					  << arithmetic_intensity << '\n';
		}

		lattice.reset();
		bgt::model::lattice_mpi_finalize(lattice_type);
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
