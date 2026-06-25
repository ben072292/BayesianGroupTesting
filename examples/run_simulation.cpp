#include "bgt/bgt.hpp"

#include <atomic>
#include <csignal>
#include <cstdlib>
#include <iostream>
#include <mpi.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
std::atomic<int> signal_count{0};

void handle_termination_signal(int signal_number)
{
	const int previous = signal_count.fetch_add(1, std::memory_order_acq_rel);
	if (previous == 0)
	{
		bgt::request_simulation_cancellation(bgt::TerminationReason::user_signal);
		return;
	}
	std::_Exit(128 + signal_number);
}

bool is_parallel_mode(bgt::SimulationMode mode)
{
	return mode != bgt::SimulationMode::local_tree;
}

bgt::SimulationMode parse_mode(const std::string &value)
{
	if (value == "local_tree")
		return bgt::SimulationMode::local_tree;
	if (value == "parallel_global_tree")
		return bgt::SimulationMode::parallel_global_tree;
	if (value == "parallel_dynamic_tree")
		return bgt::SimulationMode::parallel_dynamic_tree;
	if (value == "parallel_hybrid_tree")
		return bgt::SimulationMode::parallel_hybrid_tree;
	if (value == "parallel_partial_tree")
		return bgt::SimulationMode::parallel_partial_tree;
	if (value == "parallel_fusion_tree")
		return bgt::SimulationMode::parallel_fusion_tree;
	throw std::invalid_argument("unknown mode: " + value);
}

bgt::LatticeType parse_lattice(const std::string &value)
{
	if (value == "replicated_non_dilution")
		return bgt::LatticeType::replicated_non_dilution;
	if (value == "replicated_dilution")
		return bgt::LatticeType::replicated_dilution;
	if (value == "distributed_non_dilution")
		return bgt::LatticeType::distributed_non_dilution;
	if (value == "distributed_dilution")
		return bgt::LatticeType::distributed_dilution;
	throw std::invalid_argument("unknown lattice: " + value);
}

bgt::Provider parse_provider(const std::string &value)
{
	if (value == "auto")
		return bgt::Provider::auto_select;
	if (value == "cpu")
		return bgt::Provider::cpu;
	if (value == "cuda")
		return bgt::Provider::cuda;
	throw std::invalid_argument("unknown provider: " + value);
}

bgt::SelectorType parse_selector(const std::string &value)
{
	if (value == "auto")
		return bgt::SelectorType::auto_select;
	if (value == "op_bha")
		return bgt::SelectorType::op_bha;
	if (value == "op_bbpa")
		return bgt::SelectorType::op_bbpa;
	throw std::invalid_argument("unknown selector: " + value);
}

bgt::TrueStatePolicy parse_true_state_policy(const std::string &value)
{
	if (value == "all")
		return bgt::TrueStatePolicy::all;
	if (value == "trimmed")
		return bgt::TrueStatePolicy::trimmed;
	if (value == "symmetric")
		return bgt::TrueStatePolicy::symmetric;
	throw std::invalid_argument("unknown true-state policy: " + value);
}

std::vector<double> parse_prior(const std::string &value, int expected_size)
{
	std::vector<double> prior;
	std::stringstream stream(value);
	std::string item;
	while (std::getline(stream, item, ','))
	{
		if (!item.empty())
			prior.push_back(std::stod(item));
	}
	if (prior.size() == 1 && expected_size > 1)
	{
		const double scalar_prior = prior.front();
		prior.assign(static_cast<std::size_t>(expected_size), scalar_prior);
	}
	return prior;
}

void print_usage(const char *program)
{
	std::cerr
		<< "Usage: " << program << " --subjects N --prior P[,P...] [options]\n"
		<< "Options:\n"
		<< "  --variants K\n"
		<< "  --lattice replicated_non_dilution|replicated_dilution|distributed_non_dilution|distributed_dilution\n"
		<< "  --mode local_tree|parallel_global_tree|parallel_dynamic_tree|parallel_hybrid_tree|parallel_partial_tree|parallel_fusion_tree\n"
		<< "  --search-depth N\n"
		<< "  --global-tree-depth N\n"
		<< "  --workload-granularity N\n"
		<< "  --provider auto|cpu|cuda\n"
		<< "  --selector auto|op_bha|op_bbpa\n"
		<< "  --true-state-policy all|trimmed|symmetric\n"
		<< "  --evaluation-prior P[,P...]\n"
		<< "  --trim-percent P\n"
		<< "  --write-csv --output-dir DIR --run-name NAME\n"
		<< "  --progress [--progress-interval SECONDS] [--progress-file PATH] [--no-progress-stderr]\n";
}

class ParallelBackendSession
{
public:
	ParallelBackendSession(int *argc, char ***argv, bool enabled) : enabled_(enabled)
	{
		if (!enabled_)
			return;
		int provided = 0;
		MPI_Init_thread(argc, argv, MPI_THREAD_FUNNELED, &provided);
		if (provided < MPI_THREAD_FUNNELED)
			throw std::runtime_error("MPI_THREAD_FUNNELED is required.");
	}

	~ParallelBackendSession()
	{
		if (enabled_)
			MPI_Finalize();
	}

	ParallelBackendSession(const ParallelBackendSession &) = delete;
	ParallelBackendSession &operator=(const ParallelBackendSession &) = delete;

private:
	bool enabled_ = false;
};
} // namespace

int main(int argc, char **argv)
{
	bgt::SimulationConfig config;
	std::string prior_arg;
	std::string evaluation_prior_arg;

	for (int i = 1; i < argc; i++)
	{
		const std::string key = argv[i];
		auto require_value = [&](const char *name) -> std::string {
			if (i + 1 >= argc)
				throw std::invalid_argument(std::string("missing value for ") + name);
			return argv[++i];
		};

		if (key == "--help" || key == "-h")
		{
			print_usage(argv[0]);
			return 0;
		}
		if (key == "--subjects")
			config.subjects = std::stoi(require_value("--subjects"));
		else if (key == "--variants")
			config.variants = std::stoi(require_value("--variants"));
		else if (key == "--prior")
			prior_arg = require_value("--prior");
		else if (key == "--evaluation-prior")
			evaluation_prior_arg = require_value("--evaluation-prior");
		else if (key == "--lattice")
			config.lattice_type = parse_lattice(require_value("--lattice"));
		else if (key == "--mode")
			config.mode = parse_mode(require_value("--mode"));
		else if (key == "--search-depth")
			config.options.search_depth = std::stoi(require_value("--search-depth"));
		else if (key == "--global-tree-depth")
			config.global_tree_depth = std::stoi(require_value("--global-tree-depth"));
		else if (key == "--workload-granularity")
			config.workload_granularity = std::stoi(require_value("--workload-granularity"));
		else if (key == "--provider")
			config.options.provider = parse_provider(require_value("--provider"));
		else if (key == "--selector")
			config.options.selector = parse_selector(require_value("--selector"));
		else if (key == "--true-state-policy")
			config.true_state_policy = parse_true_state_policy(require_value("--true-state-policy"));
		else if (key == "--trim-percent")
			config.trim_percent = std::stod(require_value("--trim-percent"));
		else if (key == "--threshold-up")
			config.options.threshold_up = std::stod(require_value("--threshold-up"));
		else if (key == "--threshold-lo")
			config.options.threshold_lo = std::stod(require_value("--threshold-lo"));
		else if (key == "--branch-threshold")
			config.options.branch_threshold = std::stod(require_value("--branch-threshold"));
		else if (key == "--dilution-alpha")
			config.dilution.alpha = std::stod(require_value("--dilution-alpha"));
		else if (key == "--dilution-h")
			config.dilution.h = std::stod(require_value("--dilution-h"));
		else if (key == "--write-csv")
			config.report.write_csv = true;
		else if (key == "--output-dir")
			config.report.output_directory = require_value("--output-dir");
		else if (key == "--run-name")
			config.report.run_name = require_value("--run-name");
		else if (key == "--progress")
		{
			config.progress.enabled = true;
			config.progress.print_stderr = true;
		}
		else if (key == "--progress-interval")
			config.progress.interval_seconds = std::stod(require_value("--progress-interval"));
		else if (key == "--progress-file")
			config.progress.output_path = require_value("--progress-file");
		else if (key == "--no-progress-stderr")
			config.progress.print_stderr = false;
		else
			throw std::invalid_argument("unknown argument: " + key);
	}

	if (config.subjects <= 0 || prior_arg.empty())
	{
		print_usage(argv[0]);
		return 2;
	}

	const int prior_size = config.subjects * config.variants;
	config.prior = parse_prior(prior_arg, prior_size);
	if (!evaluation_prior_arg.empty())
		config.evaluation_prior = parse_prior(evaluation_prior_arg, prior_size);

	try
	{
		bgt::clear_simulation_cancellation();
		std::signal(SIGINT, handle_termination_signal);
		std::signal(SIGTERM, handle_termination_signal);
		ParallelBackendSession parallel_backend(&argc, &argv, is_parallel_mode(config.mode));
		const bgt::SimulationResult result = bgt::run_simulation(config);
		if (result.rank == 0)
		{
			std::cout << "mode=" << static_cast<int>(result.mode)
					  << " provider=" << static_cast<int>(result.provider)
					  << " status=" << static_cast<int>(result.status)
					  << " total_states=" << result.total_states
					  << " evaluated_states=" << result.evaluated_states
					  << " evaluated_prior_mass=" << result.evaluated_prior_mass
					  << " leaves=" << result.stats.total_leaves
					  << " correct=" << result.stats.correct_probability
					  << " expected_tests=" << result.stats.expected_tests
					  << '\n';
			if (!result.report_path.empty())
				std::cout << "report=" << result.report_path << '\n';
			if (!result.progress_path.empty())
				std::cout << "progress=" << result.progress_path << '\n';
		}
	}
	catch (const std::exception &error)
	{
		std::cerr << "bgt_simulate: " << error.what() << '\n';
		return 1;
	}

	return 0;
}
