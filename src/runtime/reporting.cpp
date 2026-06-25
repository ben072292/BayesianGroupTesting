#include "bgt/detail/runtime/simulation_helpers.hpp"
#include "bgt/detail/runtime/progress.hpp"

#include <filesystem>
#include <fstream>

namespace bgt::runtime_detail
{

std::string write_report_if_requested(const SimulationConfig &config, const SimulationResult &result)
{
	if (!config.report.write_csv || result.rank != 0)
		return {};
	std::filesystem::create_directories(config.report.output_directory);
	const std::string run_name = config.report.run_name.empty() ? simulation_mode_name(config.mode) : config.report.run_name;
	const std::string suffix = result.status == SimulationRunStatus::interrupted ? ".partial.csv" : ".csv";
	const std::filesystem::path report_path = config.report.output_directory / (run_name + suffix);
	std::ofstream out(report_path, std::ios::trunc);
	out << "field,value\n"
		<< "status," << simulation_run_status_name(result.status) << '\n'
		<< "termination_reason," << termination_reason_name(result.termination_reason) << '\n'
		<< "is_partial," << (result.status == SimulationRunStatus::interrupted ? "true" : "false") << '\n'
		<< "mode," << simulation_mode_name(result.mode) << '\n'
		<< "provider," << static_cast<int>(result.provider) << '\n'
		<< "subjects," << config.subjects << '\n'
		<< "variants," << config.variants << '\n'
		<< "world_size," << result.world_size << '\n'
		<< "total_states," << result.total_states << '\n'
		<< "evaluated_states," << result.evaluated_states << '\n'
		<< "state_coverage_fraction," << (result.total_states == 0 ? 0.0 : static_cast<double>(result.evaluated_states) / static_cast<double>(result.total_states)) << '\n'
		<< "evaluated_prior_mass," << result.evaluated_prior_mass << '\n'
		<< "progress_path," << result.progress_path << '\n'
		<< "total_leaves," << result.stats.total_leaves << '\n'
		<< "correct_probability," << result.stats.correct_probability << '\n'
		<< "incorrect_probability," << result.stats.incorrect_probability << '\n'
		<< "false_positive_probability," << result.stats.false_positive_probability << '\n'
		<< "false_negative_probability," << result.stats.false_negative_probability << '\n'
		<< "unclassified_probability," << result.stats.unclassified_probability << '\n'
		<< "expected_stages," << result.stats.expected_stages << '\n'
		<< "expected_tests," << result.stats.expected_tests << '\n'
		<< "setup_seconds," << result.timings.setup_seconds << '\n'
		<< "tree_build_seconds," << result.timings.tree_build_seconds << '\n'
		<< "evaluation_seconds," << result.timings.evaluation_seconds << '\n'
		<< "total_seconds," << result.timings.total_seconds << '\n';
	return report_path.string();
}

}
