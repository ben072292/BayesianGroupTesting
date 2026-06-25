#include "support/assertions.hpp"

#include <filesystem>
#include <fstream>
#include <sstream>

namespace
{

std::string read_text(const std::filesystem::path &path)
{
	std::ifstream in(path);
	std::ostringstream buffer;
	buffer << in.rdbuf();
	return buffer.str();
}

void expect_contains(const std::string &text, const std::string &needle, const std::string &label)
{
	bgt::test::expect_true(text.find(needle) != std::string::npos, label);
}

} // namespace

int main()
{
	bgt::ProgressConfig defaults;
	bgt::test::expect_true(!defaults.enabled, "progress disabled by default");
	bgt::test::expect_near(defaults.interval_seconds, 5.0, "progress default interval");
	bgt::test::expect_true(defaults.write_jsonl, "progress JSONL default");
	bgt::test::expect_true(!defaults.print_stderr, "progress stderr default");

	const std::filesystem::path output_dir = bgt::test::temporary_path("bgt-simulation-progress-test");
	std::filesystem::remove_all(output_dir);
	std::filesystem::create_directories(output_dir);

	bgt::SimulationConfig config = bgt::test::binary_single_subject_config();
	config.subjects = 2;
	config.prior = {0.2, 0.3};
	config.options.search_depth = 2;
	config.report.write_csv = true;
	config.report.output_directory = output_dir;
	config.report.run_name = "completed";
	config.progress.enabled = true;
	config.progress.interval_seconds = 60.0;

	bgt::clear_simulation_cancellation();
	bgt::SimulationResult completed = bgt::run_simulation(config);
	bgt::test::expect_equal(completed.status, bgt::SimulationRunStatus::completed, "completed status");
	bgt::test::expect_equal(completed.termination_reason, bgt::TerminationReason::none, "completed termination reason");
	bgt::test::expect_equal(completed.total_states, 4, "completed total states");
	bgt::test::expect_equal(completed.evaluated_states, 4, "completed evaluated states");
	bgt::test::expect_near(completed.evaluated_prior_mass, 1.0, "completed prior mass", 1e-12);
	bgt::test::expect_true(std::filesystem::exists(output_dir / "completed.csv"), "completed report exists");
	bgt::test::expect_true(std::filesystem::exists(output_dir / "completed.progress.json"), "completed progress JSON exists");
	bgt::test::expect_true(std::filesystem::exists(output_dir / "completed.progress.jsonl"), "completed progress JSONL exists");

	const std::string completed_json = read_text(output_dir / "completed.progress.json");
	expect_contains(completed_json, "\"status\":\"completed\"", "completed JSON status");
	expect_contains(completed_json, "\"evaluated_states\":4", "completed JSON evaluated states");
	expect_contains(completed_json, "\"evaluated_prior_mass\":", "completed JSON prior mass field");

	const std::string completed_report = read_text(output_dir / "completed.csv");
	expect_contains(completed_report, "status,completed", "completed report status");
	expect_contains(completed_report, "is_partial,false", "completed report partial flag");

	config.report.run_name = "interrupted";
	bgt::request_simulation_cancellation(bgt::TerminationReason::cancellation_requested);
	bgt::SimulationResult interrupted = bgt::run_simulation(config);
	bgt::test::expect_equal(interrupted.status, bgt::SimulationRunStatus::interrupted, "interrupted status");
	bgt::test::expect_equal(interrupted.termination_reason, bgt::TerminationReason::cancellation_requested, "interrupted reason");
	bgt::test::expect_equal(interrupted.evaluated_states, 0, "interrupted evaluated states");
	bgt::test::expect_near(interrupted.evaluated_prior_mass, 0.0, "interrupted prior mass");
	bgt::test::expect_true(std::filesystem::exists(output_dir / "interrupted.partial.csv"), "partial report exists");
	bgt::test::expect_true(std::filesystem::exists(output_dir / "interrupted.progress.json"), "interrupted progress JSON exists");

	const std::string interrupted_json = read_text(output_dir / "interrupted.progress.json");
	expect_contains(interrupted_json, "\"status\":\"interrupted\"", "interrupted JSON status");
	expect_contains(interrupted_json, "\"evaluated_states\":0", "interrupted JSON evaluated states");

	const std::string interrupted_report = read_text(output_dir / "interrupted.partial.csv");
	expect_contains(interrupted_report, "status,interrupted", "interrupted report status");
	expect_contains(interrupted_report, "is_partial,true", "interrupted report partial flag");

	bgt::clear_simulation_cancellation();
	std::filesystem::remove_all(output_dir);
	return 0;
}
