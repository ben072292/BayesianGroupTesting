#include "support/assertions.hpp"

#include "bgt/detail/logging_macros.hpp"

#include <filesystem>
#include <fstream>

int main()
{
	bgt::Status ok;
	bgt::test::expect_true(ok.ok(), "default status is success");

	bgt::Status backend = bgt::Status::backend_error(
		bgt::StatusCode::mpi_error, "mock backend failure", 17, "mock.cpp", 42);
	bgt::test::expect_true(!backend.ok(), "backend status is failure");
	bgt::test::expect_equal(backend.backend_code().value_or(0), 17, "backend code preserved");
	bgt::test::expect_true(backend.to_string().find("mock backend failure") != std::string::npos,
						   "status message included");

	bool threw = false;
	try
	{
		BGT_CHECK(backend);
	}
	catch (const bgt::Error &error)
	{
		threw = true;
		bgt::test::expect_equal(error.status().code(), bgt::StatusCode::mpi_error, "exception status code");
	}
	bgt::test::expect_true(threw, "BGT_CHECK throws bgt::Error");

	bgt::LogConfig config;
	config.level = bgt::LogLevel::debug;
	config.subsystems = {bgt::LogSubsystem::runtime};
	config.timestamps = false;
	const std::filesystem::path log_path = std::filesystem::temp_directory_path() / "bgt-status-logging-test.log";
	std::filesystem::remove(log_path);
	config.file_path = log_path;
	bgt::set_log_config(config);

	bgt::test::expect_true(bgt::log_enabled(bgt::LogLevel::info, bgt::LogSubsystem::runtime),
						   "runtime info log enabled");
	bgt::test::expect_true(!bgt::log_enabled(bgt::LogLevel::info, bgt::LogSubsystem::model),
						   "model log filtered");
	BGT_LOG_INFO(bgt::LogSubsystem::runtime, "status logging test %d", 123);

	std::ifstream in(log_path);
	std::string contents((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
	bgt::test::expect_true(contents.find("status logging test 123") != std::string::npos,
						   "log file contains formatted message");

	bgt::set_log_config(bgt::LogConfig{});
	std::filesystem::remove(log_path);
	return 0;
}
