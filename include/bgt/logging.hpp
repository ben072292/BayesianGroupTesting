#pragma once

#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace bgt
{

enum class LogLevel
{
	off = 0,
	fatal = 1,
	error = 2,
	warn = 3,
	info = 4,
	debug = 5,
	trace = 6
};

enum class LogSubsystem
{
	core,
	model,
	tree,
	runtime,
	parallel,
	cuda,
	jit,
	perf
};

struct LogConfig
{
	LogLevel level = LogLevel::warn;
	std::vector<LogSubsystem> subsystems;
	std::vector<int> rank_filter;
	bool color = false;
	bool timestamps = true;
	std::filesystem::path file_path;
};

void set_log_config(LogConfig config);
LogConfig get_log_config();
void set_log_level(LogLevel level);
bool log_enabled(LogLevel level, LogSubsystem subsystem);
LogLevel parse_log_level(std::string_view value);
LogSubsystem parse_log_subsystem(std::string_view value);
const char *log_level_name(LogLevel level);
const char *log_subsystem_name(LogSubsystem subsystem);
void log_message(LogLevel level, LogSubsystem subsystem, const char *file, int line, const char *fmt, ...);

} // namespace bgt
