#include "bgt/logging.hpp"

#include <algorithm>
#include <cstdarg>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>

#include <mpi.h>

namespace
{

std::mutex g_log_mutex;
bgt::LogConfig g_config;
std::once_flag g_env_once;

std::string lower(std::string value)
{
	std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
		return static_cast<char>(std::tolower(c));
	});
	return value;
}

std::vector<std::string> split(std::string_view value, char delimiter)
{
	std::vector<std::string> parts;
	std::string current;
	for (char c : value)
	{
		if (c == delimiter)
		{
			if (!current.empty())
				parts.push_back(current);
			current.clear();
		}
		else
		{
			current.push_back(c);
		}
	}
	if (!current.empty())
		parts.push_back(current);
	return parts;
}

bool contains_subsystem(const bgt::LogConfig &config, bgt::LogSubsystem subsystem)
{
	return config.subsystems.empty() ||
		   std::find(config.subsystems.begin(), config.subsystems.end(), subsystem) != config.subsystems.end();
}

bool rank_allowed(const bgt::LogConfig &config)
{
	if (config.rank_filter.empty())
		return true;
	int initialized = 0;
	MPI_Initialized(&initialized);
	if (!initialized)
		return true;
	int rank = 0;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	return std::find(config.rank_filter.begin(), config.rank_filter.end(), rank) != config.rank_filter.end();
}

void apply_environment()
{
	if (const char *level = std::getenv("BGT_LOG_LEVEL"))
		g_config.level = bgt::parse_log_level(level);
	if (const char *subsystems = std::getenv("BGT_LOG_SUBSYS"))
	{
		g_config.subsystems.clear();
		for (const std::string &part : split(subsystems, ','))
			g_config.subsystems.push_back(bgt::parse_log_subsystem(part));
	}
	if (const char *file = std::getenv("BGT_LOG_FILE"))
		g_config.file_path = file;
	if (const char *ranks = std::getenv("BGT_LOG_RANKS"))
	{
		g_config.rank_filter.clear();
		for (const std::string &part : split(ranks, ','))
			g_config.rank_filter.push_back(std::stoi(part));
	}
	if (const char *color = std::getenv("BGT_LOG_COLOR"))
		g_config.color = lower(color) == "1" || lower(color) == "true" || lower(color) == "on";
	if (const char *perf = std::getenv("BGT_PERF_LEVEL"))
	{
		g_config.level = std::max(g_config.level, bgt::parse_log_level(perf));
		if (std::find(g_config.subsystems.begin(), g_config.subsystems.end(), bgt::LogSubsystem::perf) == g_config.subsystems.end())
			g_config.subsystems.push_back(bgt::LogSubsystem::perf);
	}
}

std::string timestamp()
{
	std::time_t now = std::time(nullptr);
	std::tm tm{};
#if defined(_WIN32)
	localtime_s(&tm, &now);
#else
	localtime_r(&now, &tm);
#endif
	char buffer[32] = {};
	std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", &tm);
	return buffer;
}

std::string format_message(const char *fmt, va_list args)
{
	va_list copy;
	va_copy(copy, args);
	const int size = std::vsnprintf(nullptr, 0, fmt, copy);
	va_end(copy);
	if (size <= 0)
		return fmt == nullptr ? std::string{} : std::string(fmt);
	std::vector<char> buffer(static_cast<std::size_t>(size) + 1, '\0');
	std::vsnprintf(buffer.data(), buffer.size(), fmt, args);
	return std::string(buffer.data(), static_cast<std::size_t>(size));
}

std::string prefix(const bgt::LogConfig &config, bgt::LogLevel level, bgt::LogSubsystem subsystem, const char *file, int line)
{
	std::ostringstream out;
	if (config.timestamps)
		out << timestamp() << ' ';
	out << bgt::log_level_name(level) << ' ' << bgt::log_subsystem_name(subsystem);
	int initialized = 0;
	MPI_Initialized(&initialized);
	if (initialized)
	{
		int rank = 0;
		int world = 1;
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);
		MPI_Comm_size(MPI_COMM_WORLD, &world);
		out << " rank=" << rank << '/' << world;
	}
	out << ' ' << file << ':' << line << ": ";
	return out.str();
}

} // namespace

namespace bgt
{

void set_log_config(LogConfig config)
{
	std::lock_guard lock(g_log_mutex);
	g_config = std::move(config);
}

LogConfig get_log_config()
{
	std::call_once(g_env_once, apply_environment);
	std::lock_guard lock(g_log_mutex);
	return g_config;
}

void set_log_level(LogLevel level)
{
	std::lock_guard lock(g_log_mutex);
	g_config.level = level;
}

bool log_enabled(LogLevel level, LogSubsystem subsystem)
{
	std::call_once(g_env_once, apply_environment);
	std::lock_guard lock(g_log_mutex);
	if (g_config.level == LogLevel::off || level == LogLevel::off)
		return false;
	return static_cast<int>(level) <= static_cast<int>(g_config.level) &&
		   contains_subsystem(g_config, subsystem) && rank_allowed(g_config);
}

LogLevel parse_log_level(std::string_view value)
{
	const std::string normalized = lower(std::string(value));
	if (normalized == "off")
		return LogLevel::off;
	if (normalized == "fatal")
		return LogLevel::fatal;
	if (normalized == "error")
		return LogLevel::error;
	if (normalized == "warn" || normalized == "warning")
		return LogLevel::warn;
	if (normalized == "info")
		return LogLevel::info;
	if (normalized == "debug")
		return LogLevel::debug;
	if (normalized == "trace")
		return LogLevel::trace;
	return LogLevel::warn;
}

LogSubsystem parse_log_subsystem(std::string_view value)
{
	const std::string normalized = lower(std::string(value));
	if (normalized == "core")
		return LogSubsystem::core;
	if (normalized == "model")
		return LogSubsystem::model;
	if (normalized == "tree")
		return LogSubsystem::tree;
	if (normalized == "runtime")
		return LogSubsystem::runtime;
	if (normalized == "parallel")
		return LogSubsystem::parallel;
	if (normalized == "cuda")
		return LogSubsystem::cuda;
	if (normalized == "jit")
		return LogSubsystem::jit;
	if (normalized == "perf")
		return LogSubsystem::perf;
	return LogSubsystem::runtime;
}

const char *log_level_name(LogLevel level)
{
	switch (level)
	{
	case LogLevel::off:
		return "OFF";
	case LogLevel::fatal:
		return "FATAL";
	case LogLevel::error:
		return "ERROR";
	case LogLevel::warn:
		return "WARN";
	case LogLevel::info:
		return "INFO";
	case LogLevel::debug:
		return "DEBUG";
	case LogLevel::trace:
		return "TRACE";
	}
	return "UNKNOWN";
}

const char *log_subsystem_name(LogSubsystem subsystem)
{
	switch (subsystem)
	{
	case LogSubsystem::core:
		return "core";
	case LogSubsystem::model:
		return "model";
	case LogSubsystem::tree:
		return "tree";
	case LogSubsystem::runtime:
		return "runtime";
	case LogSubsystem::parallel:
		return "parallel";
	case LogSubsystem::cuda:
		return "cuda";
	case LogSubsystem::jit:
		return "jit";
	case LogSubsystem::perf:
		return "perf";
	}
	return "unknown";
}

void log_message(LogLevel level, LogSubsystem subsystem, const char *file, int line, const char *fmt, ...)
{
	if (!log_enabled(level, subsystem))
		return;
	va_list args;
	va_start(args, fmt);
	std::string message = format_message(fmt, args);
	va_end(args);

	std::lock_guard lock(g_log_mutex);
	const std::string line_text = prefix(g_config, level, subsystem, file, line) + message + '\n';
	if (!g_config.file_path.empty())
	{
		std::ofstream out(g_config.file_path, std::ios::app);
		out << line_text;
	}
	else
	{
		std::cerr << line_text;
	}
}

} // namespace bgt
