#pragma once

#include "bgt/config.hpp"
#include "bgt/logging.hpp"

#include <string>

namespace bgt::detail
{

class PerfScope
{
public:
	PerfScope(std::string name, LogSubsystem subsystem = LogSubsystem::perf);
	~PerfScope();

	PerfScope(const PerfScope &) = delete;
	PerfScope &operator=(const PerfScope &) = delete;

private:
#ifdef BGT_ENABLE_CALIPER_PROFILING
	std::string name_;
#endif
};

} // namespace bgt::detail

#define BGT_DETAIL_CONCAT_IMPL(a, b) a##b
#define BGT_DETAIL_CONCAT(a, b) BGT_DETAIL_CONCAT_IMPL(a, b)

#ifdef BGT_ENABLE_CALIPER_PROFILING
#define BGT_PERF_SCOPE(name) ::bgt::detail::PerfScope BGT_DETAIL_CONCAT(bgt_perf_scope_, __LINE__)((name))
#define BGT_PERF_SCOPE_FOR(subsystem, name) \
	::bgt::detail::PerfScope BGT_DETAIL_CONCAT(bgt_perf_scope_, __LINE__)((name), (subsystem))
#else
#define BGT_PERF_SCOPE(name) ((void)0)
#define BGT_PERF_SCOPE_FOR(subsystem, name) ((void)0)
#endif
