#include "bgt/detail/performance.hpp"

#ifdef BGT_ENABLE_CALIPER_PROFILING
#include <caliper/cali.h>
#endif

#include <utility>

namespace bgt::detail
{

PerfScope::PerfScope(std::string name, LogSubsystem subsystem)
#ifdef BGT_ENABLE_CALIPER_PROFILING
	: name_(std::move(name))
#endif
{
#ifndef BGT_ENABLE_CALIPER_PROFILING
	(void)name;
	(void)subsystem;
#else
	(void)subsystem;
	CALI_MARK_BEGIN(name_.c_str());
#endif
}

PerfScope::~PerfScope()
{
#ifdef BGT_ENABLE_CALIPER_PROFILING
	CALI_MARK_END(name_.c_str());
#endif
}

} // namespace bgt::detail
