#pragma once

#include "bgt/logging.hpp"

#define BGT_LOG_FATAL(subsystem, ...) ::bgt::log_message(::bgt::LogLevel::fatal, subsystem, __FILE__, __LINE__, __VA_ARGS__)
#define BGT_LOG_ERROR(subsystem, ...) ::bgt::log_message(::bgt::LogLevel::error, subsystem, __FILE__, __LINE__, __VA_ARGS__)
#define BGT_LOG_WARN(subsystem, ...) ::bgt::log_message(::bgt::LogLevel::warn, subsystem, __FILE__, __LINE__, __VA_ARGS__)
#define BGT_LOG_INFO(subsystem, ...) ::bgt::log_message(::bgt::LogLevel::info, subsystem, __FILE__, __LINE__, __VA_ARGS__)

#ifndef BGT_MINIMAL_LOGGING
#define BGT_LOG_DEBUG(subsystem, ...) ::bgt::log_message(::bgt::LogLevel::debug, subsystem, __FILE__, __LINE__, __VA_ARGS__)
#define BGT_LOG_TRACE(subsystem, ...) ::bgt::log_message(::bgt::LogLevel::trace, subsystem, __FILE__, __LINE__, __VA_ARGS__)
#else
#define BGT_LOG_DEBUG(subsystem, ...) ((void)0)
#define BGT_LOG_TRACE(subsystem, ...) ((void)0)
#endif
