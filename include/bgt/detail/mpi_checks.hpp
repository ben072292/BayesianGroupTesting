#pragma once

#include "bgt/status.hpp"

#include <mpi.h>
#include <string>

namespace bgt::detail
{

inline Status mpi_status(int code, const char *expr, const char *file, int line)
{
	if (code == MPI_SUCCESS)
		return Status{};
	char buffer[MPI_MAX_ERROR_STRING] = {};
	int length = 0;
	MPI_Error_string(code, buffer, &length);
	std::string message = "MPI call failed: ";
	message += expr;
	message += ": ";
	message.append(buffer, static_cast<std::size_t>(length));
	return Status::backend_error(StatusCode::mpi_error, std::move(message), code, file, line);
}

} // namespace bgt::detail

#define BGT_MPI_CHECK(call) ::bgt::check(::bgt::detail::mpi_status((call), #call, __FILE__, __LINE__), #call, __FILE__, __LINE__)
