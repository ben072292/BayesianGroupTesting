#include "bgt/status.hpp"

#include <sstream>
#include <utility>

namespace bgt
{

Status::Status() : code_(StatusCode::success), line_(0)
{
}

Status::Status(StatusCode code, std::string message, const char *file, int line, std::optional<int> backend_code)
	: code_(code), message_(std::move(message)), file_(file == nullptr ? "" : file), line_(line), backend_code_(backend_code)
{
}

Status Status::invalid_argument(std::string message, const char *file, int line)
{
	return Status(StatusCode::invalid_argument, std::move(message), file, line);
}

Status Status::unsupported(std::string message, const char *file, int line)
{
	return Status(StatusCode::unsupported, std::move(message), file, line);
}

Status Status::runtime_error(std::string message, const char *file, int line)
{
	return Status(StatusCode::runtime_error, std::move(message), file, line);
}

Status Status::internal_error(std::string message, const char *file, int line)
{
	return Status(StatusCode::internal_error, std::move(message), file, line);
}

Status Status::backend_error(StatusCode code, std::string message, int backend_code, const char *file, int line)
{
	return Status(code, std::move(message), file, line, backend_code);
}

std::string Status::to_string() const
{
	if (ok())
		return "success";
	std::ostringstream out;
	out << status_code_name(code_) << ": " << message_;
	if (backend_code_)
		out << " (backend_code=" << *backend_code_ << ")";
	if (!file_.empty())
		out << " [" << file_ << ':' << line_ << ']';
	return out.str();
}

Error::Error(Status status) : std::runtime_error(status.to_string()), status_(std::move(status))
{
}

void check(const Status &status, const char *, const char *, int)
{
	if (!status.ok())
		throw Error(status);
}

const char *status_code_name(StatusCode code)
{
	switch (code)
	{
	case StatusCode::success:
		return "success";
	case StatusCode::invalid_argument:
		return "invalid_argument";
	case StatusCode::unsupported:
		return "unsupported";
	case StatusCode::runtime_error:
		return "runtime_error";
	case StatusCode::mpi_error:
		return "mpi_error";
	case StatusCode::cuda_error:
		return "cuda_error";
	case StatusCode::internal_error:
		return "internal_error";
	}
	return "unknown";
}

} // namespace bgt
