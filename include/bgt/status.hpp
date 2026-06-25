#pragma once

#include <optional>
#include <stdexcept>
#include <string>

namespace bgt
{

enum class StatusCode
{
	success,
	invalid_argument,
	unsupported,
	runtime_error,
	mpi_error,
	cuda_error,
	internal_error
};

class Status
{
public:
	Status();
	Status(StatusCode code, std::string message, const char *file = nullptr, int line = 0,
		   std::optional<int> backend_code = std::nullopt);

	static Status invalid_argument(std::string message, const char *file = nullptr, int line = 0);
	static Status unsupported(std::string message, const char *file = nullptr, int line = 0);
	static Status runtime_error(std::string message, const char *file = nullptr, int line = 0);
	static Status internal_error(std::string message, const char *file = nullptr, int line = 0);
	static Status backend_error(StatusCode code, std::string message, int backend_code,
								const char *file = nullptr, int line = 0);

	bool ok() const { return code_ == StatusCode::success; }
	explicit operator bool() const { return ok(); }
	StatusCode code() const { return code_; }
	const std::string &message() const { return message_; }
	const std::string &file() const { return file_; }
	int line() const { return line_; }
	std::optional<int> backend_code() const { return backend_code_; }
	std::string to_string() const;

private:
	StatusCode code_;
	std::string message_;
	std::string file_;
	int line_;
	std::optional<int> backend_code_;
};

class Error : public std::runtime_error
{
public:
	explicit Error(Status status);
	const Status &status() const { return status_; }

private:
	Status status_;
};

void check(const Status &status, const char *expr, const char *file, int line);
const char *status_code_name(StatusCode code);

} // namespace bgt

#define BGT_CHECK(expr) ::bgt::check((expr), #expr, __FILE__, __LINE__)
#define BGT_THROW_IF_ERROR(expr) BGT_CHECK(expr)
