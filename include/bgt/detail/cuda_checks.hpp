#pragma once

#include "bgt/logging.hpp"
#include "bgt/status.hpp"

#include <cuda_runtime.h>
#include <string>

namespace bgt::detail
{

inline Status cuda_status(cudaError_t code, const char *expr, const char *file, int line)
{
	if (code == cudaSuccess)
		return Status{};
	std::string message = "CUDA call failed: ";
	message += expr;
	message += ": ";
	message += cudaGetErrorName(code);
	message += " (";
	message += cudaGetErrorString(code);
	message += ")";
	return Status::backend_error(StatusCode::cuda_error, std::move(message), static_cast<int>(code), file, line);
}

inline Status cuda_last_kernel_status(const char *label, const char *file, int line)
{
	Status launch = cuda_status(cudaGetLastError(), label, file, line);
	if (!launch.ok())
		return launch;
	return cuda_status(cudaDeviceSynchronize(), label, file, line);
}

inline void cuda_free_noexcept(void *ptr, const char *file, int line) noexcept
{
	if (ptr == nullptr)
		return;
	const cudaError_t code = cudaFree(ptr);
	if (code != cudaSuccess)
	{
		try
		{
			log_message(LogLevel::warn, LogSubsystem::cuda, file, line, "cudaFree failed: %s (%s)",
						cudaGetErrorName(code), cudaGetErrorString(code));
		}
		catch (...)
		{
		}
	}
}

inline bool cuda_has_device() noexcept
{
	int count = 0;
	const cudaError_t code = cudaGetDeviceCount(&count);
	if (code != cudaSuccess)
	{
		cudaGetLastError();
		return false;
	}
	return count > 0;
}

} // namespace bgt::detail

#define BGT_CUDA_CHECK(call) ::bgt::check(::bgt::detail::cuda_status((call), #call, __FILE__, __LINE__), #call, __FILE__, __LINE__)
#define BGT_CUDA_CHECK_LAST(label) ::bgt::check(::bgt::detail::cuda_last_kernel_status((label), __FILE__, __LINE__), (label), __FILE__, __LINE__)
#define BGT_CUDA_FREE_NOEXCEPT(ptr) ::bgt::detail::cuda_free_noexcept((ptr), __FILE__, __LINE__)
