#include "bgt/detail/model/posterior_buffer.hpp"

namespace bgt::model
{

std::size_t posterior_pool_cached_buffer_count(int count)
{
	return detail::cached_posterior_buffer_count<bgt::posterior_t>(count);
}

void clear_posterior_buffer_pool()
{
	detail::clear_posterior_buffer_pool<bgt::posterior_t>();
}

} // namespace bgt::model
