#pragma once

#include "bgt/types.hpp"

#include <cstddef>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

namespace bgt::model
{

namespace detail
{
template <typename T>
std::unordered_map<int, std::vector<std::unique_ptr<T[]>>> &posterior_buffer_pool()
{
	thread_local std::unordered_map<int, std::vector<std::unique_ptr<T[]>>> pool;
	return pool;
}

template <typename T>
T *acquire_posterior_buffer(int count)
{
	if (count <= 0)
		return nullptr;
	std::vector<std::unique_ptr<T[]>> &bucket = posterior_buffer_pool<T>()[count];
	if (bucket.empty())
		return std::make_unique<T[]>(static_cast<std::size_t>(count)).release();
	std::unique_ptr<T[]> ret = std::move(bucket.back());
	bucket.pop_back();
	return ret.release();
}

template <typename T>
void release_posterior_buffer(T *buffer, int count)
{
	if (buffer == nullptr)
		return;
	if (count <= 0)
	{
		std::unique_ptr<T[]> invalid_buffer(buffer);
		return;
	}
	posterior_buffer_pool<T>()[count].emplace_back(buffer);
}

template <typename T>
std::size_t cached_posterior_buffer_count(int count)
{
	const auto iter = posterior_buffer_pool<T>().find(count);
	return iter == posterior_buffer_pool<T>().end() ? 0 : iter->second.size();
}

template <typename T>
void clear_posterior_buffer_pool()
{
	posterior_buffer_pool<T>().clear();
}
} // namespace detail

template <typename T>
class BasicPosteriorBuffer
{
	T *_data = nullptr;
	int _count = 0;
	bool _owning = false;

public:
	BasicPosteriorBuffer() = default;
	BasicPosteriorBuffer(const BasicPosteriorBuffer &) = delete;
	BasicPosteriorBuffer &operator=(const BasicPosteriorBuffer &) = delete;

	BasicPosteriorBuffer(BasicPosteriorBuffer &&other) noexcept
		: _data(other._data), _count(other._count), _owning(other._owning)
	{
		other._data = nullptr;
		other._count = 0;
		other._owning = false;
	}

	BasicPosteriorBuffer &operator=(BasicPosteriorBuffer &&other) noexcept
	{
		if (this != &other)
		{
			reset();
			_data = other._data;
			_count = other._count;
			_owning = other._owning;
			other._data = nullptr;
			other._count = 0;
			other._owning = false;
		}
		return *this;
	}

	~BasicPosteriorBuffer()
	{
		reset();
	}

	static BasicPosteriorBuffer allocate(int count)
	{
		return BasicPosteriorBuffer(detail::acquire_posterior_buffer<T>(count), count, true);
	}

	static BasicPosteriorBuffer own(T *data, int count)
	{
		return BasicPosteriorBuffer(data, count, true);
	}

	static BasicPosteriorBuffer borrow(T *data, int count)
	{
		return BasicPosteriorBuffer(data, count, false);
	}

	T *data() const { return _data; }
	int count() const { return _count; }
	bool owning() const { return _owning; }
	explicit operator bool() const { return _data != nullptr; }

	T *release()
	{
		T *ret = _data;
		_data = nullptr;
		_count = 0;
		_owning = false;
		return ret;
	}

	void reset()
	{
		if (_data != nullptr && _owning)
			detail::release_posterior_buffer(_data, _count);
		_data = nullptr;
		_count = 0;
		_owning = false;
	}

private:
	BasicPosteriorBuffer(T *data, int count, bool owning)
		: _data(data), _count(data == nullptr ? 0 : count), _owning(data != nullptr && owning)
	{
	}
};

using PosteriorBuffer = BasicPosteriorBuffer<bgt::posterior_t>;

std::size_t posterior_pool_cached_buffer_count(int count);
void clear_posterior_buffer_pool();

} // namespace bgt::model
