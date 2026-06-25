#ifndef BGT_STATE_ENCODING_H_
#define BGT_STATE_ENCODING_H_

#include "bgt/types.hpp"

#include <climits>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace bgt
{

static const int kStateBits = BGT_STATE_BITS;

inline state_t state_bit(int bit)
{
	if (bit < 0 || bit >= kStateBits)
	{
		throw std::logic_error("state bit index exceeds configured state encoding width.");
	}
	return static_cast<state_t>(state_t(1) << bit);
}

inline int state_count(int bits)
{
	if (bits < 0 || bits > kStateBits || bits >= static_cast<int>(sizeof(int) * CHAR_BIT))
	{
		throw std::logic_error("state count exceeds configured state encoding width or int range.");
	}
	return 1 << bits;
}

inline int state_popcount(state_t value)
{
	if (sizeof(state_t) <= sizeof(unsigned int))
	{
		return __builtin_popcount(static_cast<unsigned int>(value));
	}
	return __builtin_popcountll(static_cast<unsigned long long>(value));
}

inline state_t invalid_state()
{
	return std::numeric_limits<state_t>::max();
}

} // namespace bgt

using bgt_state_t = bgt::state_t;

#endif
