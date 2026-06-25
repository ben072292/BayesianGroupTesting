#include "bgt/runtime.hpp"

namespace bgt
{

bool KernelRegistry::contains(const KernelSpec &spec) const
{
	return entries_.find(spec) != entries_.end();
}

void KernelRegistry::insert(const KernelSpec &spec)
{
	entries_[spec] = true;
}

void KernelRegistry::clear()
{
	entries_.clear();
}

std::size_t KernelRegistry::size() const
{
	return entries_.size();
}

} // namespace bgt
