#include "bgt/detail/tree/tree.hpp"
#include <algorithm>

namespace bgt::tree
{
using model::Lattice;

static bool compare_indices(const std::pair<bgt::posterior_t, int> &a, const std::pair<bgt::posterior_t, int> &b)
{
    return a.first < b.first;
}

std::vector<int> trim_true_states(const bgt::posterior_t *values, int n, bgt::host_probability_t percent)
{
    std::vector<std::pair<bgt::posterior_t, int>> indexed_values(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i)
    {
        indexed_values[static_cast<std::size_t>(i)] = std::make_pair(values[i], i);
    }
    std::sort(indexed_values.begin(), indexed_values.end(), compare_indices);

    const bgt::accumulator_t threshold = static_cast<bgt::accumulator_t>(percent / 100.0);
    std::vector<int> remaining_indices;
    remaining_indices.reserve(static_cast<std::size_t>(n));
    bgt::accumulator_t currentSum = 0.0;

    for (int i = 0; i < n; ++i)
    {
        const auto &entry = indexed_values[static_cast<std::size_t>(i)];
        if (currentSum + entry.first > threshold)
        {
            remaining_indices.push_back(entry.second);
        }
        else
        {
            currentSum += entry.first;
        }
    }

    return remaining_indices;
}

} // namespace bgt::tree
