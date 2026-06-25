#pragma once

#include "bgt/types.hpp"

namespace bgt::model
{

bgt::host_probability_t **generate_dilution(int n, bgt::host_probability_t alpha, bgt::host_probability_t h);
void free_dilution(bgt::host_probability_t **dilution, int subjects);

} // namespace bgt::model
