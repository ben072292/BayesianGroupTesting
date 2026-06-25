#include "bgt/detail/model/lattice.hpp"

namespace bgt::model
{

const static bgt::host_probability_t negative_response = 0.99;

bgt::host_probability_t ReplicatedNonDilutionLattice::response_prob(bgt_state_t experiment, bgt_state_t response, bgt_state_t true_state, bgt::host_probability_t **__restrict__ dilution) const
{
	bgt::host_probability_t ret = 1.0;
	for (int variant = 0; variant < _variants; variant++)
		ret *= ((experiment & (true_state >> (variant * _curr_subjs))) == experiment) ? ((response & (1 << variant)) == 0 ? (1.0 - negative_response) : negative_response) : ((response & (1 << variant)) == 0 ? negative_response : (1.0 - negative_response));
	return ret;
}


bgt::host_probability_t ReplicatedDilutionLattice::response_prob(bgt_state_t experiment, bgt_state_t response, bgt_state_t true_state, bgt::host_probability_t **__restrict__ dilution) const
{
    bgt::host_probability_t ret = 1.0;
    int experimentLength = bgt::state_popcount(experiment);
    for (int variant = 0; variant < _variants; variant++)
    {
        ret *= (response & (1 << variant)) != 0
                   ? dilution[experimentLength - 1][experimentLength - bgt::state_popcount(experiment & (true_state >> (variant * _curr_subjs)))]
                   : 1.0 - dilution[experimentLength - 1][experimentLength - bgt::state_popcount(experiment & (true_state >> (variant * _curr_subjs)))];
    }
    return ret;
}

} // namespace bgt::model
