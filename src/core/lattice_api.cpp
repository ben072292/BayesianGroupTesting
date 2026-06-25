#include "bgt/bgt.hpp"
#include "bgt/detail/model/lattice.hpp"
#include "bgt/detail/runtime/simulation_helpers.hpp"

#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

using bgt::model::create_lattice;
using bgt::runtime_detail::resolve_selector;
using bgt::runtime_detail::selector_is_single_test;

namespace
{
bgt_state_t reverse_bits(bgt_state_t value, int width)
{
	bgt_state_t ret = 0;
	for (int i = 0; i < width; i++)
	{
		ret <<= 1;
		ret |= (value >> i) & 1;
	}
	return ret;
}

bool is_up_set(bgt_state_t a, bgt_state_t b)
{
	return (a & b) == b;
}

int set_group_index(bgt_state_t state, const std::vector<bgt_state_t> &experiments)
{
	int ret = 0;
	for (std::size_t i = 0; i < experiments.size(); i++)
	{
		if (is_up_set(experiments[i], state))
		{
			ret |= 1 << i;
		}
	}
	return ret;
}

} // namespace

namespace bgt
{

struct Lattice::Impl
{
	std::unique_ptr<model::Lattice> native;
	int subjects = 0;
	int variants = 1;
	LatticeType type = LatticeType::replicated_non_dilution;
	std::vector<host_probability_t> prior;
};


Lattice::Lattice(LatticeType type, int subjects, std::span<const host_probability_t> prior)
	: Lattice(type, subjects, 1, prior)
{
}

Lattice::Lattice(LatticeType type, int subjects, int variants, std::span<const host_probability_t> prior) : impl_(std::make_unique<Impl>())
{
	if (subjects <= 0 || variants <= 0)
		throw std::invalid_argument("subjects and variants must be positive.");
	if (static_cast<int>(prior.size()) != subjects * variants)
		throw std::invalid_argument("prior length must equal subjects * variants.");
	impl_->subjects = subjects;
	impl_->variants = variants;
	impl_->type = type;
	impl_->prior.assign(prior.begin(), prior.end());
	impl_->native = create_lattice(to_internal_lattice_type(type), subjects, variants, impl_->prior.data());
}

Lattice::~Lattice() = default;
Lattice::Lattice(Lattice &&) noexcept = default;
Lattice &Lattice::operator=(Lattice &&) noexcept = default;

int Lattice::subjects() const { return impl_->subjects; }
int Lattice::variants() const { return impl_->variants; }
LatticeType Lattice::type() const { return impl_->type; }
model::Lattice *Lattice::native() { return impl_->native.get(); }
const model::Lattice *Lattice::native() const { return impl_->native.get(); }

host_probability_t Lattice::posterior_probability(state_t state) const
{
	const state_t internal_state = impl_->variants == 1 ? reverse_bits(state, impl_->subjects) : state;
	return impl_->native->posterior_prob(internal_state);
}

host_probability_t Lattice::upset_probability_mass(state_t experiment) const
{
	const state_t internal_experiment = impl_->variants == 1 ? reverse_bits(experiment, impl_->subjects) : experiment;
	return impl_->native->get_prob_mass(internal_experiment);
}

host_probability_t Lattice::response_probability(state_t experiment, state_t response, state_t true_state, const DilutionTable *dilution) const
{
	const state_t internal_experiment = impl_->variants == 1 ? reverse_bits(experiment, impl_->subjects) : experiment;
	const state_t internal_state = impl_->variants == 1 ? reverse_bits(true_state, impl_->subjects) : true_state;
	return impl_->native->response_prob(internal_experiment, response, internal_state, dilution == nullptr ? nullptr : const_cast<DilutionTable *>(dilution)->rows());
}

state_t Lattice::select_experiment(SelectorType selector) const
{
	if (!selector_is_single_test(selector))
		return 0;
	const state_t selected = impl_->native->select_experiment(selector);
	return impl_->variants == 1 ? reverse_bits(selected, impl_->subjects) : selected;
}

std::vector<state_t> Lattice::select_experiments(int k, SelectorType selector) const
{
	if (k < 0)
		throw std::invalid_argument("k must be non-negative.");
	std::vector<state_t> selected;
	selected.reserve(k);
	if (k == 0)
		return selected;
	const SelectorType resolved_selector = resolve_selector(selector, impl_->variants);
	if (resolved_selector == SelectorType::op_bbpa)
	{
		if (k != 1)
			throw std::invalid_argument("Op-BBPA currently selects one experiment at a time.");
		selected.push_back(select_experiment(selector));
		return selected;
	}
	if (resolved_selector != SelectorType::op_bha || impl_->variants != 1)
		throw std::invalid_argument("Op-BHA multi-experiment selection is only supported for binary lattices.");
	for (int step = 0; step < k; step++)
	{
		if (selected.empty())
		{
			selected.push_back(select_experiment(SelectorType::op_bha));
			continue;
		}

		const int total_states = state_count(impl_->subjects);
		score_t min_score = std::numeric_limits<score_t>::max();
		state_t min_experiment = 0;
		for (int candidate = 1; candidate < total_states; candidate++)
		{
			std::vector<state_t> all_experiments(selected);
			all_experiments.push_back(static_cast<state_t>(candidate));
			std::vector<accumulator_t> probability_mass(1 << all_experiments.size(), 0.0);
			for (int state = 0; state < total_states; state++)
			{
				probability_mass[set_group_index(static_cast<state_t>(state), all_experiments)] +=
					posterior_probability(static_cast<state_t>(state));
			}
			const score_t target_mass = score_t{1.0} / static_cast<score_t>(1 << all_experiments.size());
			score_t score = 0.0;
			for (accumulator_t mass : probability_mass)
				score += std::abs(mass - target_mass);
			if (score < min_score)
			{
				min_score = score;
				min_experiment = static_cast<state_t>(candidate);
			}
		}
		selected.push_back(min_experiment);
	}
	return selected;
}

void Lattice::update(state_t experiment, state_t response, const DilutionTable *dilution)
{
	const state_t internal_experiment = impl_->variants == 1 ? reverse_bits(experiment, impl_->subjects) : experiment;
	impl_->native->update_probs_in_place(internal_experiment, response, dilution == nullptr ? nullptr : const_cast<DilutionTable *>(dilution)->rows());
}

void Lattice::update_classification(host_probability_t threshold_up, host_probability_t threshold_lo)
{
	impl_->native->update_metadata(threshold_up, threshold_lo);
}

state_t Lattice::positive_classification_mask() const
{
	const state_t mask = impl_->native->pos_clas_atoms();
	return impl_->variants == 1 ? reverse_bits(mask, impl_->subjects) : mask;
}

state_t Lattice::negative_classification_mask() const
{
	const state_t mask = impl_->native->neg_clas_atoms();
	return impl_->variants == 1 ? reverse_bits(mask, impl_->subjects) : mask;
}

bool Lattice::is_classified() const
{
	return impl_->native->is_classified();
}

} // namespace bgt
