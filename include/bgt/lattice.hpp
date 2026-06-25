#pragma once

#include "bgt/types.hpp"

#include <memory>
#include <span>
#include <vector>

namespace bgt::model
{
class Lattice;
}

namespace bgt
{

class DilutionTable
{
public:
	DilutionTable() = default;
	DilutionTable(int subjects, host_probability_t alpha, host_probability_t h);

	int subjects() const { return subjects_; }
	host_probability_t **rows() { return rows_.empty() ? nullptr : rows_.data(); }
	host_probability_t *const *rows() const { return rows_.empty() ? nullptr : rows_.data(); }
	host_probability_t at(int group_size, int positives) const;

private:
	int subjects_ = 0;
	std::vector<host_probability_t> values_;
	std::vector<host_probability_t *> rows_;
};

class Lattice
{
public:
	Lattice(LatticeType type, int subjects, std::span<const host_probability_t> prior);
	Lattice(LatticeType type, int subjects, int variants, std::span<const host_probability_t> prior);
	~Lattice();

	Lattice(const Lattice &) = delete;
	Lattice &operator=(const Lattice &) = delete;
	Lattice(Lattice &&) noexcept;
	Lattice &operator=(Lattice &&) noexcept;

	int subjects() const;
	int variants() const;
	LatticeType type() const;

	host_probability_t posterior_probability(state_t state) const;
	host_probability_t upset_probability_mass(state_t experiment) const;
	host_probability_t response_probability(state_t experiment, state_t response, state_t true_state, const DilutionTable *dilution = nullptr) const;
	state_t select_experiment(SelectorType selector = SelectorType::auto_select) const;
	std::vector<state_t> select_experiments(int k, SelectorType selector = SelectorType::op_bha) const;
	void update(state_t experiment, state_t response, const DilutionTable *dilution = nullptr);
	void update_classification(host_probability_t threshold_up, host_probability_t threshold_lo);
	state_t positive_classification_mask() const;
	state_t negative_classification_mask() const;
	bool is_classified() const;

	model::Lattice *native();
	const model::Lattice *native() const;

private:
	struct Impl;
	std::unique_ptr<Impl> impl_;
};

} // namespace bgt
