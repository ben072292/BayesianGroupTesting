#pragma once

#include "bgt/detail/model/dilution.hpp"
#include "bgt/detail/model/posterior_buffer.hpp"
#include "bgt/detail/model/shrink.hpp"

#include <memory>
#include <string>
#include <vector>

namespace bgt::model
{

// Lattice parallelism
class Lattice
{

protected:
	static int rank, world_size, _orig_subjs, _variants;
	static bgt::host_probability_t *_pi0;
	int _curr_subjs = 0;			 // counter
	bgt_state_t _pos_clas_atoms = 0; // BE (binary encoding)
	bgt_state_t _neg_clas_atoms = 0; // BE
	bgt_state_t _clas_subjs = 0;	 // BE, variables used for lattice shrinking
	PosteriorBuffer _posterior;
	static int partition_count(int total, int partition_rank);
	static int partition_start(int total, int partition_rank);
	static int partition_rank(int total, int item);

public:
	Lattice() {}
	Lattice(int subjs, int variants, bgt::host_probability_t *pi0);
	Lattice(const Lattice &other, lattice_copy_op_t op);
	virtual ~Lattice();
	virtual std::unique_ptr<Lattice> create(int subjs, int variants, bgt::host_probability_t *pi0) const = 0;
	virtual std::unique_ptr<Lattice> clone(lattice_copy_op_t op) const = 0;
	inline int curr_subjs() const { return _curr_subjs; }
	inline static int variants() { return _variants; };
	inline static bgt::host_probability_t *pi0() { return _pi0; }
	inline static void pi0(bgt::host_probability_t *pi0) { _pi0 = pi0; }
	inline bgt_state_t pos_clas_atoms() const { return _pos_clas_atoms; }
	inline bgt_state_t neg_clas_atoms() const { return _neg_clas_atoms; }
	inline int curr_atoms() const { return _curr_subjs * _variants; }
	inline int orig_atoms() const { return _orig_subjs * _variants; }
	inline static int orig_subjs() { return _orig_subjs; }
	inline bgt_state_t clas_subjs() const { return _clas_subjs; }
	inline int total_states() const { return bgt::state_count(_curr_subjs * _variants); }
	inline bgt::posterior_t *posterior_probs() const { return _posterior.data(); };
	virtual int posterior_prob_count() const { return total_states(); }
	virtual bgt::host_probability_t posterior_prob(bgt_state_t state) const;
	void allocate_posterior_probs(int count);
	void own_posterior_probs(bgt::posterior_t *post_probs, int count);
	inline void own_posterior_probs(bgt::posterior_t *post_probs) { own_posterior_probs(post_probs, posterior_prob_count()); }
	void borrow_posterior_probs(bgt::posterior_t *post_probs, int count);
	inline void borrow_posterior_probs(bgt::posterior_t *post_probs) { borrow_posterior_probs(post_probs, posterior_prob_count()); }
	PosteriorBuffer take_posterior();
	void set_posterior(PosteriorBuffer buffer);
	void free_posterior_probs();
	inline void posterior_probs(bgt::posterior_t *post_probs) { own_posterior_probs(post_probs); }
	inline bool is_classified() const { return bgt::state_popcount(_pos_clas_atoms | _neg_clas_atoms) == orig_atoms(); }
	bgt_state_t *get_up_set(bgt_state_t state, bgt_state_t *ret) const;
	void generate_power_set_adder(bgt_state_t *add_index, int index_len, bgt_state_t state, bgt_state_t *ret) const;
	virtual void prior_probs(bgt::host_probability_t *pi0);
	bgt::host_probability_t prior_prob(bgt_state_t state, bgt::host_probability_t *pi0) const;
	virtual void update_probs(bgt_state_t experiment, bgt_state_t response, bgt::host_probability_t **dilution);
	virtual void update_probs_in_place(bgt_state_t experiment, bgt_state_t response, bgt::host_probability_t **dilution);
	virtual bgt::host_probability_t response_prob(bgt_state_t experiment, bgt_state_t response, bgt_state_t true_state, bgt::host_probability_t **dilution) const = 0;
	virtual std::string type() const = 0;
	virtual void update_metadata(bgt::host_probability_t thres_up, bgt::host_probability_t thres_lo);
	virtual bool update_metadata_with_shrinking(
		bgt::host_probability_t thres_up,
		bgt::host_probability_t thres_lo);
	virtual void apply_shrink_plan(const ShrinkPlan &plan);
	virtual bgt::accumulator_t get_prob_mass(bgt_state_t state) const;
	virtual bgt::accumulator_t get_atom_prob_mass(bgt_state_t atom) const;
	virtual bgt_state_t select_experiment(bgt::SelectorType selector) const;
	virtual bgt_state_t select_experiment_serial(bgt::SelectorType selector) const;
	virtual bgt_state_t op_bha() const;
	virtual bgt_state_t op_bha_serial() const;
#ifdef ENABLE_OMP
	virtual bgt_state_t op_bha_omp() const;
#endif
	virtual bgt_state_t BBPA(bgt::accumulator_t prob) const;
	virtual lattice_parallelism_t parallelism() const { return REPL_MODEL; }
	virtual dilution_t dilution() const { return NON_DILUTION; }
	/**
	 * @brief convert from distributed model to local model
	 *
	 * @return locally owned lattice model
	 */
	virtual std::unique_ptr<Lattice> to_local();
	virtual bgt_state_t BBPA_serial(bgt::accumulator_t prob) const;
#ifdef ENABLE_OMP
	virtual bgt_state_t BBPA_omp(bgt::accumulator_t prob) const;
	virtual bgt_state_t BBPA_mpi_omp(bgt::accumulator_t prob) const;
#endif
	virtual bgt_state_t BBPA_mpi(bgt::accumulator_t prob) const;
#ifdef ENABLE_SIMD
	virtual bgt_state_t BBPA_mpi_simd(bgt::accumulator_t) const { throw std::logic_error("SIMD BBPA is only available for distributed lattices."); }
#endif
#if defined(ENABLE_OMP) && defined(ENABLE_SIMD)
	virtual bgt_state_t BBPA_mpi_omp_simd(bgt::accumulator_t) const { throw std::logic_error("SIMD/OpenMP BBPA is only available for distributed lattices."); }
#endif
	static void lattice_mpi_initialize();
	static void lattice_mpi_finalize();
};

class ReplicatedDilutionLattice : public virtual Lattice
{
public:
	ReplicatedDilutionLattice() {} // default constructor

	ReplicatedDilutionLattice(int subjs, int variants, bgt::host_probability_t *pi0) : Lattice(subjs, variants, pi0) {}

	ReplicatedDilutionLattice(Lattice const &other, lattice_copy_op_t op) : Lattice(other, op) {}

	std::unique_ptr<Lattice> create(int subjs, int variants, bgt::host_probability_t *pi0) const override { return std::make_unique<ReplicatedDilutionLattice>(subjs, variants, pi0); }

	std::unique_ptr<Lattice> clone(lattice_copy_op op) const override { return std::make_unique<ReplicatedDilutionLattice>(*this, op); }

	bgt::host_probability_t response_prob(bgt_state_t experiment, bgt_state_t response, bgt_state_t true_state, bgt::host_probability_t **dilution) const override;

	dilution_t dilution() const override { return DILUTION; }

	std::string type() const override { return "Replicated-Dilution"; }
};

class ReplicatedNonDilutionLattice : public virtual Lattice
{
public:
	ReplicatedNonDilutionLattice() {}; // default constructor

	ReplicatedNonDilutionLattice(int subjs, int variants, bgt::host_probability_t *pi0) : Lattice(subjs, variants, pi0) {}

	ReplicatedNonDilutionLattice(Lattice const &other, lattice_copy_op_t op) : Lattice(other, op) {}

	std::unique_ptr<Lattice> create(int subjs, int variants, bgt::host_probability_t *pi0) const override { return std::make_unique<ReplicatedNonDilutionLattice>(subjs, variants, pi0); }

	std::unique_ptr<Lattice> clone(lattice_copy_op_t op) const override { return std::make_unique<ReplicatedNonDilutionLattice>(*this, op); }

	bgt::host_probability_t response_prob(bgt_state_t experiment, bgt_state_t response, bgt_state_t true_state, bgt::host_probability_t **dilution) const override;

	std::string type() const override { return "Replicated-NoDilution"; }
};

/*
 * READ BEFORE USE:
 * In this version of lattice model, state is represented using A0B0A1B1...
 * The input of prior should also follow this pattern.
 * Model parallelism implementation for lattice model
 */
class DistributedLattice : public virtual Lattice
{

protected:
	static bgt::posterior_t *temp_post_prob_holder;
	static bgt::accumulator_t *partition_mass;
	static MPI_Win win;
	inline int total_states_per_rank() const { return partition_count(total_states(), rank); }
	inline int state_to_offset(bgt_state_t state) const { return static_cast<int>(state) - partition_start(total_states(), state_to_rank(state)); }
	inline int state_to_rank(bgt_state_t state) const { return partition_rank(total_states(), static_cast<int>(state)); }
	inline bgt_state_t offset_to_state(int offset) const { return static_cast<bgt_state_t>(partition_start(total_states(), rank) + offset); }

public:
	DistributedLattice() {} // default constructor
	DistributedLattice(int subjs, int variants, bgt::host_probability_t *pi0);
	DistributedLattice(const Lattice &other, lattice_copy_op_t op) : Lattice(other, op) {};
	int posterior_prob_count() const override { return total_states_per_rank(); }
	bgt::host_probability_t posterior_prob(bgt_state_t state) const override;
	void prior_probs(bgt::host_probability_t *pi0) override;
	void update_probs(bgt_state_t experiment, bgt_state_t response, bgt::host_probability_t **dilution) override;
	void update_probs_in_place(bgt_state_t experiment, bgt_state_t response, bgt::host_probability_t **dilution) override;
	void update_metadata(bgt::host_probability_t thres_up, bgt::host_probability_t thres_lo) override;
	bool update_metadata_with_shrinking(
		bgt::host_probability_t thres_up,
		bgt::host_probability_t thres_lo) override;
	void apply_shrink_plan(const ShrinkPlan &plan) override;
	bgt::accumulator_t get_prob_mass(bgt_state_t) const override { throw std::logic_error("distributed lattice probability-mass queries require an explicit reduction path."); }
	bgt::accumulator_t get_atom_prob_mass(bgt_state_t atom) const override;
	bgt_state_t BBPA_serial(bgt::accumulator_t prob) const override { throw std::logic_error("Serial BBPA algorithm is not supported in distributed model."); }
#ifdef ENABLE_OMP
	bgt_state_t BBPA_omp(bgt::accumulator_t prob) const override { throw std::logic_error("OMP BBPA algorithm is not supported in distributed model"); }
	bgt_state_t BBPA_mpi_omp(bgt::accumulator_t prob) const override;
#endif
	bgt_state_t op_bha() const override;
	bgt_state_t op_bha_mpi() const;
	bgt_state_t BBPA_mpi(bgt::accumulator_t prob) const override;
#ifdef ENABLE_SIMD
	bgt_state_t BBPA_mpi_simd(bgt::accumulator_t prob) const override;
#endif
#if defined(ENABLE_OMP) && defined(ENABLE_SIMD)
	bgt_state_t BBPA_mpi_omp_simd(bgt::accumulator_t prob) const override;
#endif
	bgt_state_t BBPA(bgt::accumulator_t prob) const override;
	virtual lattice_parallelism_t parallelism() const override { return DIST_MODEL; }
	virtual dilution_t dilution() const override { return NON_DILUTION; }
	static void lattice_mpi_initialize(int subjs, int variants);
	static void lattice_mpi_finalize();
};

class DistributedDilutionLattice : public DistributedLattice, public ReplicatedDilutionLattice
{
public:
	DistributedDilutionLattice(int subjs, int variants, bgt::host_probability_t *pi0) : DistributedLattice(subjs, variants, pi0) {}

	DistributedDilutionLattice(Lattice const &other, lattice_copy_op_t copy_op) : Lattice(other, copy_op) {}

	std::unique_ptr<Lattice> create(int subjs, int variants, bgt::host_probability_t *pi0) const override { return std::make_unique<DistributedDilutionLattice>(subjs, variants, pi0); }

	std::unique_ptr<Lattice> clone(lattice_copy_op_t op) const override { return std::make_unique<DistributedDilutionLattice>(*this, op); };

	std::unique_ptr<Lattice> to_local() override
	{
		auto p = std::make_unique<ReplicatedDilutionLattice>(*this, NO_COPY_PROB_DIST);
		p->set_posterior(take_posterior());
		return p;
	}

	dilution_t dilution() const override { return DILUTION; }

	std::string type() const override { return "Distributed-Dilution"; }
};

class DistributedNonDilutionLattice : public DistributedLattice, public ReplicatedNonDilutionLattice
{
public:
	DistributedNonDilutionLattice(int subjs, int variants, bgt::host_probability_t *pi0) : DistributedLattice(subjs, variants, pi0) {}

	DistributedNonDilutionLattice(Lattice const &other, lattice_copy_op_t op) : Lattice(other, op) {}

	std::unique_ptr<Lattice> create(int subjs, int variants, bgt::host_probability_t *pi0) const override { return std::make_unique<DistributedNonDilutionLattice>(subjs, variants, pi0); }

	std::unique_ptr<Lattice> clone(lattice_copy_op_t op) const override { return std::make_unique<DistributedNonDilutionLattice>(*this, op); };

	std::unique_ptr<Lattice> to_local() override
	{
		auto p = std::make_unique<ReplicatedNonDilutionLattice>(*this, NO_COPY_PROB_DIST);
		p->set_posterior(take_posterior());
		return p;
	}

	std::string type() const override { return "Distributed-NoDilution"; }
};

typedef struct BBPAResult
{
	bgt::accumulator_t min;
	bgt_state_t candidate;

	BBPAResult(bgt::accumulator_t val1 = bgt::accumulator_t{2.0}, bgt_state_t val2 = bgt::invalid_state())
	{
		min = val1;
		candidate = val2;
	}

	inline void reset() { min = bgt::accumulator_t{2.0}, candidate = bgt::invalid_state(); }

	inline static void create_mpi_type(MPI_Datatype *BBPAResult_type)
	{
		int lengths[2] = {1, 1};

		// Calculate displacements
		// In C, by default padding can be inserted between fields. MPI_Get_address will allow
		// to get the address of each struct field and calculate the corresponding displacement
		// relative to that struct base address. The displacements thus calculated will therefore
		// include padding if any.
		MPI_Aint displacements[2];
		struct BBPAResult dummy_best_result;
		MPI_Aint base_address;
		BGT_MPI_CHECK(MPI_Get_address(&dummy_best_result, &base_address));
		BGT_MPI_CHECK(MPI_Get_address(&dummy_best_result.min, &displacements[0]));
		BGT_MPI_CHECK(MPI_Get_address(&dummy_best_result.candidate, &displacements[1]));
		displacements[0] = MPI_Aint_diff(displacements[0], base_address);
		displacements[1] = MPI_Aint_diff(displacements[1], base_address);

		MPI_Datatype types[2] = {bgt_accumulator_mpi_type(), bgt_state_mpi_type()};
		BGT_MPI_CHECK(MPI_Type_create_struct(2, lengths, displacements, types, BBPAResult_type));
	}

	inline static void mpi_reduce(BBPAResult *in, BBPAResult *inout, int *len, MPI_Datatype *dptr)
	{
		if (in->min < inout->min || (in->min == inout->min && in->candidate < inout->candidate))
		{
			inout->min = in->min;
			inout->candidate = in->candidate;
		}
	}

	static void min_assign(BBPAResult &a, BBPAResult &b)
	{
		if (a.min > b.min || (a.min == b.min && b.candidate < a.candidate))
		{
			a.min = b.min;
			a.candidate = b.candidate;
		}
	}
} BBPAResult;

extern MPI_Datatype BBPAResult_type;
extern MPI_Op BBPA_op;

std::unique_ptr<Lattice> create_lattice(lattice_type_t type, int subjs, int variants, bgt::host_probability_t *pi0);
std::unique_ptr<Lattice> clone_lattice(lattice_type_t type, lattice_copy_op_t op, const Lattice &lattice);
void lattice_mpi_initialize(lattice_type_t type, int subjs, int variants);
void lattice_mpi_finalize(lattice_type_t type);


} // namespace bgt::model
