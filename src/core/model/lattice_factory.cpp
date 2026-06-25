#include "bgt/detail/model/lattice.hpp"

#include <stdexcept>

namespace bgt::model
{

std::unique_ptr<Lattice> create_lattice(lattice_type_t type, int subjs, int variants, bgt::host_probability_t *pi0)
{
	switch (type)
	{
	case DIST_NON_DILUTION:
		return std::make_unique<DistributedNonDilutionLattice>(subjs, variants, pi0);
	case DIST_DILUTION:
		return std::make_unique<DistributedDilutionLattice>(subjs, variants, pi0);
	case REPL_NON_DILUTION:
		return std::make_unique<ReplicatedNonDilutionLattice>(subjs, variants, pi0);
	case REPL_DILUTION:
		return std::make_unique<ReplicatedDilutionLattice>(subjs, variants, pi0);
	default:
		throw bgt::Error(bgt::Status::invalid_argument("nonexisting lattice type", __FILE__, __LINE__));
	}
}

std::unique_ptr<Lattice> clone_lattice(lattice_type_t type, lattice_copy_op_t op, const Lattice &lattice)
{
	switch (type)
	{
	case DIST_NON_DILUTION:
		return std::make_unique<DistributedNonDilutionLattice>(lattice, op);
	case DIST_DILUTION:
		return std::make_unique<DistributedDilutionLattice>(lattice, op);
	case REPL_NON_DILUTION:
		return std::make_unique<ReplicatedNonDilutionLattice>(lattice, op);
	case REPL_DILUTION:
		return std::make_unique<ReplicatedDilutionLattice>(lattice, op);
	default:
		throw bgt::Error(bgt::Status::invalid_argument("nonexisting lattice type", __FILE__, __LINE__));
	}
}

void update_lattice_probs(Lattice *lattice, bgt_state_t experiment, bgt_state_t responses, bgt::host_probability_t **dilution)
{
	lattice->update_probs_in_place(experiment, responses, dilution);
}

bgt_state_t lattice_bbpa(Lattice *lattice)
{
	return lattice->BBPA(1.0 / (1 << lattice->variants()));
}

void lattice_mpi_initialize(lattice_type_t type, int subjs, int variants)
{
	// std::cout << type << std::endl;
	switch (type)
	{
	case DIST_NON_DILUTION:
		DistributedLattice::lattice_mpi_initialize(subjs, variants);
		break;
		// std::cout << type << std::endl;
	case DIST_DILUTION:
		DistributedLattice::lattice_mpi_initialize(subjs, variants);
		break;
	case REPL_NON_DILUTION:
		Lattice::lattice_mpi_initialize();
		break;
	case REPL_DILUTION:
		Lattice::lattice_mpi_initialize();
		break;
	default:
		throw bgt::Error(bgt::Status::invalid_argument("nonexisting lattice type", __FILE__, __LINE__));
	}
}

void lattice_mpi_finalize(lattice_type_t type)
{
	switch (type)
	{
	case DIST_NON_DILUTION:
		DistributedLattice::lattice_mpi_finalize();
		break;
	case DIST_DILUTION:
		DistributedLattice::lattice_mpi_finalize();
		break;
	case REPL_NON_DILUTION:
		Lattice::lattice_mpi_finalize();
		break;
	case REPL_DILUTION:
		Lattice::lattice_mpi_finalize();
		break;
	default:
		throw bgt::Error(bgt::Status::invalid_argument("nonexisting lattice type", __FILE__, __LINE__));
	}
}

} // namespace bgt::model
