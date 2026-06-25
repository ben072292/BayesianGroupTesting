#pragma once

#include "bgt/types.hpp"

enum lattice_copy_op
{
	NO_COPY_PROB_DIST,
	SHALLOW_COPY_PROB_DIST,
	DEEP_COPY_PROB_DIST
};
using lattice_copy_op_t = lattice_copy_op;

enum lattice_parallelism
{
	DIST_MODEL,
	REPL_MODEL = 2
};
using lattice_parallelism_t = lattice_parallelism;

enum dilution
{
	NON_DILUTION,
	DILUTION
};
using dilution_t = dilution;

enum lattice_type
{
	DIST_NON_DILUTION = 1,
	DIST_DILUTION = 2,
	REPL_NON_DILUTION = 3,
	REPL_DILUTION = 4
};
using lattice_type_t = lattice_type;

inline lattice_type_t to_internal_lattice_type(bgt::LatticeType type)
{
	switch (type)
	{
	case bgt::LatticeType::distributed_non_dilution:
		return DIST_NON_DILUTION;
	case bgt::LatticeType::distributed_dilution:
		return DIST_DILUTION;
	case bgt::LatticeType::replicated_non_dilution:
		return REPL_NON_DILUTION;
	case bgt::LatticeType::replicated_dilution:
		return REPL_DILUTION;
	}
	return REPL_NON_DILUTION;
}

inline bgt::LatticeType to_public_lattice_type(lattice_type_t type)
{
	switch (type)
	{
	case DIST_NON_DILUTION:
		return bgt::LatticeType::distributed_non_dilution;
	case DIST_DILUTION:
		return bgt::LatticeType::distributed_dilution;
	case REPL_NON_DILUTION:
		return bgt::LatticeType::replicated_non_dilution;
	case REPL_DILUTION:
		return bgt::LatticeType::replicated_dilution;
	}
	return bgt::LatticeType::replicated_non_dilution;
}
