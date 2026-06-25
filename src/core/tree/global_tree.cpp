#include "bgt/detail/tree/tree.hpp"

namespace bgt::tree
{
using model::Lattice;


int GlobalTree::rank = -1;
int GlobalTree::world_size = 0;
MPI_Datatype GlobalTree::stats_mpi_type;
MPI_Op GlobalTree::stats_mpi_op;

GlobalTree::GlobalTree(std::unique_ptr<Lattice> lattice, bgt_state_t ex, bgt_state_t res, int k, int curr_stage)
    : GlobalTree(std::move(lattice), ex, res, curr_stage)
{
    if (!_lattice->is_classified() && curr_stage < _search_depth)
    {
        children(1 << variants());

        bgt_state_t BBPA = select_experiment();
        bgt_state_t ex = true_ex(BBPA);

        for (int re = 0; re < (1 << variants()); re++)
        {
            auto p = _lattice->clone(SHALLOW_COPY_PROB_DIST);

            if (re == (1 << variants()) - 1)
            {
                p->set_posterior(_lattice->take_posterior());
                p->update_probs_in_place(BBPA, re, _dilution);
            }
            else
            {
                p->update_probs(BBPA, re, _dilution);
            }

            if (p->update_metadata_with_shrinking(_thres_up, _thres_lo))
            {
                p = p->to_local();
            }
            set_child(re, std::make_unique<GlobalTree>(std::move(p), ex, re, k, _curr_stage + 1));
        }
    }
    else
    { // Clean in advance to save memory
        destroy_posterior_probs();
    }
}

GlobalTree::GlobalTree(const Tree &other, bool deep) : LocalTree(other, false)
{
    if (deep && other.has_children())
    {
        children(1 << variants());
        for (int i = 0; i < (1 << variants()); i++)
        {
            if (other.child(i) != nullptr)
                set_child(i, other.child(i)->clone(deep));
        }
    }
}

std::unique_ptr<Tree> GlobalTree::clone(bool deep) const
{
    return std::make_unique<GlobalTree>(*this, deep);
}

void GlobalTree::initialize_mpi(int, int k)
{
    // Get the number of processes
    BGT_MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &world_size));
    // Get the rank of the process
    BGT_MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    TreeStatsAccumulator::create_mpi_type(&stats_mpi_type, _search_depth, k);
    BGT_MPI_CHECK(MPI_Type_commit(&stats_mpi_type));
    BGT_MPI_CHECK(MPI_Op_create((MPI_User_function *)&TreeStatsAccumulator::mpi_reduce, true, &stats_mpi_op));
}

void GlobalTree::finalize_mpi()
{
    // Free datatype
    BGT_MPI_CHECK(MPI_Type_free(&stats_mpi_type));
    // Free tree stat reduce op
    BGT_MPI_CHECK(MPI_Op_free(&stats_mpi_op));
}

} // namespace bgt::tree
