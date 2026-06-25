#include "bgt/detail/tree/tree.hpp"

namespace bgt::tree
{
using model::Lattice;


namespace
{
int partition_count(int total, int rank, int world_size)
{
    const int base = total / world_size;
    const int remainder = total % world_size;
    return base + (rank < remainder ? 1 : 0);
}

int partition_start(int total, int rank, int world_size)
{
    const int base = total / world_size;
    const int remainder = total % world_size;
    return rank * base + (rank < remainder ? rank : remainder);
}
}

std::vector<FusionTree *> FusionTree::sequence_tracer;

FusionTree::FusionTree(std::unique_ptr<Lattice> lattice, bgt_state_t ex, bgt_state_t res, int k, int curr_stage, bgt::branch_probability_t prun_thres_sum, bgt::branch_probability_t curr_prun_thres_sum, bgt::branch_probability_t prun_thres)
    : FusionTree(std::move(lattice), ex, res, curr_stage)
{
    sequence_tracer[curr_stage] = this;

    if (!_lattice->is_classified() && curr_stage < _search_depth)
    {
        children(1 << variants());

        bgt_state_t BBPA = select_experiment();
        bgt_state_t ex = true_ex(BBPA);

        for (int re = 0; re < (1 << variants()); re++)
        {
            // Fusion tree pruning process
            bgt::branch_probability_t fusion_tree_branch_prob = fusion_branch_prob(ex, re);
            BGT_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &fusion_tree_branch_prob, 1, bgt_accumulator_mpi_type(), MPI_SUM, MPI_COMM_WORLD));

            if (curr_prun_thres_sum < prun_thres_sum &&
                fusion_tree_branch_prob < prun_thres &&
                _lattice->curr_atoms() >= _lattice->orig_atoms())
            { // Can be pruned through fusion tree
                curr_prun_thres_sum += fusion_tree_branch_prob;

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

                p->update_metadata(_thres_up, _thres_lo); // No need to shrink as only metadata matters
                p->free_posterior_probs();

                set_child(re, std::make_unique<FusionTree>(std::move(p), ex, re, curr_stage));
            }
            else
            { // Cannot be pruned
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

                set_child(re, std::make_unique<FusionTree>(std::move(p), ex, re, k, _curr_stage + 1, prun_thres_sum, curr_prun_thres_sum, prun_thres));
            }
        }
    }
    else
    { // Clean in advance to save memory
        destroy_posterior_probs();
    }
}

FusionTree::FusionTree(const Tree &other, bool deep) : GlobalTree(other, false)
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

std::unique_ptr<Tree> FusionTree::clone(bool deep) const
{
    return std::make_unique<FusionTree>(*this, deep);
}

bgt::branch_probability_t FusionTree::fusion_branch_prob(int ex, int res)
{
    bgt::branch_probability_t ret = 0.0;
    const int total_states = 1 << _lattice->orig_atoms();
    const int start_state = partition_start(total_states, rank, world_size);
    const int stop_state = start_state + partition_count(total_states, rank, world_size);
    for (int i = start_state; i < stop_state; i++)
    {
        const bgt::host_probability_t coef = _lattice->prior_prob(i, Lattice::pi0());
        bgt::branch_probability_t temp_branch_prob = 1.0;
        for (int j = 1; j <= _curr_stage; j++)
        {
            temp_branch_prob *= _lattice->response_prob(sequence_tracer[j]->ex(), sequence_tracer[j]->ex_res(), i, _dilution);
        }
        temp_branch_prob *= _lattice->response_prob(ex, res, i, _dilution);
        ret += temp_branch_prob * coef;
    }
    return ret;
}

void FusionTree::initialize_mpi(int subjs, int k)
{
    GlobalTree::initialize_mpi(subjs, k);
    sequence_tracer.assign(static_cast<size_t>(_search_depth + 1), nullptr);
}

void FusionTree::finalize_mpi()
{
    GlobalTree::finalize_mpi();
    sequence_tracer.clear();
}

} // namespace bgt::tree
