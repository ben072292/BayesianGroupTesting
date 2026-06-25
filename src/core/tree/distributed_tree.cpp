#include "bgt/detail/tree/tree.hpp"

namespace bgt::tree
{
using model::Lattice;


int DistributedTree::rank = -1;
int DistributedTree::world_size = 0;
std::vector<DistributedTree *> DistributedTree::backtrack;
MPI_Datatype DistributedTree::stats_mpi_type;
MPI_Op DistributedTree::stats_mpi_op;

DistributedTree::DistributedTree(std::unique_ptr<Lattice> lattice, bgt_state_t selected_experiment, bgt_state_t ex, bgt_state_t res, int curr_stage, bgt::branch_probability_t prob)
    : Tree(std::move(lattice), ex, res, curr_stage, prob)
{
    _selected_experiment = selected_experiment;
}

DistributedTree::DistributedTree(std::unique_ptr<Lattice> lattice, bgt_state_t selected_experiment, bgt_state_t ex, bgt_state_t res, int k, int curr_stage, int expansion_depth)
    : DistributedTree(std::move(lattice), selected_experiment, ex, res, curr_stage, 0.0)
{
    if (!_lattice->is_classified() && curr_stage < expansion_depth)
    {
        children(1 << variants());
        bgt_state_t BBPA = select_experiment();
        bgt_state_t ex = true_ex(BBPA);                             // full-sized experiment should be generated before posterior probability distribution is updated, because unupdated clas_subj_ should be used to calculate the correct value
        for (int re = 0; re < (1 << variants()); re++)
        {
            auto p = _lattice->clone(SHALLOW_COPY_PROB_DIST);
            if (re != (1 << variants()) - 1 || curr_stage == 0)
            {
                p->update_probs(BBPA, re, _dilution);
            }
            else
            {
                // Transfer the parent posterior into the final response branch so
                // the tree does not keep two owned buffers for the same state.
                p->set_posterior(_lattice->take_posterior());
                p->update_probs_in_place(BBPA, re, _dilution);
            }
            if (p->update_metadata_with_shrinking(_thres_up, _thres_lo
                                                  ))
                p = p->to_local();
            set_child(re, std::make_unique<DistributedTree>(std::move(p), BBPA, ex, re, k, _curr_stage + 1, expansion_depth));
        }
    }
    else
    {
        // This node is terminal for the configured tree, so posterior storage is
        // released before later true-state evaluation.
        destroy_posterior_probs();
    }
}

DistributedTree::DistributedTree(const Tree &other, bool deep) : Tree(other, false)
{
    _selected_experiment = dynamic_cast<const DistributedTree &>(other)._selected_experiment;
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

std::unique_ptr<Tree> DistributedTree::clone(bool deep) const
{
    return std::make_unique<DistributedTree>(*this, deep);
}

void DistributedTree::eval(Tree *node, Lattice *orig_lattice, bgt_state_t true_state)
{
    if (!node->lattice()->is_classified() && node->curr_stage() < _search_depth && node->branch_prob() > _thres_branch)
    {
        if (!node->has_children()) // no child is explored, allocate children and perform computation
        {
            node->children(1 << node->variants());
            bgt_state_t BBPA = node->select_experiment_serial();
            bgt_state_t ex = node->true_ex(BBPA);

            for (int re = 0; re < (1 << node->variants()); re++)
            {
                const bgt::branch_probability_t child_prob = static_cast<bgt::branch_probability_t>(
                    node->branch_prob() * orig_lattice->response_prob(ex, re, true_state, _dilution));
                auto p = node->lattice()->clone(SHALLOW_COPY_PROB_DIST);
                if (re == (1 << variants()) - 1 && node->curr_stage() != 0)
                {
                    // Move the parent posterior into the last child branch. Other
                    // branches must allocate because they are evaluated before the
                    // parent buffer can be consumed.
                    p->set_posterior(node->lattice()->take_posterior());
                    p->update_probs_in_place(BBPA, re, _dilution);
                }
                else
                    p->update_probs(BBPA, re, _dilution);
                if (p->update_metadata_with_shrinking(_thres_up, _thres_lo
                                                      ))
                    p = p->to_local();
                node->set_child(re, std::make_unique<DistributedTree>(std::move(p), BBPA, ex, re, node->curr_stage() + 1, child_prob));
                eval(node->child(re), orig_lattice, true_state);
            }
        }
        else // test selection has been performed, children is (partially) initialized
        {
            // find BBPA and ex
            bgt_state_t BBPA = bgt::invalid_state(), ex = bgt::invalid_state();
            for (int re = 0; re < (1 << node->variants()); re++)
            {
                if (node->child(re) != nullptr)
                {
                    BBPA = dynamic_cast<DistributedTree *>(node->child(re))->_selected_experiment;
                    ex = node->child(re)->ex();
                    break;
                }
            }
            if (BBPA == bgt::invalid_state())
            {
                BGT_LOG_FATAL(bgt::LogSubsystem::tree, "DistributedTree child metadata is inconsistent.");
                throw bgt::Error(bgt::Status::internal_error("distributed tree child metadata is inconsistent", __FILE__, __LINE__));
            }
            for (int re = 0; re < (1 << node->variants()); re++)
            {
                const bgt::branch_probability_t child_prob = static_cast<bgt::branch_probability_t>(
                    node->branch_prob() * orig_lattice->response_prob(ex, re, true_state, _dilution));
                if (node->child(re) != nullptr) // fill in only posterior probability and step in
                {
                    node->child(re)->branch_prob(child_prob);
                    auto p = node->lattice()->clone(SHALLOW_COPY_PROB_DIST); // apply calculated branch probability first
                    if (re == (1 << variants()) - 1 && node->curr_stage() != 0)          // because of early-stopping, children may be computed out-of-order
                    {
                        p->set_posterior(node->lattice()->take_posterior());
                        p->update_probs_in_place(BBPA, re, _dilution);
                    }
                    else
                        p->update_probs(BBPA, re, _dilution);
                    if (p->update_metadata_with_shrinking(_thres_up, _thres_lo
                                                          ))
                        p = p->to_local();
                    node->child(re)->lattice()->set_posterior(p->take_posterior());
                    eval(node->child(re), orig_lattice, true_state);
                }
            }
        }
    }
    else
    {
        // A pre-constructed global tree might already have cleared this pointer.
        if (node->lattice()->posterior_probs() != nullptr)
        {
            node->destroy_posterior_probs();
        }
    }
}

void DistributedTree::lazy_eval(Tree *node, Lattice *orig_lattice, bgt_state_t true_state)
{
    backtrack[node->curr_stage()] = dynamic_cast<DistributedTree *>(node);
    if (!node->lattice()->is_classified() && node->curr_stage() < _search_depth && node->branch_prob() > _thres_branch)
    {
        if (!node->has_children()) // no child is explored, allocate children and perform computation
        {
            if (node->lattice()->posterior_probs() == nullptr)
            {
                int start_stage = 0;
                for (int i = node->curr_stage(); i >= 0; i--)
                {
                    if (backtrack[i]->lattice()->posterior_probs() != nullptr)
                    {
                        start_stage = i;
                        break;
                    }
                }
                for (int i = start_stage; i < node->curr_stage(); i++) // recover posterior probs
                {
                    auto p = backtrack[i]->lattice()->clone(SHALLOW_COPY_PROB_DIST);
                    if (i != 0)
                    {
                        // Rebuild the missing posterior chain by moving the nearest
                        // retained ancestor buffer forward one edge at a time.
                        p->set_posterior(backtrack[i]->lattice()->take_posterior());
                        p->update_probs_in_place(backtrack[i + 1]->_selected_experiment, backtrack[i + 1]->_res, _dilution);
                    }
                    else
                    {
                        p->update_probs(backtrack[i + 1]->_selected_experiment, backtrack[i + 1]->_res, _dilution);
                    }
                    if (p->update_metadata_with_shrinking(_thres_up, _thres_lo
                                                          ))
                        p = p->to_local();
                    backtrack[i + 1]->lattice()->set_posterior(p->take_posterior());
                }
            }

            node->children(1 << node->variants());
            bgt_state_t BBPA = node->select_experiment_serial();
            bgt_state_t ex = node->true_ex(BBPA);

            for (int re = 0; re < (1 << node->variants()); re++)
            {
                const bgt::branch_probability_t child_prob = static_cast<bgt::branch_probability_t>(
                    node->branch_prob() * orig_lattice->response_prob(ex, re, true_state, _dilution));
                auto p = node->lattice()->clone(SHALLOW_COPY_PROB_DIST);
                if (re == (1 << variants()) - 1 && node->curr_stage() != 0)
                {
                    // Same final-branch ownership transfer as eager evaluation.
                    p->set_posterior(node->lattice()->take_posterior());
                    p->update_probs_in_place(BBPA, re, _dilution);
                }
                else
                    p->update_probs(BBPA, re, _dilution);
                if (p->update_metadata_with_shrinking(_thres_up, _thres_lo
                                                      ))
                    p = p->to_local();
                node->set_child(re, std::make_unique<DistributedTree>(std::move(p), BBPA, ex, re, node->curr_stage() + 1, child_prob));
                lazy_eval(node->child(re), orig_lattice, true_state);
            }
        }
        else // test selection has been performed, children is (partially) initialized
        {
            // find BBPA and ex
            bgt_state_t ex = bgt::invalid_state();
            for (int re = 0; re < (1 << node->variants()); re++)
            {
                if (node->child(re) != nullptr)
                {
                    ex = node->child(re)->ex();
                    break;
                }
            }
            for (int re = 0; re < (1 << node->variants()); re++)
            {
                const bgt::branch_probability_t child_prob = static_cast<bgt::branch_probability_t>(
                    node->branch_prob() * orig_lattice->response_prob(ex, re, true_state, _dilution));
                node->child(re)->branch_prob(child_prob);
                lazy_eval(node->child(re), orig_lattice, true_state);
            }
        }
    }
    else
    {
        // A pre-constructed global tree might already have cleared this pointer.
        if (node->lattice()->posterior_probs() != nullptr)
        {
            node->destroy_posterior_probs();
        }
    }
}

void DistributedTree::initialize_mpi(int subjs, int k, int search_depth)
{
    // Get the number of processes
    BGT_MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &world_size));
    // Get the rank of the process
    BGT_MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

    backtrack.assign(static_cast<size_t>(search_depth + 1), nullptr);

    TreeStatsAccumulator::create_mpi_type(&stats_mpi_type, _search_depth, k);
    BGT_MPI_CHECK(MPI_Type_commit(&stats_mpi_type));
    BGT_MPI_CHECK(MPI_Op_create((MPI_User_function *)&TreeStatsAccumulator::mpi_reduce, true, &stats_mpi_op));
}

void DistributedTree::finalize_mpi()
{
    // Free datatype
    BGT_MPI_CHECK(MPI_Type_free(&stats_mpi_type));
    // Free tree stat reduce op
    BGT_MPI_CHECK(MPI_Op_free(&stats_mpi_op));

    backtrack.clear();
}

} // namespace bgt::tree
