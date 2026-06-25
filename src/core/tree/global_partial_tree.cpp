#include "bgt/detail/tree/tree.hpp"

namespace bgt::tree
{
using model::Lattice;


GlobalPartialTree::GlobalPartialTree(const Tree &other, bool deep) : DistributedTree(other, false)
{
    _selected_experiment = dynamic_cast<const GlobalPartialTree &>(other)._selected_experiment;
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

std::unique_ptr<Tree> GlobalPartialTree::clone(bool deep) const
{
    return std::make_unique<GlobalPartialTree>(*this, deep);
}

void GlobalPartialTree::eval(Tree *node, Lattice *orig_lattice, bgt_state_t true_state)
{
    if (!node->lattice()->is_classified() && node->curr_stage() < _search_depth && node->branch_prob() > _thres_branch)
    {
        if (!node->has_children()) // no child is explored, allocate children and perform computation
        {
            node->children(1 << node->variants());
            bgt_state_t BBPA = node->select_experiment();
            bgt_state_t ex = node->true_ex(BBPA);

            for (int re = 0; re < (1 << node->variants()); re++)
            {
                const bgt::branch_probability_t child_prob = static_cast<bgt::branch_probability_t>(
                    node->branch_prob() * orig_lattice->response_prob(ex, re, true_state, _dilution));
                auto p = node->lattice()->clone(SHALLOW_COPY_PROB_DIST);
                if (re == (1 << variants()) - 1 && node->curr_stage() != 0)
                {
                    // Transfer the parent posterior into the final response branch
                    // once all earlier branches have allocated their own updates.
                    p->set_posterior(node->lattice()->take_posterior());
                    p->update_probs_in_place(BBPA, re, _dilution);
                }
                else
                    p->update_probs(BBPA, re, _dilution);
                if (p->update_metadata_with_shrinking(_thres_up, _thres_lo
                                                      ))
                    p = p->to_local();
                node->set_child(re, std::make_unique<GlobalPartialTree>(std::move(p), BBPA, ex, re, node->curr_stage() + 1, child_prob));
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
                    BBPA = dynamic_cast<GlobalPartialTree *>(node->child(re))->_selected_experiment;
                    ex = node->child(re)->ex();
                    break;
                }
            }
            for (int re = 0; re < (1 << node->variants()); re++)
            {
                const bgt::branch_probability_t child_prob = static_cast<bgt::branch_probability_t>(
                    node->branch_prob() * orig_lattice->response_prob(ex, re, true_state, _dilution));
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
    else
    {
        // A pre-constructed global tree might already have cleared this pointer.
        if (node->lattice()->posterior_probs() != nullptr)
        {
            node->destroy_posterior_probs();
        }
    }
}

void GlobalPartialTree::lazy_eval(Tree *node, Lattice *orig_lattice, bgt_state_t true_state)
{
    backtrack[node->curr_stage()] = dynamic_cast<GlobalPartialTree *>(node);
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
                        // Rebuild the missing posterior chain from the nearest
                        // retained ancestor during lazy evaluation.
                        p->set_posterior(backtrack[i]->lattice()->take_posterior());
                        p->update_probs_in_place(backtrack[i + 1]->selected_experiment(), backtrack[i + 1]->ex_res(), _dilution);
                    }
                    else
                    {
                        p->update_probs(backtrack[i + 1]->selected_experiment(), backtrack[i + 1]->ex_res(), _dilution);
                    }
                    if (p->update_metadata_with_shrinking(_thres_up, _thres_lo
                                                          ))
                        p = p->to_local();
                    backtrack[i + 1]->lattice()->set_posterior(p->take_posterior());
                }
            }

            node->children(1 << node->variants());
            bgt_state_t BBPA = node->select_experiment();
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
                node->set_child(re, std::make_unique<GlobalPartialTree>(std::move(p), BBPA, ex, re, node->curr_stage() + 1, child_prob));
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

} // namespace bgt::tree
