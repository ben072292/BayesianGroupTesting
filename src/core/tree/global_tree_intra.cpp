#include "bgt/detail/tree/tree.hpp"

namespace bgt::tree
{
using model::Lattice;


LocalTree::LocalTree(std::unique_ptr<Lattice> lattice, bgt_state_t ex, bgt_state_t res, int k, int curr_stage)
    : LocalTree(std::move(lattice), ex, res, curr_stage)
{
    if (!_lattice->is_classified() && curr_stage < _search_depth)
    {
        children(1 << variants());
        bgt_state_t BBPA = select_experiment();
        bgt_state_t ex = true_ex(BBPA); // full-sized experiment should be generated before posterior probability distribution is updated, because unupdated clas_subj_ should be used to calculate the correct value
        for (int re = 0; re < (1 << variants()); re++)
        {
            auto p = _lattice->clone(SHALLOW_COPY_PROB_DIST);
            if (re != (1 << variants()) - 1)
            {
                p->update_probs(BBPA, re, _dilution);
            }
            else
            {
                // Transfer the parent posterior into the final response branch so
                // the tree avoids a redundant full posterior allocation.
                p->set_posterior(_lattice->take_posterior());
                p->update_probs_in_place(BBPA, re, _dilution);
            }
            if (p->update_metadata_with_shrinking(_thres_up, _thres_lo))
                p = p->to_local();
            set_child(re, std::make_unique<LocalTree>(std::move(p), ex, re, k, _curr_stage + 1));
        }
    }
    else
    {
        // Terminal nodes do not need a posterior during later statistic traversal.
        destroy_posterior_probs();
    }
}

LocalTree::LocalTree(const Tree &other, bool deep) : Tree(other, false)
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

std::unique_ptr<Tree> LocalTree::clone(bool deep) const
{
    return std::make_unique<LocalTree>(*this, deep);
}

} // namespace bgt::tree
