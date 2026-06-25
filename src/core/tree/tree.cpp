#include "bgt/detail/tree/tree.hpp"

#include <cmath>

namespace bgt::tree
{
using model::Lattice;


int Tree::_search_depth = 0;
bgt::SelectorType Tree::_selector = bgt::SelectorType::auto_select;
bgt::host_probability_t Tree::_thres_up = 0.01;
bgt::host_probability_t Tree::_thres_lo = 0.01;
bgt::branch_probability_t Tree::_thres_branch = 0.001;
bgt::host_probability_t **Tree::_dilution;

Tree::Tree(std::unique_ptr<Lattice> lattice, bgt_state_t ex, bgt_state_t res, int curr_stage, bgt::branch_probability_t prob)
    : _lattice(std::move(lattice))
{
    _ex = ex;
    _res = res;
    _curr_stage = curr_stage;
    _branch_prob = prob;
}

Tree::Tree(const Tree &other, bool deep)
    : _lattice(other._lattice->clone(NO_COPY_PROB_DIST))
{
    _ex = other._ex;
    _res = other._res;
    _branch_prob = other._branch_prob;
    _curr_stage = other._curr_stage;
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

Tree::~Tree() = default;

std::unique_ptr<Tree> Tree::clone(bool deep) const
{
    return std::make_unique<Tree>(*this, deep);
}

// Convert halving selection to full-size experiment (because of lattice shrinking).
// This function should be called inside the tree before posterior probability update.
// Also note it is recommended ot not use this function if lattice shrinking is disabled,
// though initial evaluation shows no statistical changes.
bgt_state_t Tree::true_ex(bgt_state_t halving)
{
    bgt_state_t ret = 0, pos = 0;
    for (int i = 0; i < Lattice::orig_subjs(); i++)
    {
        if (!(_lattice->clas_subjs() & (1 << i)))
        {
            if ((halving & (1 << pos)))
                ret |= (1 << i);
            ++pos;
        }
    }
    return ret;
}

void Tree::parse(bgt_state_t true_state, const Lattice *org_lattice, bgt::host_probability_t sym_coef, TreeStatsAccumulator *stat) const
{
    stat->clear();
    const bgt::host_probability_t coef = org_lattice->prior_prob(true_state, Lattice::pi0()) * sym_coef;
    std::vector<const Tree *> leaves;
    find_all_stat(this, &leaves);
    int size = leaves.size(), k = stat->k();

    for (int i = 0; i < size; i++)
    {
        const Tree *leaf = leaves[static_cast<size_t>(i)];
        int index = leaf->curr_stage();
        if (leaf->is_classified() && leaf->is_correct_clas(true_state))
        {
            stat->correct()[index] += leaf->_branch_prob * coef;
        }
        else if (leaf->is_classified() && !leaf->is_correct_clas(true_state))
        {
            stat->incorrect()[index] += leaf->_branch_prob * coef;
            stat->fp()[index] += leaf->fp(true_state) * coef * leaf->_branch_prob;
            stat->fn()[index] += leaf->fn(true_state) * coef * leaf->_branch_prob;
        }
        else if (!leaf->is_classified())
        {
            stat->unclassified(stat->unclassified() + leaf->_branch_prob);
        }
        stat->expected_stage(stat->expected_stage() + std::ceil(static_cast<bgt::host_probability_t>(index) / static_cast<bgt::host_probability_t>(k)) * leaf->_branch_prob);
        stat->expected_test(stat->expected_test() + index * leaf->_branch_prob);
    }
    for (int i = 0; i < size; i++)
    {
        const Tree *leaf = leaves[static_cast<size_t>(i)];
        int index = leaf->curr_stage();
        stat->stage_sd(std::pow((std::ceil(static_cast<bgt::host_probability_t>(index) / static_cast<bgt::host_probability_t>(k)) - stat->expected_stage()), 2) * leaf->_branch_prob);
        stat->test_sd(std::pow(index - stat->expected_test(), 2) * leaf->_branch_prob);
    }
    stat->unclassified(stat->unclassified() * coef);
    stat->stage_sd(std::sqrt(stat->stage_sd()) * coef);
    stat->test_sd(std::sqrt(stat->test_sd()) * coef);
    stat->expected_stage(stat->expected_stage() * coef);
    stat->expected_test(stat->expected_test() * coef);
}

bool Tree::is_correct_clas(bgt_state_t true_state) const
{
    return _lattice->neg_clas_atoms() == true_state;
}

// neg_clas ^ true_state filter out atoms that are wrongly classified
// then & true_state (1 means negative, 0 means positive) filters out wrong positives
// that was suppose to be negatives
bgt::host_probability_t Tree::fp(bgt_state_t true_state) const
{
    return total_positive() == 0.0 ? 0.0 : bgt::state_popcount((_lattice->neg_clas_atoms() ^ true_state) & true_state) / total_positive();
}

// neg_clas ^ true_state filter out atoms that are wrongly classified
// then & ~true_state (0 means negative, 1 means positive) filters out wrong negatives
// that was suppose to be positives
bgt::host_probability_t Tree::fn(bgt_state_t true_state) const
{
    return total_negative() == 0.0 ? 0.0 : bgt::state_popcount((_lattice->neg_clas_atoms() ^ true_state) & (~true_state)) / total_negative();
}

void Tree::find_all_leaves(const Tree *node, std::vector<const Tree *> *leaves)
{
    if (node == nullptr)
        return;
    if (!node->has_children())
    {
        leaves->push_back(node);
    }
    else
    {
        for (int i = 0; i < (1 << node->variants()); i++)
        {
            if (node->child(i) != nullptr)
            {
                find_all_leaves(node->child(i), leaves);
            }
        }
    }
}

void Tree::find_all_stat(const Tree *node, std::vector<const Tree *> *leaves)
{
    if (node == nullptr || node->_branch_prob < _thres_branch)
        return;
    if (!node->has_children())
    {
        leaves->push_back(node);
    }
    else
    {
        for (int i = 0; i < (1 << node->variants()); i++)
        {
            if (node->child(i) != nullptr)
            {
                find_all_stat(node->child(i), leaves);
            }
        }
    }
}

void apply_true_state_helper(const Lattice *__restrict__ org_lattice, Tree *__restrict__ node, bgt_state_t true_state, bgt::branch_probability_t prob, bgt::branch_probability_t thres_branch, bgt::host_probability_t **dilution)
{
    if (node == nullptr)
        return;
    node->branch_prob(prob);
    if (node->has_children())
    {
        for (int i = 0; i < (1 << node->variants()); i++)
        {
            if (node->child(i) != nullptr)
            {

                const bgt::branch_probability_t child_prob = static_cast<bgt::branch_probability_t>(
                    prob * org_lattice->response_prob(node->child(i)->ex(), node->child(i)->ex_res(), true_state, dilution));
                if (child_prob > thres_branch)
                {
                    apply_true_state_helper(org_lattice, node->child(i), true_state, child_prob, thres_branch, dilution);
                }
                else
                {
                    node->child(i)->branch_prob(0.0); // otherwise parsing stat tree will lead to incorrect result
                }
            }
        }
    }
}

bgt_state_t Tree::select_experiment() const
{
    return _lattice->select_experiment(_selector);
}

bgt_state_t Tree::select_experiment_serial() const
{
    return _lattice->select_experiment_serial(_selector);
}

void Tree::apply_true_state(const Lattice *org_lattice, bgt_state_t true_state)
{
    apply_true_state_helper(org_lattice, this, true_state, 1.0, _thres_branch, _dilution);
}

} // namespace bgt::tree
