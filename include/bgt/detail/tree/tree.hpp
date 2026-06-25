#pragma once
#include "bgt/detail/model/lattice.hpp"
#include "bgt/detail/tree/tree_stat.hpp"
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace bgt::tree
{

class Tree
{
protected:
    bgt_state_t _ex, _res;
    int _curr_stage;
    bgt::branch_probability_t _branch_prob;
    std::unique_ptr<model::Lattice> _lattice;
    std::vector<std::unique_ptr<Tree>> _children;
    static int _search_depth;
    static bgt::SelectorType _selector;
    static bgt::host_probability_t _thres_up, _thres_lo;
    static bgt::branch_probability_t _thres_branch;
    static bgt::host_probability_t **_dilution;

public:
    Tree(){};
    Tree(std::unique_ptr<model::Lattice> lattice, bgt_state_t ex, bgt_state_t res, int curr_stage, bgt::branch_probability_t prob);
    Tree(const Tree &other, bool deep);
    virtual ~Tree();
    virtual std::unique_ptr<Tree> clone(bool deep) const;
    model::Lattice *lattice() const { return _lattice.get(); }
    inline void lattice(std::unique_ptr<model::Lattice> lattice) { _lattice = std::move(lattice); }
    inline bgt_state_t ex() const { return _ex; }
    bgt_state_t true_ex(bgt_state_t halving);
    inline bgt_state_t ex_res() const { return _res; }
    inline int curr_stage() const { return _curr_stage; }
    inline int curr_subjs() const { return _lattice->curr_subjs(); }
    inline static int variants() { return model::Lattice::variants(); }
    inline bgt::branch_probability_t branch_prob() const { return _branch_prob; }
    inline void branch_prob(bgt::branch_probability_t prob) { _branch_prob = prob; }
    inline static bgt::host_probability_t thres_up() { return _thres_up; }
    inline static void thres_up(bgt::host_probability_t thres_up) { _thres_up = thres_up; }
    inline static bgt::host_probability_t thres_lo() { return _thres_lo; }
    inline static void thres_lo(bgt::host_probability_t thres_lo) { _thres_lo = thres_lo; }
    inline static bgt::branch_probability_t thres_branch() { return _thres_branch; }
    inline static void thres_branch(bgt::branch_probability_t thres_branch) { _thres_branch = thres_branch; }
    inline static int search_depth() { return _search_depth; }
    inline static void search_depth(int search_depth) { _search_depth = search_depth; }
    inline static bgt::SelectorType selector() { return _selector; }
    inline static void selector(bgt::SelectorType selector) { _selector = selector; }
    inline static bgt::host_probability_t **dilution() { return _dilution; }
    inline static void dilution(bgt::host_probability_t **dilution) { _dilution = dilution; }
    inline bool has_children() const { return !_children.empty(); }
    inline Tree *child(int index) const { return _children[static_cast<size_t>(index)].get(); }
    inline void set_child(int index, std::unique_ptr<Tree> child) { _children[static_cast<size_t>(index)] = std::move(child); }
    inline void children(int num)
    {
        _children.clear();
        _children.resize(static_cast<size_t>(num));
    }
    inline bool is_classified() const { return _lattice->is_classified(); }
    void parse(bgt_state_t true_state, const model::Lattice *org_lattice, bgt::host_probability_t sym_coef, TreeStatsAccumulator *ret) const;
    inline bgt::host_probability_t total_positive() const { return bgt::state_popcount(_lattice->pos_clas_atoms()); }
    inline bgt::host_probability_t total_negative() const { return bgt::state_popcount(_lattice->neg_clas_atoms()); }
    bool is_correct_clas(bgt_state_t true_state) const;
    bgt::host_probability_t fp(bgt_state_t true_state) const;
    bgt::host_probability_t fn(bgt_state_t true_state) const;
    static void find_all_leaves(const Tree *node, std::vector<const Tree *> *leaves);
    static void find_all_stat(const Tree *node, std::vector<const Tree *> *leaves);
    void apply_true_state(const model::Lattice *org_lattice, bgt_state_t true_state);
    bgt_state_t select_experiment() const;
    bgt_state_t select_experiment_serial() const;
    virtual std::string type() { return "Base Tree"; }
    inline void destroy_posterior_probs() { _lattice->free_posterior_probs(); }
};

class LocalTree : public Tree
{
public:
    LocalTree() : Tree() {}
    LocalTree(std::unique_ptr<model::Lattice> lattice, bgt_state_t ex, bgt_state_t res, int curr_stage) : Tree(std::move(lattice), ex, res, curr_stage, 0.0) {} // must be initialized to 0.0 so that stat tree can generate correct info
    LocalTree(std::unique_ptr<model::Lattice> lattice, bgt_state_t ex, bgt_state_t res, int k, int curr_stage);
    LocalTree(const Tree &other, bool deep);
    std::unique_ptr<Tree> clone(bool deep) const override;
    virtual std::string type() override { return "Global Tree Serial"; }
};

class GlobalTree : public LocalTree
{
protected:
    static int rank, world_size;

public:
    GlobalTree() : LocalTree() {}
    GlobalTree(std::unique_ptr<model::Lattice> lattice, bgt_state_t ex, bgt_state_t res, int curr_stage) : LocalTree(std::move(lattice), ex, res, curr_stage) {}
    GlobalTree(std::unique_ptr<model::Lattice> lattice, bgt_state_t ex, bgt_state_t res, int k, int curr_stage);
    GlobalTree(const Tree &other, bool deep);
    std::unique_ptr<Tree> clone(bool deep) const override;
    virtual std::string type() override { return "Global Tree"; }

    static MPI_Datatype stats_mpi_type;
    static MPI_Op stats_mpi_op;
    static void initialize_mpi(int subjs, int k);
    static void finalize_mpi();
};

class DistributedTree : public Tree
{
protected:
    static int rank, world_size;
    static std::vector<DistributedTree *> backtrack;
    bgt_state_t _selected_experiment;

public:
    DistributedTree(std::unique_ptr<model::Lattice> lattice, bgt_state_t selected_experiment, bgt_state_t ex, bgt_state_t res, int curr_stage, bgt::branch_probability_t prob);
    DistributedTree(std::unique_ptr<model::Lattice> lattice, bgt_state_t selected_experiment, bgt_state_t ex, bgt_state_t res, int k, int curr_stage, int expansion_depth);
    DistributedTree(const Tree &other, bool deep);
    std::unique_ptr<Tree> clone(bool deep) const override;
    inline bgt_state_t selected_experiment() const { return _selected_experiment; }
    inline void selected_experiment(bgt_state_t value) { _selected_experiment = value; }
    static void eval(Tree *node, model::Lattice *orig_lattice, bgt_state_t true_state);
    static void lazy_eval(Tree *node, model::Lattice *orig_lattice, bgt_state_t true_state);
    static MPI_Datatype stats_mpi_type;
    static MPI_Op stats_mpi_op;
    static void initialize_mpi(int subjs, int k, int search_depth);
    static void finalize_mpi();
    virtual std::string type() override { return "Distributed Tree"; }
};

class GlobalPartialTree : public DistributedTree
{
public:
    GlobalPartialTree(std::unique_ptr<model::Lattice> lattice, bgt_state_t halving, bgt_state_t ex, bgt_state_t res, int curr_stage, bgt::branch_probability_t prob) : DistributedTree(std::move(lattice), halving, ex, res, curr_stage, prob){}
    GlobalPartialTree(const Tree &other, bool deep);
    std::unique_ptr<Tree> clone(bool deep) const override;
    static void eval(Tree *node, model::Lattice *orig_lattice, bgt_state_t true_state);
    static void lazy_eval(Tree *node, model::Lattice *orig_lattice, bgt_state_t true_state);
    virtual std::string type() override { return "Global Partial Tree"; }
};

class FusionTree : public GlobalTree
{
protected:
    static std::vector<FusionTree *> sequence_tracer;

public:
    FusionTree(std::unique_ptr<model::Lattice> lattice, bgt_state_t ex, bgt_state_t res, int curr_stage) : GlobalTree(std::move(lattice), ex, res, curr_stage) {}
    FusionTree(std::unique_ptr<model::Lattice> lattice, bgt_state_t ex, bgt_state_t res, int k, int curr_stage, bgt::branch_probability_t prun_thres_sum, bgt::branch_probability_t curr_prun_thres_sum, bgt::branch_probability_t prun_thres);
    FusionTree(const Tree &other, bool deep);
    std::unique_ptr<Tree> clone(bool deep) const override;
    bgt::branch_probability_t fusion_branch_prob(int ex, int res);
    virtual std::string type() override { return "Fusion Tree"; }
    static void initialize_mpi(int subjs, int k);
    static void finalize_mpi();
};

std::vector<bgt_state_t> generate_symmetric_true_states(int subjs, int variants, std::vector<int> &symm_coefficients);
std::vector<int> trim_true_states(const bgt::posterior_t *values, int n, bgt::host_probability_t percent);

} // namespace bgt::tree
