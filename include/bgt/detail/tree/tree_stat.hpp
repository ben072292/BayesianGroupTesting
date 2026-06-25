#pragma once
#include "bgt/detail/common.hpp"

namespace bgt::tree
{

struct TreeStatsAccumulator
{
private:
    int _depth, _k, _total_leaves;
    bgt::statistic_t _unclas, _exp_stage, _exp_test, _stage_sd, _test_sd;
    bgt::statistic_t _correct[30];
    bgt::statistic_t _incorrect[30];
    bgt::statistic_t _fp[30];
    bgt::statistic_t _fn[30];

public:
    TreeStatsAccumulator(int depth, int k)
    {
        _depth = depth;
        _k = k;
        clear();
    }

    ~TreeStatsAccumulator(){}

    void clear();
    int depth() const { return _depth; }
    int k() const { return _k; }
    bgt::statistic_t *correct() { return _correct; }
    bgt::statistic_t *incorrect() { return _incorrect; }
    bgt::statistic_t *fp() { return _fp; }
    bgt::statistic_t *fn() { return _fn; }
    bgt::statistic_t unclassified() const { return _unclas; }
    void unclassified(bgt::statistic_t val) { _unclas = val; }
    bgt::statistic_t expected_stage() const { return _exp_stage; }
    void expected_stage(bgt::statistic_t val) { _exp_stage = val; }
    bgt::statistic_t expected_test() const { return _exp_test; }
    void expected_test(bgt::statistic_t val) { _exp_test = val; }
    bgt::statistic_t stage_sd() const { return _stage_sd; }
    void stage_sd(bgt::statistic_t val) { _stage_sd = val; }
    bgt::statistic_t test_sd() const { return _test_sd; }
    void test_sd(bgt::statistic_t val) { _test_sd = val; }
    int total_leaves() const { return _total_leaves; }
    void total_leaves(int val) { _total_leaves = val; }
    void merge(TreeStatsAccumulator *other);
    void output_detail() const;

    static void mpi_reduce(TreeStatsAccumulator* in, TreeStatsAccumulator* inout, int* len, MPI_Datatype *dptr);
    static void create_mpi_type(MPI_Datatype *stats_mpi_type, int stages, int k);

};

} // namespace bgt::tree
