#include "bgt/detail/model/lattice.hpp"

using namespace bgt::model;

#include <cmath>
#include <cstdlib>
#include <iostream>

namespace
{
void expect_near(double actual, double expected, const char *label)
{
    const double tolerance =
#if BGT_POSTERIOR_BITS == 32 || BGT_ACCUMULATOR_BITS == 32
        1e-5;
#else
        1e-12;
#endif
    if (std::fabs(actual - expected) > tolerance)
    {
        std::cerr << label << ": expected " << expected << ", got " << actual << std::endl;
        std::exit(1);
    }
}

void expect_equal(bgt_state_t actual, bgt_state_t expected, const char *label)
{
    if (actual != expected)
    {
        std::cerr << label << ": expected " << expected << ", got " << actual << std::endl;
        std::exit(1);
    }
}
}

int main()
{
    double pi0[] = {0.2, 0.4};
    auto lattice = create_lattice(REPL_NON_DILUTION, 2, 1, pi0);

    expect_near(lattice->posterior_prob(0), 0.08, "prior state 0");
    expect_near(lattice->posterior_prob(1), 0.32, "prior state 1");
    expect_near(lattice->posterior_prob(2), 0.12, "prior state 2");
    expect_near(lattice->posterior_prob(3), 0.48, "prior state 3");

    bgt_state_t up_set[4] = {0, 0, 0, 0};
    lattice->get_up_set(1, up_set);
    expect_equal(up_set[0], 1, "up-set state 0");
    expect_equal(up_set[1], 3, "up-set state 1");

    lattice->update_probs(1, 1, nullptr);
    const double denominator = 0.08 * 0.01 + 0.32 * 0.99 + 0.12 * 0.01 + 0.48 * 0.99;
    expect_near(lattice->posterior_prob(0), 0.08 * 0.01 / denominator, "posterior state 0");
    expect_near(lattice->posterior_prob(1), 0.32 * 0.99 / denominator, "posterior state 1");
    expect_near(lattice->posterior_prob(2), 0.12 * 0.01 / denominator, "posterior state 2");
    expect_near(lattice->posterior_prob(3), 0.48 * 0.99 / denominator, "posterior state 3");
    double **dilution = generate_dilution(2, 0.99, 0.005);
    expect_near(dilution[0][0], 0.99, "dilution n=1 r=0");
    expect_near(dilution[0][1], 0.01, "dilution n=1 r=1");
    expect_near(dilution[1][0], 0.99, "dilution n=2 r=0");
    expect_near(dilution[1][1], 1.0 - 0.99 / 1.005, "dilution n=2 r=1");
    expect_near(dilution[1][2], 0.01, "dilution n=2 r=2");

    free_dilution(dilution, 2);

    return 0;
}
