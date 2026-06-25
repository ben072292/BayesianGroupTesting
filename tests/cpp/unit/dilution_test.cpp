#include "support/assertions.hpp"

#include <vector>

int main()
{
	std::vector<double> prior{0.2, 0.4};
	bgt::Lattice dilution_lattice(bgt::LatticeType::replicated_dilution, 2, prior);
	bgt::DilutionTable dilution(2, 0.99, 0.005);

	bgt::test::expect_near(dilution_lattice.response_probability(3, 1, 3, &dilution), 0.99, "dilution all negative");
	bgt::test::expect_near(dilution_lattice.response_probability(3, 1, 1, &dilution), 1.0 - 0.99 / 1.005, "dilution one positive");
	bgt::test::expect_near(dilution_lattice.response_probability(3, 0, 1, &dilution), 0.99 / 1.005, "dilution positive response");

	return 0;
}
