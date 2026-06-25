#include "bgt/detail/model/dilution.hpp"
#include "bgt/lattice.hpp"

#include <memory>
#include <stdexcept>

namespace bgt::model
{

bgt::host_probability_t **generate_dilution(int n, bgt::host_probability_t alpha, bgt::host_probability_t h)
{
	std::unique_ptr<bgt::host_probability_t *[]> ret = std::make_unique<bgt::host_probability_t *[]>(static_cast<std::size_t>(n));
    int k;
    for (int rk = 1; rk <= n; rk++)
    {
		std::unique_ptr<bgt::host_probability_t[]> row = std::make_unique<bgt::host_probability_t[]>(static_cast<std::size_t>(rk + 1));
		row[0] = alpha;
        for (int r = 1; r <= rk; r++)
        {
            k = rk - r;
			row[r] = 1 - alpha * r / (k * h + r);
        }
		ret[rk - 1] = row.release();
    }
	return ret.release();
}

void free_dilution(bgt::host_probability_t **dilution, int subjects)
{
	if (dilution == nullptr)
		return;
	for (int i = 0; i < subjects; i++)
	{
		std::unique_ptr<bgt::host_probability_t[]> row(dilution[i]);
	}
	std::unique_ptr<bgt::host_probability_t *[]> rows(dilution);
}

} // namespace bgt::model

namespace bgt
{

DilutionTable::DilutionTable(int subjects, bgt::host_probability_t alpha, bgt::host_probability_t h) : subjects_(subjects)
{
	if (subjects <= 0)
		throw std::invalid_argument("subjects must be positive.");
	values_.assign(static_cast<std::size_t>(subjects) * static_cast<std::size_t>(subjects + 1), 0.0);
	rows_.resize(subjects);
	for (int rk = 1; rk <= subjects; rk++)
	{
		rows_[rk - 1] = values_.data() + static_cast<std::size_t>(rk - 1) * static_cast<std::size_t>(subjects + 1);
		rows_[rk - 1][0] = alpha;
		for (int r = 1; r <= rk; r++)
		{
			const int k = rk - r;
			rows_[rk - 1][r] = 1 - alpha * r / (k * h + r);
		}
	}
}

bgt::host_probability_t DilutionTable::at(int group_size, int positives) const
{
	if (group_size <= 0 || group_size > subjects_ || positives < 0 || positives > group_size)
		throw std::out_of_range("dilution table index is out of range.");
	return rows_[group_size - 1][positives];
}

} // namespace bgt
