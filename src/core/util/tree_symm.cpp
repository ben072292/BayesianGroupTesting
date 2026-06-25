#include "bgt/detail/tree/tree.hpp"
#include <algorithm>
#include <unordered_map>

namespace bgt::tree
{
using model::Lattice;

// generate n prime numbers
void generate_primes(size_t n, std::vector<int> &prime_array)
{
    int num = 2; // Start with the first prime number

    while (prime_array.size() < n)
    {
        bool isPrime = true;
        for (int i = 2; i * i <= num; ++i)
        {
            if (num % i == 0)
            {
                isPrime = false;
                break;
            }
        }
        if (isPrime)
        {
            prime_array.push_back(num);
        }
        ++num;
    }
}

double n_choose_k(int n, int k)
{
    if (n < 0 || k < 0 || k > n)
        return 0.0;

    std::vector<double> dp(static_cast<size_t>(k + 1), 0.0);
    dp[0] = 1.0;

    for (int i = 1; i <= n; ++i)
    {
        for (int j = std::min(i, k); j > 0; --j)
            dp[static_cast<size_t>(j)] += dp[static_cast<size_t>(j - 1)];
    }

    return dp[static_cast<size_t>(k)];
}

// Function to calculate the product of a combination
long long calculate_product(const std::vector<int> &primes, const std::vector<int> &combination)
{
    long long product = 1;
    for (int num : combination)
    {
        product *= primes[num];
    }
    return product;
}

// Function to generate combinations with repetition and store product sums
void generateCombinations(const std::vector<int> &primes, int N, std::vector<int> &combination, size_t curr_idx, std::vector<long long> &combination_products)
{
    if (N == 0)
    {
        long long product = calculate_product(primes, combination);
        combination_products.push_back(product);
        return;
    }

    if (curr_idx == primes.size())
    {
        return;
    }

    for (size_t i = curr_idx; i < primes.size(); ++i)
    {
        combination.push_back(i);
        generateCombinations(primes, N - 1, combination, i, combination_products);
        combination.pop_back();
    }
}

std::vector<bgt_state_t> generate_symmetric_true_states(int subjs, int variants, std::vector<int> &symm_coefficients)
{
    /**
     * N is subjs, k is variants.
     * The formula "Number of Combinations = (N + (1<<k) - 1) choose N" is derived from the concept of
     * combinations with repetition, which is a way to calculate the number of ways to choose (1<<k) items
     * from a set of N items when repetition is allowed.
     * In your case, you have 12 slots (N) and 4 choices (00, 01, 10 and 11). To calculate the
     * number of combinations of filling the 12 slots, you can think of it as choosing
     * from the 4 choices with repetition allowed.
     * The formula "Number of Combinations = (n + (1<<k) - 1) choose N" is a general formula for this situation. In this case:
     * (1<<k) is the number of choices (4: 00, 01, 10, and 11).
     * N is the number of slots (12).
     * So, using the formula:
     * Number of Combinations = (4 + 12 - 1) choose 12 = 15 choose 12 = 455
     */
    const int size = static_cast<int>(n_choose_k(subjs + (1 << variants) - 1, subjs));
    std::vector<bgt_state_t> symm_true_states(static_cast<size_t>(size), 0);
    symm_coefficients.assign(static_cast<size_t>(size), 0);
    std::vector<char> set_symmetric_true_state(static_cast<size_t>(size), 0);

    std::vector<int> prime_array;
    std::vector<int> combination_array;
    std::vector<long long> combination_product_array;
    generate_primes((1 << variants), prime_array);
    generateCombinations(prime_array, subjs, combination_array, 0, combination_product_array);
    if (combination_product_array.size() != static_cast<size_t>(size))
    {
        BGT_LOG_FATAL(bgt::LogSubsystem::tree, "Symmetry combination algorithm generated an unexpected number of states.");
        throw bgt::Error(bgt::Status::internal_error("symmetry combination algorithm generated an unexpected number of states", __FILE__, __LINE__));
    }
    std::unordered_map<long long, size_t> combination_product_map;
    for (size_t i = 0; i < combination_product_array.size(); ++i)
    {
        combination_product_map[combination_product_array[i]] = i;
    }
    long long prime_product = 1ll;
    for (int state = 0; state < bgt::state_count(subjs * variants); state++)
    {
        for (int subj = 0; subj < subjs; subj++)
        {
            size_t prime_idx = 0;
            for (int variant = 0; variant < variants; variant++)
            {
                prime_idx |= (((state >> (variant * subjs)) & (1 << subj)) ? (1 << variant) : 0);
            }
            prime_product *= prime_array[prime_idx];
        }
        int prime_product_idx = combination_product_map[prime_product];
        if (!set_symmetric_true_state[static_cast<size_t>(prime_product_idx)])
        {
            symm_true_states[static_cast<size_t>(prime_product_idx)] = static_cast<bgt_state_t>(state);
            set_symmetric_true_state[static_cast<size_t>(prime_product_idx)] = true;
        }
        symm_coefficients[static_cast<size_t>(prime_product_idx)]++;
        prime_product = 1ll;
    }
    return symm_true_states;
}

} // namespace bgt::tree
