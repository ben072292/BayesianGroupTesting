#include "bgt/detail/model/lattice.hpp"
#include "bgt/detail/cuda_provider.hpp"
#include "bgt/detail/performance.hpp"
#include "bgt/detail/runtime/simulation_helpers.hpp"

#include <stdexcept>
#include <vector>

using bgt::model::create_lattice;
using bgt::tree::LocalTree;
using bgt::tree::Tree;
using bgt::tree::TreeStatsAccumulator;
using bgt::runtime_detail::accumulate_tree_stats;
using bgt::runtime_detail::clear_tree_stats;
using bgt::runtime_detail::lattice_type_is_distributed;
using bgt::runtime_detail::lattice_type_uses_dilution;
using bgt::runtime_detail::resolve_selector;

namespace bgt
{

TreeStats CpuProvider::run(LatticeType type, int subjects, int variants, std::span<const host_probability_t> prior,
						   const SimulationOptions &options, const DilutionTable *dilution) const
{
	BGT_PERF_SCOPE_FOR(LogSubsystem::runtime, "provider.cpu.run");
	if (subjects <= 0 || variants <= 0 || options.search_depth < 0)
		throw std::invalid_argument("invalid simulation dimensions.");
	if (resolve_selector(options.selector, variants) == SelectorType::op_bha && variants != 1)
		throw std::invalid_argument("Op-BHA tree selection is only supported for binary lattices.");
	if (lattice_type_is_distributed(type))
		throw std::invalid_argument("CPU provider expects a replicated lattice; use an parallel simulation mode for distributed lattices.");
	if (lattice_type_uses_dilution(type) && dilution == nullptr)
		throw std::invalid_argument("dilution table is required for dilution models.");

	TreeStats stats;
	clear_tree_stats(stats);
	std::vector<host_probability_t> prior_holder(prior.begin(), prior.end());
	auto original_lattice = create_lattice(to_internal_lattice_type(type), subjects, variants, prior_holder.data());
	auto tree_lattice = create_lattice(to_internal_lattice_type(type), subjects, variants, prior_holder.data());

	Tree::search_depth(options.search_depth);
	Tree::thres_up(options.threshold_up);
	Tree::thres_lo(options.threshold_lo);
	Tree::thres_branch(options.branch_threshold);
	Tree::selector(resolve_selector(options.selector, variants));
	Tree::dilution(dilution == nullptr ? nullptr : const_cast<DilutionTable *>(dilution)->rows());

	LocalTree tree(std::move(tree_lattice), 0, 0, 1, 0);
	std::vector<const Tree *> leaves;
	Tree::find_all_leaves(&tree, &leaves);
	stats.total_leaves = static_cast<int>(leaves.size());

	const int total_states = state_count(subjects * variants);
	for (int true_state = 0; true_state < total_states; true_state++)
	{
		TreeStatsAccumulator state_stat(options.search_depth, 1);
		tree.apply_true_state(original_lattice.get(), static_cast<state_t>(true_state));
		tree.parse(static_cast<state_t>(true_state), original_lattice.get(), 1.0, &state_stat);
		accumulate_tree_stats(state_stat, stats);
	}
	return stats;
}

bool CudaProvider::available() const
{
#ifdef BGT_ENABLE_CUDA
	return bgt_cuda_provider_available() != 0;
#else
	return false;
#endif
}

TreeStats CudaProvider::run(LatticeType type, int subjects, int variants, std::span<const host_probability_t> prior,
							const SimulationOptions &options, const DilutionTable *dilution) const
{
	BGT_PERF_SCOPE_FOR(LogSubsystem::cuda, "provider.cuda.run");
	if (resolve_selector(options.selector, variants) == SelectorType::op_bha && variants != 1)
		throw std::invalid_argument("Op-BHA tree selection is only supported for binary lattices.");
	if (lattice_type_uses_dilution(type) && dilution == nullptr)
		throw std::invalid_argument("dilution table is required for dilution models.");
#ifdef BGT_ENABLE_CUDA
	const SelectorType selector = resolve_selector(options.selector, variants);
	TreeStats stats;
	const bool distributed = lattice_type_is_distributed(type);
	const bool requested_gin = distributed && options.compile_options.enable_nccl_gin;
	bool use_gin = false;
#ifdef BGT_ENABLE_NCCL_GIN
	use_gin = requested_gin;
#else
	if (requested_gin && options.provider == Provider::cuda)
		throw std::runtime_error("NCCL GIN was requested, but this build was not configured with BGT_ENABLE_NCCL_GIN.");
#endif
	auto run_once = [&](bool enable_gin) {
		return distributed
				   ? bgt_cuda_provider_run_distributed(
						 lattice_type_uses_dilution(type) ? 1 : 0, subjects, variants,
						 prior.data(), options.search_depth, options.threshold_up, options.threshold_lo,
						 options.branch_threshold, static_cast<int>(selector), enable_gin ? 1 : 0,
						 dilution == nullptr ? nullptr : const_cast<DilutionTable *>(dilution)->rows(), &stats)
				   : bgt_cuda_provider_run(
						 lattice_type_uses_dilution(type) ? 1 : 0, subjects, variants,
						 prior.data(), options.search_depth, options.threshold_up, options.threshold_lo,
						 options.branch_threshold, static_cast<int>(selector),
						 dilution == nullptr ? nullptr : const_cast<DilutionTable *>(dilution)->rows(), &stats);
	};
	int status = 0;
	try
	{
		status = run_once(use_gin);
	}
	catch (const std::exception &)
	{
		if (!(use_gin && options.provider == Provider::auto_select))
			throw;
		status = run_once(false);
	}
	if (status != 0 && use_gin && options.provider == Provider::auto_select)
	{
		status = run_once(false);
	}
	if (status != 0)
		throw std::runtime_error("CUDA provider failed or does not support this KernelSpec.");
	return stats;
#else
	(void)type;
	(void)subjects;
	(void)variants;
	(void)prior;
	(void)options;
	(void)dilution;
	throw std::runtime_error("CUDA provider was not compiled.");
#endif
}


bool cuda_available()
{
	return CudaProvider{}.available();
}

std::vector<Provider> available_providers()
{
	return Runtime{}.available_providers();
}

} // namespace bgt
