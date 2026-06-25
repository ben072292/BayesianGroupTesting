#pragma once

#include "bgt/types.hpp"

int bgt_cuda_provider_available();
int bgt_cuda_provider_run(int use_dilution, int subjects, int variants,
						  const bgt::host_probability_t *prior, int search_depth,
						  bgt::host_probability_t threshold_up, bgt::host_probability_t threshold_lo, bgt::branch_probability_t branch_threshold,
						  int selector, bgt::host_probability_t **dilution, bgt::TreeStats *stats);
int bgt_cuda_provider_run_distributed(int use_dilution, int subjects, int variants,
									  const bgt::host_probability_t *prior, int search_depth,
									  bgt::host_probability_t threshold_up, bgt::host_probability_t threshold_lo, bgt::branch_probability_t branch_threshold,
									  int selector, int enable_nccl_gin, bgt::host_probability_t **dilution, bgt::TreeStats *stats);
int bgt_cuda_provider_benchmark_select(int subjects, int variants,
									   const bgt::host_probability_t *prior,
									   int selector,
									   int iterations,
									   int warmup,
									   bgt::state_t *candidate,
									   double *init_seconds,
									   double *mean_seconds,
									   double *min_seconds,
									   double *max_seconds,
									   double *stddev_seconds);
