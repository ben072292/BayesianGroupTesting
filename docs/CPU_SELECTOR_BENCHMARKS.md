# CPU Selector Benchmarks And Scaling Notes

This note records the current CPU-path measurements for Op-BHA and Op-BBPA and the scaling decisions they imply. The numbers below are local Mac measurements, so they are useful for relative behavior and ablations, not final HPC roofline claims. On cluster hardware, repeat the same commands with hardware counters.

## Intended Parallel Model

There are two separate parallel layers:

- Tree-level parallelism: `parallel_dynamic_tree` and `parallel_hybrid_tree` use rank 0 as a work dispatcher. Workers send a `READY` message, rank 0 receives with `MPI_ANY_SOURCE`, and the next true-state chunk is assigned first-come-first-serve. Workers run `DistributedTree::lazy_eval`, which calls `select_experiment_serial()`. For binary trees, this means Op-BHA is computed locally by the worker with the zeta-transform path, not by a distributed-lattice allreduce.
- Distributed-lattice selector primitives: `DistributedLattice::op_bha()` and `DistributedLattice::BBPA()` operate on a posterior array already partitioned across ranks. These are lower-level collectives for partitioned lattice storage, not the preferred scheduling model for dynamic binary tree simulation.

For large scale BGT tree simulation, the default target should therefore be single-core efficient Op-BHA inside each worker plus MPI wildcard P2P work dispatch at the tree level. Collective selector primitives remain useful when the lattice itself must be distributed.

## Benchmark Driver

Build with examples enabled:

```bash
cmake -S . -B build -DBGT_BUILD_EXAMPLES=ON
cmake --build build -j 8 --target selector_roofline_benchmark
```

Example commands:

```bash
build/bin/selector_roofline_benchmark --model=replicated --selector=op_bha --subjects=14 --variants=1 --iterations=50 --warmup=10
build/bin/selector_roofline_benchmark --model=replicated --selector=op_bbpa --subjects=14 --variants=1 --iterations=10 --warmup=2
mpirun -np 2 build/bin/selector_roofline_benchmark --model=distributed --selector=op_bbpa --subjects=7 --variants=2 --iterations=30 --warmup=5
```

The benchmark reports an estimated operation count, estimated memory traffic, estimated collective payload, effective GOP/s, effective GB/s, and arithmetic intensity. These are model estimates, not hardware-counter measurements.

## Roofline Interpretation

Op-BHA for binary lattices is a subset zeta transform over the posterior array. Its estimated work is roughly `subjects * 2^(subjects - 1)` additions plus a scan. Arithmetic intensity is low, so it is memory-bandwidth sensitive.

Op-BBPA uses the same exact objective as the original brute-force sweep, but the default CPU implementation now evaluates it through subset histograms, zeta transforms, and inclusion-exclusion. For binary `k=1`, this degenerates to the same upset-mass transform used by Op-BHA with a two-bucket BBPA score. For multinomial `k>1`, the transform builds one feature table per non-empty variant subset and reconstructs exact response buckets without looping over every experiment/state pair.

For hardware roofline on Linux/HPC, use `perf`, LIKWID, VTune, or Nsight Compute/Systems for GPU/NCCL runs. The local benchmark is intentionally dependency-light and only measures wall time.

## Current Measurements

All rows use the default optimized build unless stated otherwise.

| Model | Selector | Subjects | Variants | Ranks | Mean seconds | Candidate | Effective GOP/s | Effective GB/s | Note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| replicated | Op-BHA | 25 | 1 | 1 | 2.18e-1 | 8388607 | 2.23 | 49.8 | binary zeta transform |
| replicated | Op-BBPA | 25 | 1 | 1 | 2.42e-1 | 8388607 | 2.84 | 48.2 | binary exact BBPA transform |
| replicated | Op-BBPA | 12 | 2 | 1 | 6.09e-2 | 4095 | 2.48 | 15.5 | multinomial exact transform with two-variant fast path |
| distributed | Op-BBPA | 12 | 2 | 2 | 3.18e-2 | 4095 | 2.38 | 14.8 | partitioned histogram, reduce-to-root transform |
| distributed | Op-BHA | 25 | 1 | 2 | 2.78e-1 | 8388607 | 1.75 | 38.1 | partitioned-lattice collective primitive |
| distributed | Op-BBPA | 25 | 1 | 2 | 2.74e-1 | 8388607 | 2.39 | 41.1 | binary partitioned BBPA transform |

The large multinomial selector is the important new result: before the exact-transform BBPA path, `subjects=12, variants=2` did not finish within a 10-second timeout on this machine. The optimized replicated selector now runs in about `6.1e-2 s`, and the two-rank partitioned-lattice version runs in about `3.2e-2 s`.

The intended tree scheduler was also smoke-tested through the public CLI:

```bash
mpirun -np 4 build/bin/bgt_simulate --mode parallel_dynamic_tree \
  --lattice replicated_non_dilution --subjects 12 --variants 1 \
  --prior 0.03 --search-depth 2 --selector op_bha \
  --provider cpu --workload-granularity 16
```

That run used `parallel_dynamic_tree`, evaluated all `4096` true states, and completed through the wildcard P2P master/worker scheduler.

## Ablations And Changes

### True Op-BHA zeta transform

Implemented `Lattice::op_bha_serial()` as a subset zeta transform over posterior masses. This preserves the Op-BHA objective: choose the experiment whose positive-test probability is closest to 0.5. Binary BGT now uses this path through `select_experiment(SelectorType::op_bha)` and through `auto_select`.

Outcome: large win for binary BGT selector calls, and this is the right single-core kernel for dynamic P2P tree simulation.

### Partitioned-lattice Op-BHA collective primitive

For partitioned posterior storage, distributed Op-BHA now reduces the scattered posterior vector to rank 0, computes the zeta transform once, and broadcasts only the chosen candidate. This is still a collective primitive, not the dynamic tree scheduling model.

Outcome: avoids running the zeta transform on every rank after reduction. Local two-rank timings vary on macOS, so this needs cluster measurement before claiming a final percentage.

### Exact-transform Op-BBPA

Binary BBPA now uses the same positive-mask zeta transform as Op-BHA and scores the two exact response buckets. Multinomial BBPA builds variant-subset intersection histograms and uses inclusion-exclusion to recover exact response masses. This is mathematically equivalent to the original brute-force BBPA objective but avoids the `O(states * experiments)` sweep.

Outcome: `subjects=12, variants=2` moved from a 10-second timeout to about `6.1e-2 s` replicated and `3.2e-2 s` with two partitioned ranks on this Mac. The current implementation includes a direct `k=2` histogram/scoring fast path because this is the common binary-plus-one-variant multinomial case.

### Distributed BBPA reduce-to-root

Distributed BBPA previously used full-table `MPI_Allreduce`, then every rank redundantly scanned the complete experiment/response table. It now reduces the transform feature table to rank 0, scans once, and broadcasts the candidate.

Outcome: tiny local two-rank cases are roughly neutral, but the communication and redundant scan pattern is better for large rank counts because only rank 0 needs the full reduced table result.

### OpenMP ablation

OpenMP improved large replicated Op-BHA on this 12-core Mac: `subjects=22`, one rank, improved from about `3.39e-2 s` to about `2.04e-2 s` with 12 threads. However, this is not the production scaling model for distributed runs.

The distributed BBPA auto-dispatch no longer uses OpenMP array reductions. In testing, `OMP_NUM_THREADS=4` changed the selected multinomial candidate on the tiny distributed case and was slower. Distributed selector dispatch now prefers single-core/SIMD per rank plus MPI collectives.

## Current Bottlenecks

- Op-BHA is memory-bandwidth bound after the zeta-transform improvement.
- Op-BBPA is now dominated by feature-table histogram writes, zeta-transform memory traffic, and inclusion-exclusion reads.
- Distributed-lattice selectors still require large reduction payloads proportional to the posterior/partition table size.
- Tree-level dynamic scheduling should be preferred for binary BGT because it avoids forcing every rank into every selector call.
- Further meaningful gains should come from JIT-specialized kernels, cache tiling for BBPA, GPU/NCCL provider work, and better task granularity policies, not from more OpenMP threading inside MPI ranks.

## Cluster Follow-Up

On the deployment system, rerun:

```bash
mpirun -np 1 build/bin/selector_roofline_benchmark --model=replicated --selector=op_bha --subjects=22 --variants=1 --iterations=20 --warmup=5
mpirun -np 2 build/bin/selector_roofline_benchmark --model=distributed --selector=op_bha --subjects=22 --variants=1 --iterations=20 --warmup=5
mpirun -np 4 build/bin/selector_roofline_benchmark --model=distributed --selector=op_bbpa --subjects=8 --variants=2 --iterations=10 --warmup=3
```

Also run full tree simulations with `parallel_dynamic_tree` and binary `op_bha` to measure the intended wildcard P2P scheduling path, varying `--workload-granularity`.
