# GPU Test Plan

This document records the CUDA/NCCL validation work to run when GPU hardware is available.
The local development machine used for the initial implementation did not have `nvcc`, so
the CPU/MPI build was verified locally and the CUDA path still needs hardware validation.

## Required Environments

Test at least these configurations:

- Single process, single GPU.
- Multi-process, single node, one GPU per rank.
- Multi-process, uneven rank count for tiny lattices, including ranks greater than state count.
- Multi-node if available, using the same MPI launcher expected in production.
- CUDA with NCCL collectives enabled: `BGT_ENABLE_CUDA=ON`, `BGT_ENABLE_NCCL=ON`.
- CUDA with NCCL GIN compiled when `nccl_device.h` and CUDA 12.2 or newer are available.
- Runtime GIN opt-in through `SimulationOptions.compile_options.enable_nccl_gin = true`.
- Posterior/accumulator precision matrix: `float64/float64`, `float32/float64`, and `float32/float32`.
- State width matrix: `BGT_STATE_BITS=8,16,32,64`.

## Build Checks

Clean configure/build commands:

```bash
cmake -S . -B build/cuda \
  -DBUILD_TESTING=ON \
  -DBGT_ENABLE_CUDA=ON \
  -DBGT_ENABLE_NCCL=ON

cmake --build build/cuda -j
```

Expected build behavior:

- `BGT_ENABLE_CUDA=ON` fails clearly when no CUDA compiler is available.
- `BGT_ENABLE_NCCL=ON` fails clearly when `nccl.h` or `libnccl` is unavailable.
- `BGT_ENABLE_NCCL=AUTO` builds CUDA without NCCL when NCCL is missing.
- `BGT_ENABLE_NCCL_GIN=ON` fails clearly unless CUDA 12.2+ and `nccl_device.h` are available.
- `BGT_ENABLE_NCCL_GIN=AUTO` never blocks ordinary NCCL collective support.
- Default `CompileOptions` leave `enable_nccl_gin=false`; ordinary NCCL is the default distributed CUDA path.
- Forced CUDA with `enable_nccl_gin=true` fails clearly when GIN is not compiled or setup fails.
- Auto provider selection with `enable_nccl_gin=true` falls back to ordinary NCCL when GIN setup fails.

## Correctness Tests

Run CUDA CTest targets first:

```bash
ctest --test-dir build/cuda --output-on-failure -L cuda
```

Then add or run targeted cases for:

- Replicated single-GPU tree parity against CPU for binary non-dilution.
- Replicated single-GPU tree parity against CPU for binary dilution.
- Replicated single-GPU tree parity against CPU for multinomial non-dilution.
- Binary `k=1` default selector uses Op-BHA and matches forced BBPA/brute-force statistics on small cases.
- Multinomial `k>1` uses BBPA/brute-force and rejects forced Op-BHA.
- Distributed CUDA `parallel_global_tree` parity against CPU parallel/global tree on tiny exhaustive cases.
- Distributed CUDA parity for world sizes `1,2,3,4` and uneven partitions.
- Distributed CUDA parity when `world_size > total_states`.
- Dilution and non-dilution distributed CUDA parity.
- Probability invariants: no NaN/Inf, nonnegative probabilities, and final statistic mass sums to expected totals within tolerance.
- Branch threshold behavior against CPU for `branch_threshold=0` and a positive threshold.
- Tree traversal overflow handling for deeper trees; either no overflow in supported cases or a clear error path.

## NCCL-Specific Checks

For multi-rank CUDA runs, verify that CUDA reductions use NCCL collectives:

- Op-BHA upset-mass reductions.
- BBPA partition-mass reductions.
- Posterior denominator reductions.
- Atom classification mass reductions.
- Final tree-statistic reductions.

Keep MPI limited to process bootstrap, rank/world discovery, and NCCL unique-id broadcast in the CUDA provider. A source scan should continue to show no `MPI_Allreduce` or `MPI_Reduce` in `src/kernels/cuda/cuda_provider.cu`.

When GIN is enabled, verify the full lifecycle:

- Host NCCL communicator is initialized once per rank.
- `ncclDevCommCreate` is called with CTA/signal counts matching the launched GIN grid.
- Symmetric reduction memory is allocated with `ncclMemAlloc`.
- The symmetric buffer is registered with `ncclCommWindowRegister(..., NCCL_WIN_COLL_SYMMETRIC)`.
- Reductions use the symmetric inbox layout `[local vector][world_size inbox vectors]`.
- Cleanup deregisters the window, frees symmetric memory, destroys the device communicator, and destroys the host communicator.
- GIN reductions match ordinary NCCL reductions exactly within the configured accumulator precision tolerance.

Suggested source check:

```bash
rg -n "MPI_Allreduce|MPI_Reduce" src/kernels/cuda/cuda_provider.cu
```

This should return no matches.

## Runtime And Python Tests

Run Python tests from a CUDA build/install:

```bash
pytest tests/python -m "cuda or integration or e2e"
```

Add Python coverage for:

- `bgt.run_simulation(config)` with `Provider.cuda`.
- `Provider.auto_select` selecting CUDA when CUDA/NCCL are available.
- `Provider.auto_select` falling back to CPU when CUDA or NCCL is unavailable.
- NumPy prior arrays for binary, multinomial, dilution, and non-dilution cases.
- Multi-rank invocation through `mpirun -np N python -m ...` for `parallel_global_tree`.
- JIT cache keys include CUDA, NCCL, GIN, state width, posterior precision, accumulator precision, selector, variants, and CUDA architecture.

## Performance And Profiling

Use NVIDIA tools rather than built-in timers:

- Nsight Systems for host/device overlap, NCCL collective timing, launch overhead, and MPI bootstrap overhead.
- Nsight Compute for kernel occupancy, memory throughput, atomics pressure, and branch efficiency.
- `nvidia-smi` or DCGM for memory footprint and device utilization.
- NCCL debug logs only when diagnosing collective setup: `NCCL_DEBUG=INFO`.

Performance cases should compare:

- CPU local tree vs single-GPU CUDA tree.
- CPU MPI `parallel_global_tree` vs NCCL CUDA `parallel_global_tree`.
- Op-BHA binary selection vs BBPA/brute-force binary selection.
- Float32 posterior storage vs float64 posterior storage.
- Varying subjects, variants, search depth, and branch threshold.

Record:

- End-to-end runtime.
- Kernel-level time for prior initialization, selection, posterior update, classification, and tree-stat traversal.
- NCCL collective time and payload size.
- Peak GPU memory.
- Number and size of CUDA allocations.

## Memory And Reliability Checks

Run sanitizers/debug tools where practical:

```bash
compute-sanitizer --tool memcheck ./build/cuda/bin/cuda_tree_parity_test
compute-sanitizer --tool racecheck ./build/cuda/bin/cuda_tree_parity_test
```

Check:

- No invalid global memory access.
- No leaked CUDA allocations across repeated simulation runs.
- No NCCL communicator or stream lifecycle leaks.
- No stale device data between tree branches.
- No race in atomic reductions.
- Stable results across repeated runs with the same inputs.

## Acceptance Criteria

The CUDA implementation is ready to trust when:

- All CUDA C++ and Python tests pass on single-GPU and multi-rank configurations.
- CUDA statistics match CPU reference statistics within precision-appropriate tolerances.
- Distributed CUDA matches CPU distributed or exhaustive local references on small deterministic cases.
- No CUDA provider path uses MPI reductions for numerical data movement.
- `Provider.auto_select` chooses CUDA only when the required CUDA/NCCL support is available.
- Forced CUDA reports clear errors when required support is missing.
- Nsight profiling shows the heavy work runs on GPU, with no unexpected full posterior-array host transfers in the inner tree loop.
