# CUDA Provider Design Notes

The CUDA provider keeps the public `bgt::run_simulation(config)` API unchanged and accelerates the current tree path behind `Provider::cuda` or `Provider::auto_select`.

## Data Flow

1. Copy the prior vector to device memory and initialize the root posterior.
2. Select an experiment:
   - binary lattices with `SelectorType::op_bha` use a subset-lattice zeta transform,
   - multinomial or forced brute-force selection uses BBPA response-bucket masses.
3. For each response branch, update and normalize the posterior.
4. Classify atoms from one device atom-mass pass.
5. Build the host tree skeleton while reusing one device posterior buffer per depth.
6. Flatten the tree into contiguous device nodes.
7. Traverse the flattened tree on device, one true state per thread.
8. Reduce tree statistics across ranks with ordinary NCCL by default, or NCCL GIN when explicitly enabled.

## Workspace Ownership

`CudaWorkspace` owns reusable device buffers for posterior depth slots, reduction scratch, best-candidate reductions, atom masses, masks, tree nodes, priors, dilution tables, and tree statistics. Raw pointers are passed only at kernel, CUDA, and NCCL call boundaries.

Depth-indexed posterior buffers avoid repeated `cudaMalloc`/`cudaFree` during recursive tree construction. A child depth buffer can be reused for the next sibling after that sibling's subtree has been constructed, while the parent depth buffer remains intact during recursion.

## Distributed CUDA

MPI is used only for process bootstrap, rank/world discovery, and broadcasting the NCCL unique ID. Numerical reductions in the CUDA provider should use NCCL, not MPI reductions.

Ordinary NCCL collectives are the default distributed CUDA path. `CompileOptions.enable_nccl_gin` is a runtime opt-in. When it is enabled and compiled, the provider creates an NCCL device communicator and registers symmetric memory for a reusable GIN reduction workspace.

If GIN setup or execution fails under `Provider::cuda`, the error is surfaced. Under `Provider::auto_select`, the provider retries the same distributed CUDA run with ordinary NCCL.
