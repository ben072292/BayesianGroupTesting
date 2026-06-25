# BayesianGroupTesting

BayesianGroupTesting is a C++20 library for Bayesian group testing with binary
and multinomial response models. It represents all model families through one
lattice engine:

- Binary group testing is the `variants = 1` case.
- Multinomial group testing is the `variants > 1` case.
- Dilution and non-dilution response models use the same state lattice.
- Dynamic experiment selection and exact posterior partitioning share the same
  runtime API.

The library is intended for high-performance simulation and experiment design:
posterior updates, adaptive test selection, tree simulation, parallel true-state
evaluation, and CPU/CUDA provider dispatch for Python and native C++.

## Bayesian Group Testing

Group testing pools subjects into one experiment. A test response gives
information about every subject in the pool, so one test can classify many
subjects when prevalence is low or when the posterior is already concentrated.

Bayesian group testing keeps a posterior distribution over possible hidden
states. After every experiment:

1. Choose a pool of subjects.
2. Observe a response.
3. Update the posterior by Bayes' rule using the response model.
4. Mark subjects as positive, negative, or still uncertain using posterior
   thresholds.
5. Repeat until the stopping rule or tree depth is reached.

This project treats experiment selection as posterior partitioning. A good test
splits posterior mass across possible responses as evenly as possible, because
balanced responses are more informative.

## Unified Lattice Model

The core object is `bgt::Lattice`. It stores a posterior over encoded states and
provides posterior mass queries, test selection, Bayesian updates, and
classification metadata.

For `subjects = n` and `variants = k`, an encoded state has `n * k` atoms. The
same representation covers:

- `k = 1`: binary positive/negative group testing.
- `k > 1`: multinomial or multi-response testing, where each pooled test can
  produce a response mask over variants.

Priors are passed as one probability per subject-variant atom. The exact encoded
state type is selected at configure time:

```bash
-DBGT_STATE_BITS=8   # uint8_t
-DBGT_STATE_BITS=16  # uint16_t
-DBGT_STATE_BITS=32  # uint32_t, default
-DBGT_STATE_BITS=64  # uint64_t
```

Use the smallest width that can represent `subjects * variants` atoms.

## Response Models

BayesianGroupTesting supports two response families.

Non-dilution models use fixed true-positive and false-positive behavior for a
pooled response. This is the usual model when the test response is assumed not
to degrade with pool size.

Dilution models make the response probability depend on the pool size and the
number of positive subjects in the pool. `bgt::DilutionTable` owns the table used
by posterior updates and tree simulation.

```cpp
bgt::DilutionTable dilution(subjects, alpha, h);
lattice.update(experiment, response, &dilution);
```

## Experiment Selection

The selector controls how the next experiment is chosen.

- `bgt::SelectorType::op_bha`: optimized Bayesian halving. This is the dynamic
  selector for binary-style adaptive trees.
- `bgt::SelectorType::op_bbpa`: exact Bayesian posterior partitioning. This is
  the general partitioning path and also covers `k = 1`.
- `bgt::SelectorType::brute_force`: request the exact BBPA objective; optimized
  providers may use mathematically equivalent transforms instead of a literal
  experiment-by-state sweep.
- `bgt::SelectorType::auto_select`: resolves to Op-BHA for binary lattices and
  Op-BBPA for multinomial lattices.

For binary runs, Op-BHA is usually the natural adaptive selector. For
multinomial runs, or when an exact partitioning search is requested, Op-BBPA is
the general path.

## Tree Simulation

Tree simulation evaluates an adaptive testing policy over possible true states.
It reports aggregate quantities such as:

- total leaves,
- correct and incorrect classification probability,
- false positive and false negative probability,
- unclassified probability,
- expected stages,
- expected tests.

The C++ result type is `bgt::TreeStats`; the Python result exposes the same
fields.

## Runtime Providers

`bgt::Runtime` is the provider and specialization boundary. It builds a
`bgt::KernelSpec` from:

- provider: CPU, CUDA, or auto,
- subjects and variants,
- state width,
- dilution mode,
- selector,
- precision,
- CUDA architecture when applicable.

Provider policy:

- `bgt::Provider::auto_select`: try CUDA when compiled and available, otherwise
  fall back to CPU.
- `bgt::Provider::cpu`: force CPU.
- `bgt::Provider::cuda`: force CUDA and throw if unavailable or unsupported.

The runtime has an in-memory and disk cache surface for `KernelSpec` entries.
Native CPU/CUDA template specialization and loadable JIT artifacts are the next
runtime layer; the generic providers remain the correctness fallback.

## C++ API

Native users include one public header:

```cpp
#include "bgt/bgt.hpp"
```

The public API lives in namespace `bgt` and uses C++ RAII types:

- `bgt::Lattice`
- `bgt::Runtime`
- `bgt::KernelSpec`
- `bgt::DilutionTable`
- `bgt::SimulationConfig`
- `bgt::SimulationResult`
- `bgt::TreeStats`

### Binary Example

```cpp
#include "bgt/bgt.hpp"

#include <vector>

int main()
{
    std::vector<double> prior{0.2, 0.4};

    bgt::Lattice lattice(
        bgt::LatticeType::replicated_non_dilution,
        2,
        prior);

    bgt::state_t experiment = lattice.select_experiment(
        bgt::SelectorType::op_bha);
    lattice.update(experiment, 1);

    bgt::SimulationOptions options;
    options.provider = bgt::Provider::cpu;
    options.selector = bgt::SelectorType::op_bha;
    options.search_depth = 1;

    bgt::SimulationConfig config;
    config.lattice_type = bgt::LatticeType::replicated_non_dilution;
    config.subjects = 2;
    config.variants = 1;
    config.prior = prior;
    config.options = options;

    bgt::SimulationResult result = bgt::run_simulation(config);
    bgt::TreeStats stats = result.stats;

    (void)stats;
    return 0;
}
```

### Multinomial Example

```cpp
#include "bgt/bgt.hpp"

#include <vector>

int main()
{
    constexpr int subjects = 2;
    constexpr int variants = 3;
    std::vector<double> prior{
        0.01, 0.02, 0.03,
        0.04, 0.05, 0.06};

    bgt::Lattice lattice(
        bgt::LatticeType::replicated_non_dilution,
        subjects,
        variants,
        prior);

    bgt::state_t experiment = lattice.select_experiment(
        bgt::SelectorType::op_bbpa);
    bgt::state_t response = 0b101;
    lattice.update(experiment, response);
}
```

## Python API

The Python package is `bayesian_group_testing` and is backed by nanobind.

```python
import numpy as np
import bayesian_group_testing as bgt

options = bgt.SimulationOptions()
options.provider = bgt.Provider.cpu
options.selector = bgt.SelectorType.op_bha
options.search_depth = 1
options.compile_options.enable_simd = True
options.compile_options.enable_openmp = False
options.compile_options.optimization = bgt.OptimizationLevel.release

config = bgt.SimulationConfig()
config.lattice_type = bgt.LatticeType.replicated_non_dilution
config.subjects = 1
config.variants = 1
config.prior = np.array([0.2], dtype=np.float64)
config.options = options

result = bgt.run_simulation(config)
stats = result.stats

assert stats.total_leaves == 2
```

Top-level Python helpers include:

- `bgt.run_simulation`
- `bgt.select_experiment`
- `bgt.update`
- `bgt.available_providers`
- `bgt.jit_cache_info`
- `bgt.clear_jit_cache`

## Parallel Execution And CUDA

Parallel tree modes are exposed through framework-agnostic names such as
`parallel_dynamic_tree` and `parallel_global_tree`. The current backend uses MPI
for distributed execution and uneven true-state partitions. Worker ranks request
work dynamically, allowing first-come-first-served scheduling over state tasks.
For binary Op-BHA tree simulations, `parallel_dynamic_tree` is the intended
large-scale mode: rank 0 dispatches true-state chunks with wildcard P2P receives,
and each worker runs the local Op-BHA selector on its assigned tree work.

The unified command-line entrypoint is `bgt_simulate`:

```bash
mpirun -np 4 ./build/dev/bin/bgt_simulate \
  --mode parallel_dynamic_tree \
  --lattice replicated_non_dilution \
  --subjects 8 \
  --prior 0.02 \
  --search-depth 2 \
  --workload-granularity 4
```

Long tree simulations can opt into structured progress output:

```bash
./build/bin/bgt_simulate \
  --mode local_tree \
  --lattice replicated_non_dilution \
  --subjects 12 \
  --prior 0.02 \
  --search-depth 10 \
  --progress \
  --progress-interval 5 \
  --write-csv \
  --output-dir runs \
  --run-name n12-depth10
```

When progress is enabled, rank 0 writes the latest snapshot to
`<run-name>.progress.json` and appends history to `<run-name>.progress.jsonl`.
The CLI also prints progress to stderr unless `--no-progress-stderr` is set.
The first `SIGINT` or `SIGTERM` requests graceful cancellation: workers finish
their current true-state evaluation, final partial reductions run, and the CSV
report is written as `<run-name>.partial.csv`. Partial metrics are exact
coverage-weighted aggregates over evaluated true states only; use
`evaluated_states`, `state_coverage_fraction`, and `evaluated_prior_mass` to
interpret completeness.

CUDA support is optional and lives behind the runtime provider API. CUDA kernels
are organized under `src/kernels/cuda/` by purpose: posterior initialization,
Op-BHA/BBPA mass evaluation, posterior update, normalization, classification,
and tree-statistic traversal. Multi-rank CUDA `parallel_global_tree` runs use
NCCL collectives for posterior denominators, experiment-selection masses, atom
classification masses, and final tree-statistic reductions. MPI is still used
to initialize ranks and broadcast the NCCL unique id.

CUDA builds require a CUDA compiler:

```bash
cmake -S . -B build/cuda \
  -DBUILD_TESTING=ON \
  -DBGT_ENABLE_CUDA=ON \
  -DBGT_ENABLE_NCCL=ON

cmake --build build/cuda -j --target cuda_tree_parity_test
ctest --test-dir build/cuda --output-on-failure --timeout 60 -R cuda_tree_parity_test
```

The CUDA/NCCL hardware validation checklist is maintained in
[`docs/GPU_TEST_PLAN.md`](docs/GPU_TEST_PLAN.md), and the provider data flow is
summarized in [`docs/CUDA_PROVIDER_DESIGN.md`](docs/CUDA_PROVIDER_DESIGN.md).
CPU selector roofline notes and Op-BHA/Op-BBPA ablations are documented in
[`docs/CPU_SELECTOR_BENCHMARKS.md`](docs/CPU_SELECTOR_BENCHMARKS.md), and the
selector algorithm/comparison note is
[`docs/SELECTOR_ALGORITHMS.md`](docs/SELECTOR_ALGORITHMS.md).

## Repository Layout

```text
BayesianGroupTesting/
├── CMakeLists.txt
├── include/bgt/                  # Public C++ API
├── src/core/                     # Lattice, tree, probability, and runtime implementation
├── src/kernels/cuda/             # CUDA provider kernels
├── src/bindings/python/          # nanobind module
├── python/bayesian_group_testing/
├── tests/
│   ├── cpp/                      # Unit, integration, e2e, regression, and support tests
│   ├── python/                   # pytest suites using matching categories
│   └── fixtures/
├── benchmarks/
├── examples/
├── cmake/
├── docs/
└── pyproject.toml
```

## Build

### C++

```bash
cd /Users/weicongchen/Desktop/BGT/BayesianGroupTesting

cmake -S . -B build/dev \
  -DBUILD_TESTING=ON \
  -DBGT_ENABLE_CUDA=OFF \
  -DBGT_ENABLE_SIMD=ON \
  -DBGT_ENABLE_OPENMP=OFF \
  -DBGT_ENABLE_NATIVE_CPU=ON \
  -DBGT_ENABLE_FAST_MATH=ON \
  -DBGT_PROFILE_BACKEND=none \
  -DBGT_OPTIMIZATION_LEVEL=release \
  -DBGT_STATE_BITS=32

cmake --build build/dev -j
ctest --test-dir build/dev --output-on-failure --timeout 60
```

On macOS with AppleClang, CMake automatically uses Homebrew `libomp` from
`/opt/homebrew/opt/libomp` or `/usr/local/opt/libomp` when the stock OpenMP
probe fails.

Compilation knobs are intentionally feature-oriented rather than algorithm
variant-oriented. The native build and Python/JIT runtime share the same model:

- `BGT_ENABLE_SIMD` / `CompileOptions.enable_simd`
- `BGT_ENABLE_OPENMP` / `CompileOptions.enable_openmp`
- `BGT_ENABLE_CUDA` / `CompileOptions.enable_cuda`
- `BGT_ENABLE_NATIVE_CPU` / `CompileOptions.native_cpu`
- `BGT_ENABLE_FAST_MATH` / `CompileOptions.fast_math`
- `BGT_PROFILE_BACKEND` / `CompileOptions.profile_backend`
- `BGT_OPTIMIZATION_LEVEL` / `CompileOptions.optimization`

`KernelSpec` includes `CompileOptions`, so JIT cache entries are separated by
the requested compilation feature set.

Profiling is opt-in and uses external tooling:

- `BGT_PROFILE_BACKEND=none`: no profiling hooks in library scopes.
- `BGT_PROFILE_BACKEND=caliper`: emits Caliper annotations for HPC traces.
  This requires Caliper to be installed and discoverable by CMake.

Use NVIDIA Nsight Systems/Compute for CUDA kernel and GPU-memory profiling.
Use `selector_roofline_benchmark` for quick selector-level model estimates, and
use Google Benchmark or platform profilers for publishable kernel measurements.

### Python

```bash
python -m pip install .
python -m pytest tests/python
```

The Python wheel uses `scikit-build-core` and `nanobind`.

## Tests

The C++ suite covers:

- unit tests for state/model math, lattice behavior, dilution, and compile options,
- integration tests for local simulation, runtime/JIT cache behavior, and parallel tree modes,
- e2e tests for the `bgt_simulate` CLI and report generation,
- regression tests for deterministic BGT/BMGT numerical cases,
- CUDA parity when CUDA is enabled.

The Python suite mirrors those categories with pytest markers: `unit`,
`integration`, `e2e`, `regression`, `parallel`, `cuda`, and `slow`.

## Research Lineage

The project follows four connected lines of work.

Bayesian group testing with dilution effects gives the statistical model:
posterior inference over hidden infection states, response probabilities that
can depend on pool size, and adaptive test selection under uncertainty.

HiBGT focuses on making Bayesian group testing practical for high-performance
simulation. It accelerates posterior updates, experiment selection, and tree
evaluation for binary disease surveillance workloads.

SBGT scales the binary Bayesian group testing workflow with distributed
execution. It uses parallel tree simulation to evaluate adaptive policies across
many possible true states.

SBMGT extends the same Bayesian idea to multinomial responses. Instead of a
separate implementation boundary, this repository treats the multinomial case
as the same lattice model with `variants > 1`; binary BGT is simply the
`variants = 1` specialization.

## Publications

Please cite the relevant papers when using this code or its algorithms.

```bibtex
@inproceedings{chen_hibgt_2022,
  title = {{HiBGT}: {High}-{Performance} {Bayesian} {Group} {Testing} for {COVID}-19},
  author = {Chen, Weicong and Tatsuoka, Curtis and Lu, Xiaoyi},
  booktitle = {2022 {IEEE} 29th {International} {Conference} on {High} {Performance} {Computing}, {Data}, and {Analytics} ({HiPC})},
  pages = {176--185},
  year = {2022},
  publisher = {IEEE},
  address = {Bengaluru, India},
  doi = {10.1109/HiPC56025.2022.00033},
  url = {https://ieeexplore.ieee.org/document/10106329/}
}

@inproceedings{chen_sbgt_2023,
  title = {{SBGT}: {Scaling} {Bayesian}-based {Group} {Testing} for {Disease} {Surveillance}},
  author = {Chen, Weicong and Qi, Hao and Lu, Xiaoyi and Tatsuoka, Curtis},
  booktitle = {2023 {IEEE} {International} {Parallel} and {Distributed} {Processing} {Symposium} ({IPDPS})},
  pages = {951--962},
  year = {2023},
  doi = {10.1109/IPDPS54959.2023.00099},
  url = {https://ieeexplore.ieee.org/document/10177490}
}

@article{tatsuoka_bayesian_2022,
  title = {Bayesian group testing with dilution effects},
  author = {Tatsuoka, Curtis and Chen, Weicong and Lu, Xiaoyi},
  journal = {Biostatistics},
  pages = {kxac004},
  year = {2022},
  doi = {10.1093/biostatistics/kxac004},
  url = {https://doi.org/10.1093/biostatistics/kxac004}
}

@inproceedings{chen_sbmgt_2025,
  title = {{SBMGT}: {Scaling} {Bayesian} {Multinomial} {Group} {Testing}},
  author = {Chen, Weicong and Qi, Hao and Tatsuoka, Curtis and Lu, Xiaoyi},
  booktitle = {Proceedings of the 30th {ACM} {SIGPLAN} {Annual} {Symposium} on {Principles} and {Practice} of {Parallel} {Programming}},
  series = {{PPoPP} '25},
  pages = {512--523},
  year = {2025},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  doi = {10.1145/3710848.3710861},
  url = {https://dl.acm.org/doi/10.1145/3710848.3710861}
}
```
