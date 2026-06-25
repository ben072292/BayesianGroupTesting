# Op-BHA And Op-BBPA Selector Algorithms

This note documents the selector kernels used by BayesianGroupTesting and compares
them with the original research prototypes. It focuses on exact experiment
selection, not posterior update or tree-statistic evaluation.

The local `bayesgrouptestbackend` checkout was empty when this note was written,
so the original BGT Op-BHA discussion is based on the research-prototype design:
Java/Spark binary Bayesian group testing with halving-style subset pruning. The
original BMGT Op-BBPA discussion is source-checked against
`../Multinomial-Group-Testing/java` and `../Multinomial-Group-Testing/cpp`.

## Notation

| Symbol | Meaning |
| --- | --- |
| `n` | current active subject count after shrinking |
| `k` | variant count; binary BGT is `k = 1` |
| `A = n * k` | active atom count |
| `2^A` | posterior state count |
| `2^n` | candidate experiment count |
| `2^k` | response bucket count |
| `p` | MPI rank count |
| `E` | candidate experiment, represented as an `n`-bit subject mask |
| `S_v` | subject mask that is positive for variant `v` in a posterior state |

An experiment `E` is positive for variant `v` when every subject in `E` is
positive for that variant:

```text
(E & S_v) == E
```

For binary BGT (`k = 1`), this reduces to the subset relation `E subset S`.

## Binary Op-BHA

Op-BHA chooses the binary experiment whose positive-test probability is closest
to `0.5`.

For each experiment `E`, define the upset mass:

```text
M(E) = sum posterior[S] over all states S such that E subset S
```

The selected experiment is:

```text
argmin_E |M(E) - 0.5|
```

### Worked Example

For `n = 3`, suppose the posterior over binary states is:

```text
state  mass
000    0.10
001    0.10
010    0.20
011    0.10
100    0.05
101    0.05
110    0.20
111    0.20
```

For experiment `E = 010`, the positive states are those containing subject 1:

```text
010, 011, 110, 111
```

So:

```text
M(010) = 0.20 + 0.10 + 0.20 + 0.20 = 0.70
```

For experiment `E = 101`, the positive states are:

```text
101, 111
```

So:

```text
M(101) = 0.05 + 0.20 = 0.25
```

The selector scans all `E` and chooses the mask whose `M(E)` is closest to
`0.5`.

### Why `subset_zeta_transform` Is `O(n * 2^n)`

`subset_zeta_transform` computes every `M(E)` in-place from the posterior array.
For each subject bit, it adds the mass of states with that bit set into the
corresponding state with that bit cleared.

For `n = 3`, the first bit sweep performs:

```text
mass[000] += mass[001]
mass[010] += mass[011]
mass[100] += mass[101]
mass[110] += mass[111]
```

The next sweeps do the same for bits 1 and 2. After all sweeps:

```text
mass[E] = sum mass[S] for all S containing E
```

Each bit sweep performs `2^(n - 1)` additions. There are `n` sweeps, so the
exact count is:

```text
n * 2^(n - 1) additions = O(n * 2^n)
```

The final best-experiment scan is `O(2^n)`, so serial Op-BHA is
`O(n * 2^n)` work and `O(2^n)` memory.

### Original Op-BHA Prototype

The original BGT prototype used binary halving logic with subset-lattice
pruning. The monotonic property is:

```text
if E subset F, then M(F) <= M(E)
```

That gives two pruning rules:

```text
if M(E) < 0.5, every superset of E is farther from 0.5 and can be pruned
if M(E) > 0.5, every subset of E is farther from 0.5 and can be pruned
```

This can visit far fewer than `2^n` candidate experiments when the posterior
mass moves away from `0.5` quickly. Its worst-case behavior is still exponential
in `n`, and the actual cost depends on how candidate masses are computed and
cached.

Let:

```text
V_pruned = number of candidate experiments visited by the pruned search
C_mass   = cost to evaluate M(E) for one candidate
```

Then the serial cost is:

```text
O(V_pruned * C_mass), with V_pruned <= 2^n
```

If `M(E)` is computed by scanning the posterior for each candidate, then
`C_mass = O(2^n)` for binary BGT and the worst case is `O(4^n)`. If a dense
upset-mass table is precomputed, then mass lookup is `O(1)` after
`O(n * 2^n)` preprocessing, which is the approach used by the current C++
implementation.

### Original Distributed Op-BHA Prototype

The original Java/Spark-style distributed model is best understood as a
replicated-posterior task-parallel search. The lattice/posterior is broadcast to
workers, workers evaluate assigned candidate or tree-search tasks, and the
driver reduces the best candidate or intermediate task results.

With `p` workers and ignoring Spark scheduler overhead:

```text
per-worker compute: O((V_pruned / p) * C_mass)
worker memory:      O(2^n) replicated binary posterior
communication:      broadcast posterior plus reductions of task results
```

This is different from the current partitioned-lattice collective primitive,
where every rank owns a disjoint posterior slice and rank 0 reduces a dense
`2^n` mass table. It is also different from the current `parallel_dynamic_tree`
mode, which uses MPI point-to-point work dispatch for true-state tree tasks and
runs local Op-BHA inside each worker.

### Current Op-BHA

The current implementation deliberately uses a dense exact transform:

- `src/kernels/cpu/bbpa.cpp::Lattice::op_bha_serial`
- `src/kernels/cpu/bbpa.cpp::subset_zeta_transform`

This path does not rely on branch-heavy pruning. It has predictable memory
access, deterministic `O(n * 2^n)` work, and is a good single-core kernel for
tree workers.

The distributed-lattice primitive is:

- `src/kernels/cpu/bbpa.cpp::DistributedLattice::op_bha_mpi`

Each rank contributes a dense `2^n` local positive-mask table. Rank 0 reduces
the tables, performs the zeta transform and scan, then broadcasts one selected
experiment.

Distributed-lattice Op-BHA costs:

```text
per-rank local state pass: O(2^n / p) for binary partitioned posterior
communication:             O(2^n) accumulator values reduced to rank 0
rank-0 transform and scan:  O(n * 2^n)
memory per rank:            O(2^n)
```

This is a collective primitive for partitioned posterior storage. It is not the
same thing as `parallel_dynamic_tree`, where rank 0 dispatches true-state work
with point-to-point messages and each worker runs local Op-BHA on its assigned
tree task.

## Multinomial Op-BBPA

Op-BBPA chooses the experiment whose response-bucket probabilities are closest
to the target mass. For `k` variants there are `2^k` response buckets. The usual
target is:

```text
prob = 1 / 2^k
```

For an experiment `E`, BBPA scores:

```text
score(E) = sum_R |P(response R | E) - prob|
```

and returns the experiment with minimum score.

### Original Explicit Sweep

The original Java BMGT selector in
`ProductLatticeBitwiseBase.findHalvingStates` loops over:

```text
for each experiment E in 2^n:
  for each posterior state S in 2^(n*k):
    for each variant v in k:
      for each subject l in n:
        decide if E is positive for v
    add posterior[S] to one of 2^k response buckets
  score the response buckets
```

Asymptotic work:

```text
O(2^n * 2^(n*k) * k * n)
```

The original C++ BMGT selector replaced the innermost subject loop with bitwise
subset checks:

```text
(E & S_v) == E
```

That reduces the hot partition-id computation to roughly:

```text
O(2^n * 2^(n*k) * k)
```

Some C++ variants build the full experiment/response table and score it after
the state pass. Other variants compute one experiment at a time. In all cases,
the defining cost is the explicit experiment-by-state sweep.

### Original Distributed Op-BBPA

The original replicated MPI selector divides the experiment range across ranks.
Every rank keeps the full posterior and scores only its assigned experiments:

```text
per-rank work:   O((2^n / p) * 2^(n*k) * k)
per-rank memory: O((2^n / p) * 2^k) response masses plus full posterior
communication:  O(1) custom best-result Allreduce
```

This scales the candidate scan, but it replicates the posterior on every rank.

The original distributed posterior model partitions posterior states across
ranks, builds a dense experiment/response table, and uses `MPI_Allreduce` on
that full table:

```text
per-rank work:   O((2^(n*k) / p) * 2^n * k)
per-rank memory: O(2^n * 2^k)
communication:  O(2^n * 2^k) accumulator values
```

That reduces posterior memory pressure but introduces a large collective payload.

## Current Exact-Transform Op-BBPA

The current implementation is exact BBPA, but it avoids the explicit
experiment-by-state sweep.

The main idea is to build feature tables for variant subsets. For every
non-empty subset of variants `B`, define:

```text
F_B(E) = mass of states where E is positive for every variant in B
```

Before the zeta transform, the code builds histograms by exact intersection
masks. For a posterior state:

```text
S_v = subject mask positive for variant v
intersection(B) = bitwise AND of S_v over v in B
```

Then the mass is added to:

```text
feature_mass[B, intersection(B)]
```

After applying `subset_zeta_transform` to each feature table:

```text
feature_mass[B, E] = F_B(E)
```

Exact response buckets are then reconstructed by inclusion-exclusion.

### `k = 2` Example

For two variants, only three non-empty feature tables are needed:

```text
F_0(E)   = mass where variant 0 is positive
F_1(E)   = mass where variant 1 is positive
F_01(E)  = mass where both variants are positive
```

The four response buckets for experiment `E` are:

```text
both positive:       F_01(E)
variant 1 only:      F_1(E) - F_01(E)
variant 0 only:      F_0(E) - F_01(E)
neither positive:    total - F_0(E) - F_1(E) + F_01(E)
```

This is why `k = 2` has a specialized fast path in
`build_two_variant_intersection_masses` and
`best_two_variant_partition_candidate`: the generic inclusion-exclusion loop is
not needed, and the state pass writes only three tables.

The complexity comes from separating state aggregation from experiment scoring.
There are `2^(2n)` posterior states because a state contains two `n`-bit masks:

```text
S0 = subjects positive for variant 0
S1 = subjects positive for variant 1
```

For each posterior state, the optimized pass performs constant work:

```text
H0[S0]        += mass
H1[S1]        += mass
H01[S0 & S1]  += mass
```

That state pass is therefore:

```text
O(2^(2n))
```

Each table has `2^n` entries. A subset zeta transform over one table costs
`O(n * 2^n)`, and `k = 2` needs exactly three tables:

```text
H0, H1, H01
```

So the transform work is:

```text
3 * O(n * 2^n) = O(n * 2^n)
```

After the transforms, each candidate experiment has constant-time bucket
formulas using `F_0(E)`, `F_1(E)`, and `F_01(E)`. Scanning all experiments is
`O(2^n)`, which is dominated by the transform term. The total optimized
`k = 2` BBPA cost is:

```text
O(2^(2n)) + O(n * 2^n) + O(2^n)
= O(2^(2n) + n * 2^n)
```

The avoided brute-force workflow would nest every experiment over every state:

```text
for each experiment E in 2^n:
  for each state S in 2^(2n):
    compute the response bucket for E against S
```

That explicit sweep is roughly `O(2^(3n))` before even counting per-variant
partition-id work. The zeta-transform workflow is the reason the current
`k = 2` selector is practical for larger `n`.

### Current Serial Complexity

For binary `k = 1`, BBPA degenerates to the same positive-mask transform used by
Op-BHA, then scores two response buckets:

```text
work:   O(n * 2^n)
memory: O(2^n)
```

For two variants `k = 2`:

```text
state pass:       O(2^(2n))
zeta transforms:  O(3 * n * 2^n)
candidate scan:   O(2^n)
memory:           O(4 * 2^n)
```

For generic `k > 1`:

```text
state pass:       O(2^(n*k) * 2^k)
zeta transforms:  O((2^k - 1) * n * 2^n)
candidate scan:   O(2^n * 3^k)
memory:           O(2^k * 2^n)
```

The `3^k` term comes from inclusion-exclusion. For each response, the code loops
over all submasks of that response's negative-variant set. Summed across all
responses, this is:

```text
sum over responses R of 2^popcount(R) = (1 + 2)^k = 3^k
```

For the project target cases where `k` is small, this is much cheaper than
sweeping every experiment against every posterior state.

### Current Distributed Complexity

The current partitioned-lattice implementation is:

- `src/kernels/cpu/bbpa.cpp::DistributedLattice::BBPA_mpi`

Each rank scans its posterior state partition and builds a dense feature table
for all experiments. Rank 0 reduces the feature tables, performs transforms and
candidate scoring, then broadcasts the selected experiment.

For `k > 1`:

```text
per-rank state pass:      O((2^(n*k) / p) * 2^k)
communication to rank 0:  O(2^k * 2^n)
rank-0 transform:         O((2^k - 1) * n * 2^n)
rank-0 candidate scan:    O(2^n * 3^k), or O(2^n) for k = 2
memory per rank:          O(2^k * 2^n)
```

This is better than the original partitioned explicit sweep because the
per-rank state pass no longer multiplies by `2^n` candidate experiments.
However, it still has a full-table reduction and a rank-0 transform/scan
bottleneck.

## Serial And Distributed Comparison

| Selector | Implementation | Exact objective | Work | Memory per rank | Communication | Main bottleneck |
| --- | --- | --- | --- | --- | --- | --- |
| Op-BHA | original BGT pruning, serial | yes | `O(V_pruned * C_mass)`, `V_pruned <= 2^n` | implementation dependent | none | branchy search and mass evaluation |
| Op-BHA | original BGT pruning, Spark-style distributed | yes | `O((V_pruned / p) * C_mass)` plus scheduler overhead | replicated binary posterior | broadcast plus task-result reductions | scheduler overhead and replicated posterior |
| Op-BHA | current serial | yes | `O(n * 2^n)` | `O(2^n)` | none | memory bandwidth in zeta transform |
| Op-BHA | current distributed lattice | yes | local `O(2^n / p)` fill plus rank-0 `O(n * 2^n)` | `O(2^n)` | `O(2^n)` reduce plus scalar broadcast | full-table collective and rank-0 transform |
| Op-BHA | current dynamic tree | yes per worker task | local serial selector per assigned true-state task | replicated per worker task | small P2P work messages and final stats | task granularity and single-core selector speed |
| Op-BBPA | original Java BMGT serial | yes | `O(2^n * 2^(n*k) * k * n)` | `O(2^k)` per candidate | none | explicit experiment/state/subject sweep |
| Op-BBPA | original C++ BMGT serial | yes | about `O(2^n * 2^(n*k) * k)` | `O(2^n * 2^k)` or `O(2^k)` depending variant | none | explicit experiment/state sweep |
| Op-BBPA | original replicated MPI | yes | `O((2^n / p) * 2^(n*k) * k)` | full posterior plus local response table | scalar best-result Allreduce | replicated posterior and state sweep |
| Op-BBPA | original distributed posterior MPI | yes | `O((2^(n*k) / p) * 2^n * k)` | `O(2^n * 2^k)` | `O(2^n * 2^k)` Allreduce | full-table collective |
| Op-BBPA | current serial, `k = 1` | yes | `O(n * 2^n)` | `O(2^n)` | none | memory bandwidth |
| Op-BBPA | current serial, `k = 2` | yes | `O(2^(2n) + n * 2^n)` | `O(4 * 2^n)` | none | posterior state pass |
| Op-BBPA | current serial, generic `k` | yes | `O(2^(n*k) * 2^k + (2^k - 1) * n * 2^n + 2^n * 3^k)` | `O(2^k * 2^n)` | none | feature-table writes and inclusion-exclusion |
| Op-BBPA | current distributed lattice | yes | state pass divided by `p`; rank-0 transform and scan unchanged | `O(2^k * 2^n)` | `O(2^k * 2^n)` reduce plus scalar broadcast | reduction payload and rank-0 work |

## Practical Guidance

- For binary tree simulation, prefer `SelectorType::auto_select` or
  `SelectorType::op_bha`. The default resolves to Op-BHA and uses the exact
  zeta-transform kernel.
- For multinomial tree simulation, Op-BHA is not valid because the response
  space has `2^k` buckets. Use Op-BBPA.
- Keep explicit brute-force BBPA as a benchmark/reference path only. The
  production selector should use exact transforms.
- For large-scale binary tree runs, the best near-term scaling model is
  tree-level work dispatch: rank 0 assigns true-state tasks first-come-first-
  serve, and workers run efficient local Op-BHA. This avoids forcing every rank
  through a dense distributed selector at every tree node.
- For truly partitioned multinomial posterior storage, the current distributed
  exact-transform BBPA is mathematically right, but the full feature-table
  reduction is the scaling limit. Future work should reduce batched candidate
  scores or tiled feature tables rather than reduce `O(2^k * 2^n)` data for
  every selection.
