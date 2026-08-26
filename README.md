# MO_NCO Engineering Prototype

> This public repository contains the reviewed `V21E3R1_V9R2R1`
> engineering-reproduction candidate. Read
> [`docs/V9R2R1_PUBLIC_RELEASE_BOUNDARY.md`](docs/V9R2R1_PUBLIC_RELEASE_BOUNDARY.md)
> and [`docs/GITHUB_CHAT_ACCESS_SCOPE.md`](docs/GITHUB_CHAT_ACCESS_SCOPE.md)
> before interpreting any evidence. The canonical 203-file source closure and
> scoped engineering checks are reproducible; the public-checkout full suite is
> not closed because historical fixtures and large evidence artifacts are not
> publicly redistributable here. Repository-wide green, a complete artifact-
> hashed environment lock, scientific independence, open-science licensing,
> and every later-phase authorization are explicitly false.

## V9R2R1 engineering-reproduction entry point

- Canonical source identity: [`V21E3R1_V9R2R1_SOURCE_MANIFEST.json`](V21E3R1_V9R2R1_SOURCE_MANIFEST.json)
  (`203` files; source-tree SHA-256
  `50ad30da8670eb488848e6db084084185fea7725e86c7fea480639caa193d9eb`).
- Complete commands and mode semantics: [`docs/V9R2R1_RUNBOOK.md`](docs/V9R2R1_RUNBOOK.md).
- Public-release boundary: [`docs/V9R2R1_PUBLIC_RELEASE_BOUNDARY.md`](docs/V9R2R1_PUBLIC_RELEASE_BOUNDARY.md).
- Exact historical failure registry and top-level envelope: [`provenance/`](provenance/).
- Explicit included/prohibited/rights-ambiguous dependency disposition:
  [`provenance/V9R2R1_PUBLIC_CHECKOUT_DEPENDENCY_DISPOSITION.json`](provenance/V9R2R1_PUBLIC_CHECKOUT_DEPENDENCY_DISPOSITION.json).
- Final public-checkout validation receipt:
  [`provenance/V9R2R1_PUBLIC_CHECKOUT_VALIDATION_RECEIPT.json`](provenance/V9R2R1_PUBLIC_CHECKOUT_VALIDATION_RECEIPT.json).

The checked-in exact-eight JUnit is historical reference evidence. A fresh
public checkout is deliberately not claimed to reproduce the full private
artifact warehouse; run `FullLive` to test that boundary fail-closed. The final
public-tree observation was `78 failed, 1328 passed, 4 skipped, 267 subtests
passed`; all 78 outcomes are classified in the receipt and none authorizes a
scientific or publication phase.

This repository contains engineering implementations and falsification gates
for the theory in `theory_corrected(4).tex`.

The first implemented target is a bi-objective Euclidean TSP. It uses:

- a connected symmetric 2-opt proposal graph with city 0 fixed;
- an interacting-particle Metropolis sampler;
- a nondominated Pareto archive updated at explicit stopping times;
- a scalar archive potential, so Metropolis deltas are algebraically complete
  differences of the implemented binary64 Hamiltonian within each frozen
  archive epoch;
- diagnostics for acceptance rate, archive size, 2D hypervolume, empirical
  energy, and positive archive jumps.

## Run the demo

```powershell
C:\miniconda3\python.exe -m mo_nco.run_demo --cities 30 --particles 48 --iterations 2000 --seed 7
```

The archive is written to `outputs/archive.csv` by default.

## Run a benchmark

Synthetic benchmark:

```powershell
C:\miniconda3\python.exe -m mo_nco.run_benchmark `
  --algorithms ips-theory,ips,ips-neural,nsga2,moead,random2opt `
  --seeds 0,1,2 `
  --cities 30 `
  --population 48 `
  --iterations 2000 `
  --instance-seed 123 `
  --output-dir outputs\benchmark
```

TSPLIB-style bi-objective benchmark, where each objective is one `.tsp` file
over the same city ids:

```powershell
C:\miniconda3\python.exe -m mo_nco.run_benchmark `
  --algorithms ips-theory,ips,ips-neural,nsga2,moead,random2opt `
  --seeds 0,1,2,3,4 `
  --population 32 `
  --iterations 1500 `
  --tsplib-files benchmarks\tsplib\demo_obj1.tsp,benchmarks\tsplib\demo_obj2.tsp `
  --output-dir outputs\benchmark_tsplib_formal `
  --log-period 150 `
  --archive-update-period 25
```

Simple bi-objective coordinate CSV:

```powershell
C:\miniconda3\python.exe -m mo_nco.run_benchmark `
  --algorithms ips,ips-neural,nsga2,moead,random2opt `
  --seeds 0,1,2 `
  --bitsp-file benchmarks\bitsp\demo_bitsp.csv `
  --output-dir outputs\benchmark_bitsp
```

Benchmark outputs:

- `runs.csv`: one row per algorithm/seed.
- `summary.csv` and `summary.json`: mean/std/min/max statistics.
- `comparison.md`: publication-style comparison table sorted by mean
  hypervolume, including HV, anytime AUC, IGD+, additive epsilon, HV/sec, and
  HV/eval.
- `paired_comparison.md`: matched-seed deltas and exact two-sided sign tests.
- `anytime.csv`: archive-snapshot HV-vs-evaluation and HV-vs-time curve points
  under the same final common reference point.
- `pareto_fronts.svg`: dependency-free Pareto front visualization.
- `archives/*.csv`: final nondominated archive for each run.

The `iterations` argument is treated as a candidate-evaluation budget. The
benchmark runner wraps every instance in `CountingTSPInstance`, so
`runs.csv`, `summary.csv`, and `comparison.md` report the exact number of true
objective evaluations used by each run.

## Run a benchmark suite

```powershell
C:\miniconda3\python.exe -m mo_nco.run_suite `
  --suite benchmarks\suite_demo.json `
  --algorithms ips-theory,ips-theory-core,pymoo-nsga2,pymoo-moead,motsp-pls,moead,nsga2,random2opt `
  --seeds 0,1,2,3,4 `
  --population 32 `
  --evaluations 3000 `
  --output-dir outputs\suite_formal
```

Suite outputs include `aggregate_runs.csv`, one subdirectory per benchmark
case, and `suite_summary.md` with cross-case mean HV, AUC, and HV/sec ranks.

## Annealed Pareto-SMC / Feynman--Kac mainline

The new `annealed-pareto-smc` entry is a separate algorithmic object, not a
renaming of the historical zero-temperature batch heuristic. It uses a
predeclared manifest, continuous frozen-box augmented-Tchebycheff targets,
explicit inter-stage incremental weights, ESS-triggered multinomial resampling
within each fixed reference type, and symmetric 2-opt MH mutation using the
complete implemented binary64 acceptance ratio while the stage target is
frozen. The Pareto archive and the untruncated epsilon-cell
ledger are reporting observers and never feed back into the particle system.

```powershell
$env:MO_NCO_PARETO_SMC_SPEC = 'D:\MO_NCO\benchmarks\pareto_smc_v1_spec.json'
C:\miniconda3\envs\ssm_env\python.exe -m mo_nco.run_suite `
  --suite D:\MO_NCO\benchmarks\suite_theory_v2_smoke.json `
  --algorithms annealed-pareto-smc,ips-theory-certified `
  --seeds 0,1 `
  --population 32 `
  --evaluations 512 `
  --override-case-evaluations `
  --execution-order seed-major-balanced-v1 `
  --output-dir D:\MO_NCO\outputs\pareto_smc_smoke
```

The exact tiny-state audit recomputes the true finite Pareto front, final typed
target cell masses and \(p_{\min}\), then exercises the conditional
finite-particle-to-coverage-to-HV/IGD certificate:

```powershell
C:\miniconda3\envs\ssm_env\python.exe -m mo_nco.run_pareto_smc_audit `
  --spec D:\MO_NCO\benchmarks\pareto_smc_v1_spec.json `
  --output D:\MO_NCO\outputs\pareto_smc_v1_tiny_audit.json `
  --cities 5 `
  --instance-seed 20260726 `
  --algorithm-seed 0 `
  --population 32 `
  --evaluations 512
```

Without a source-bound finite-particle MSE certificate, the correct audit
result is `MECHANICS PASS / SCIENTIFIC_GATE UNRESOLVED`; ESS and implemented-
kernel MH invariance are deliberately not treated as concentration or finite-budget
coverage proofs. Run the full local validation with:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File `
  D:\MO_NCO\RUN_PARETO_SMC_V1_VALIDATION.ps1
```

The predeclared 35-case, four-arm, five-seed, 512-evaluation matched protocol
freezes an algorithm-disjoint metric-reference manifest from the completed
calibration output before launching any current arm. It then verifies the full
case × arm × seed matrix and runs case-clustered CI/W-T-L/trimmed/winsorized
and sign-flip adoption analysis:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File `
  D:\MO_NCO\RUN_PARETO_SMC_35CASE_4ARM_5SEED_EVAL512.ps1
```

Completion and adoption are separate: the launcher exits successfully after a
complete, mechanically valid run even when the strict analysis correctly
returns `REJECT`.

### Pareto-SMC v15 review-response branch

v15 separates two different random objects instead of transferring a
certificate between them:

- `AnnealedParetoSMCOptimizer` with more than one particle per type remains an
  interacting, resampled Feynman--Kac system;
- `run_independent_replica_batch` is a separate collection of typed annealed
  MH replicas with no population interaction or resampling.  Only this branch
  is consumed by the direct binomial/occupancy probability certificates.

The v15 modules also freeze a rational Cartesian cell manifest, use exact
dyadic edge sums and exact endpoint-cell classification, compute the ordinary
IGD and IGD+ reference average with the standard arithmetic mean, provide
exact-rational Clopper--Pearson bracketing and replica-count planning, and
propagate deterministic archive-cap and reference-fidelity errors.  The MH
energy and acceptance decision remain binary64 and are explicitly labelled
`not_machine_exact`; exact edge sums do not turn that kernel into exact-real
MH.

The current `evaluate_v15_publication_gate` is a component-level claim-hygiene
check, not yet a raw-artifact end-to-end verifier.  At the current repository
state, local finite-reference regression guards are testable, but the composed
P0 gate remains closed until every child artifact is bound to one case context
and recomputed from canonical raw inputs.  In addition,
the P1 theory gate remains closed because adaptive allocation, out-of-sample
finite-menu selection, and an intrinsic-dimension family certificate are
missing.  External future-beacon controls, study-level commitment, a zero-
`sorry` Lean probability core, and matched competitive evidence are also not
established.  Therefore the only authorized submission verdict is `HOLD`.
See `PARETO_SMC_V15_THEORY_AND_ALGORITHM.md` and
`PARETO_SMC_V15_RUNBOOK.md`.

### Pareto-SMC v13 staged publication protocol

v13 is retained as the historical staged protocol; v15 supersedes it for the
current v15 claim boundary but does not change the v11 or v12
benchmark aliases. The historical workflow is:

```python
from mo_nco import (
    load_v13_pilot_artifact,
    run_v13_confirm_from_signed_receipt,
    run_v13_pilot_freeze,
    write_v13_pilot_artifact,
)
```

The layer binds domain-separated pilot/confirm seeds, full-type-sweep
checkpoints, assignment-simplex feasibility, a minimum-refresh
certificate-cost design, exact v2 witness tours for both the anchor and full
finite reference sets, a canonical sparse-reference metric bridge, a
v2 self-hashed cross-process pilot artifact whose complete canonical pilot
result is receipt-bound, and an externally signed Ed25519
pre-confirm authorization. Ordinary IGD applies to the retained
one-witness-per-anchor support before nondominated filtering; IGD+ and
shifted-front HV apply to its nondominated view.

The signed timestamp is metadata, not an independent wall-clock proof, and the
finite sparse bridge is not a universal fixed-size archive theorem. See
`PARETO_SMC_V13_THEORY_AND_ALGORITHM.md` and
`PARETO_SMC_V13_RUNBOOK.md`. Competitive experiments and the systematic
literature review remain `NOT RUN`; submission remains `HOLD`, and neither
scalability nor state-of-the-art claims are authorized. Until an external
no-preview/launch-ordering record is bound, only the conditional certificate
content gate may pass; the formal and publication packet gates remain
`NOT_ESTABLISHED`.

Multi-budget sweeps:

```powershell
C:\miniconda3\python.exe -m mo_nco.run_budget_sweep `
  --suite benchmarks\suite_public_motsp_35.json `
  --algorithms ips-theory,ips-theory-core,pymoo-nsga2,pymoo-moead,motsp-pls `
  --seeds 0,1,2,3,4 `
  --budgets 512,1024,2048 `
  --output-dir outputs\budget_sweep_formal
```

Formal SOTA-gated MOTSP suite:

```powershell
$env:MO_NCO_BASELINE_PAQUETE="C:\path\to\real_paquete_adapter.exe"
$env:MO_NCO_BASELINE_TPLS="C:\path\to\real_tpls_adapter.exe"
$env:MO_NCO_BASELINE_MOGLS="C:\path\to\real_mogls_adapter.exe"

# Or use bridge templates for real binaries that consume TSPLIB/CSV matrices:
$env:MO_NCO_BRIDGE_PAQUETE="C:\path\to\paquete_wrapper.exe --obj0 {obj0_tsp} --obj1 {obj1_tsp} --seed {seed} --budget {evaluations} --out {output_csv} --diag {diagnostics_csv}"
$env:MO_NCO_BRIDGE_TPLS="C:\path\to\tpls_wrapper.exe --matrix-dir {matrix_dir} --seed {seed} --evals {evaluations} --out {output_csv} --diag {diagnostics_csv}"
$env:MO_NCO_BRIDGE_MOGLS="C:\path\to\mogls_wrapper.exe --input {input_json} --seed {seed} --budget {evaluations} --out {output_csv} --diag {diagnostics_csv}"

C:\miniconda3\python.exe -m mo_nco.run_formal_sota_suite `
  --suite benchmarks\suite_public_motsp_35.json `
  --seeds 0,1,2,3,4,5,6,7,8,9 `
  --budgets 512,1024 `
  --external-smoke-test `
  --output-dir outputs\formal_sota_suite
```

This runner refuses to make a SOTA-gated run unless the real Paquete/TPLS/MOGLS
commands are configured. Use `--preflight-only` to check solver availability,
or `--allow-missing-external` only for non-SOTA development runs. See
`RUN_35CASE_SOTA_COMMANDS.md` for the full 35+ case, 10-seed, multi-budget
runbook and the external solver output contract.

Train/test neural prior protocol:

```powershell
C:\miniconda3\python.exe -m mo_nco.run_train_neural_prior `
  --suite benchmarks\suite_public_motsp_35.json `
  --output outputs\neural_prior_split70_seed0\prior.json `
  --train-fraction 0.7 `
  --seed 0

$env:MO_NCO_NEURAL_PRIOR_PATH="D:\MO_NCO\outputs\neural_prior_split70_seed0\prior.json"
C:\miniconda3\python.exe -m mo_nco.run_budget_sweep `
  --suite outputs\neural_prior_split70_seed0\splits\public_pair_motsp_suite_test.json `
  --algorithms ips-offline-neural,ips-scalar-greedy,ips-theory,pymoo-moead,pymoo-nsga2 `
  --seeds 0,1,2,3,4,5,6,7,8,9 `
  --budgets 512 `
  --output-dir outputs\test_split_frozen_neural
```

## Run ablations

```powershell
C:\miniconda3\python.exe -m mo_nco.run_ablation `
  --seeds 0,1,2,3,4 `
  --population 32 `
  --evaluations 3000 `
  --tsplib-files benchmarks\tsplib\demo_obj1.tsp,benchmarks\tsplib\demo_obj2.tsp `
  --output-dir outputs\ablation_ips
```

The ablation runner compares the full theory-aligned low-temperature IPS
against changes to neighborhood size, proposal mixing, finite temperature,
archive-conditioned scalar fields, and archive-parent proposals.

Cross-instance ablations use the same public suite protocol:

```powershell
C:\miniconda3\python.exe -m mo_nco.run_ablation_suite `
  --suite benchmarks\suite_public_motsp_35.json `
  --seeds 0,1,2,3,4 `
  --population 32 `
  --evaluations 3000 `
  --output-dir outputs\ablation_suite_formal
```

To build a larger public MOTSP suite from public TSPLIB instances:

```powershell
C:\miniconda3\python.exe -m mo_nco.build_public_motsp_suite `
  --max-cases 35 `
  --population 32 `
  --evaluations 1024
```

The generated suite first pairs public TSPLIB TSP instances with identical
dimensions, then extends to Paquete-style public BOTSP objective pairs when
more cases are requested.

## Mature baseline adapters

Names beginning with `external-`, `pymoo-`, `jmetal-`, or `platemo-` are routed
through `mo_nco.mature_baselines.ExternalBaselineOptimizer`. Install optional
baseline dependencies with:

```powershell
C:\miniconda3\python.exe -m pip install -r requirements-optional.txt
```

`pymoo-nsga2` and `pymoo-moead` have built-in command adapters:

```powershell
C:\miniconda3\python.exe -m mo_nco.run_benchmark `
  --algorithms ips-theory,pymoo-nsga2,pymoo-moead,moead `
  --seeds 0,1 `
  --population 32 `
  --iterations 1000 `
  --tsplib-files benchmarks\tsplib\demo_obj1.tsp,benchmarks\tsplib\demo_obj2.tsp `
  --output-dir outputs\benchmark_with_pymoo
```

For other mature implementations, configure a command with an environment
variable such as:

```powershell
$env:MO_NCO_BASELINE_PYMOO_NSGA2="C:\path\to\runner.exe"
$env:MO_NCO_BASELINE_PAQUETE="C:\path\to\paquete_adapter.exe"
$env:MO_NCO_BASELINE_TPLS="C:\path\to\tpls_adapter.exe"
$env:MO_NCO_BASELINE_MOGLS="C:\path\to\mogls_adapter.exe"
$env:MO_NCO_BASELINE_LKH_MOTSP="C:\path\to\lkh_motsp_adapter.exe"
```

The command receives `input.json output.csv`. This keeps mature external
implementations on the same instance, seed, population, and evaluation-budget
protocol as the local algorithms. For fair anytime AUC, the command should also
write `output.diagnostics.csv` with rows keyed by `evaluations`, `tour`, and
`objective_0`, `objective_1`, ...; otherwise the adapter falls back to a final
archive snapshot only.

## Run tests

```powershell
C:\miniconda3\python.exe -m unittest discover -s tests -v
```

## Theory-to-code mapping

- `mo_nco.instance.MultiObjectiveTSPInstance`: finite feasible CO space and
  objective evaluation.
- `mo_nco.evaluation.CountingTSPInstance`: explicit objective-evaluation
  counter and budget guard used by the benchmark runner.
- `mo_nco.instance.MultiObjectiveTSPInstance.evaluate_two_opt`: exact
  counted 2-opt delta evaluation for symmetric objectives, with full-evaluation
  fallback for asymmetric matrices.
- `mo_nco.tsplib`: TSPLIB and simple bi-objective TSP file loaders.
- `mo_nco.moves.two_opt`: symmetric connected feasible-move proposal.
- `mo_nco.archive.ParetoArchive`: stopping-time archive state.
- `mo_nco.potential.ScalarArchivePotential`: scalar flat-potential oracle.
- `mo_nco.ips_efficient.TheoryAlignedIPSOptimizer`: fast implementation of the
  frozen scalar-potential IPS with explicit temperature schedule, Boltzmann
  replacement, archive epochs, scalar-greedy initialization, learned candidate
  proposal scoring, and a zero-temperature limit.
- `mo_nco.neural_potential.NeuralScalarPotential`: trainable scalar neural
  potential fitted at archive stopping times, then frozen during the next
  Metropolis epoch.
- `mo_nco.sampler.IPSMetropolisOptimizer`: particle IPS-Metropolis dynamics.
- `mo_nco.pareto_smc.AnnealedParetoSMCOptimizer`: typed annealed
  Feynman--Kac particle system with archive-independent targets and exact
  within-stage proposal accounting.  Its binary64 MH decision is not claimed
  to be machine-exact.
- `mo_nco.pareto_bounds`: conditional finite-particle, epsilon-cell,
  standard reference-averaged ordinary-IGD/IGD+, and fixed-reference HV
  reduction certificates.
- `mo_nco.pareto_independent_replica_runner`: separately named, noninteracting
  typed annealed-MH replica runner for direct endpoint certificates.
- `mo_nco.pareto_independent_replica_certificate`: exact-rational
  Clopper--Pearson, power, replica-count, false-PASS, and mutually exclusive
  cell-occupancy calculations.
- `mo_nco.pareto_frozen_cells`: hash-frozen rational Cartesian cell contract
  with exact endpoint classification and no clipping.
- `mo_nco.pareto_dyadic_objective`: exact dyadic tour-edge sums and cached
  symmetric/asymmetric 2-opt accounting, without an exact-MH overclaim.
- `mo_nco.pareto_archive_cap_certificate`: deterministic Gonzalez cap and
  ordinary-IGD/IGD+/shifted-HV error propagation.
- `mo_nco.pareto_reference_fidelity`: supplied-front-to-reference-to-output
  conditional composition; it is not a true-front certificate until a
  completeness artifact and its provenance are independently verified.
- `mo_nco.pareto_kernel_perturbation`: conditional rational perturbation bounds
  and a standalone fail-closed interval-decision helper for binary64-vs-ideal
  kernels; neither is currently wired into the executor or publication gate.
- `mo_nco.pareto_v15_context`: canonical case/instance/configuration/cell/
  reference/type-cell/pilot/confirm hash binding.  The replica runner verifies
  its own subset; remaining child-certificate and raw-artifact binding is an
  explicit open P0.
- `mo_nco.pareto_v15_publication_gate`: component-contract checker with the
  composed P0/P1/operational/formal/competitive gates held closed; engineering
  tests and caller-supplied booleans cannot open the publication gate.
- `mo_nco.pareto_smc_spec`: fail-closed loader for the predeclared run
  manifest.
- `mo_nco.baselines`: NSGA-II, MOEA/D, and random 2-opt baselines.
- `mo_nco.mature_baselines`: protocol adapter for mature external baselines.
- `mo_nco.benchmark_suite`: multi-instance benchmark suite runner.
- `mo_nco.budget_sweep`: multi-budget suite runner for anytime/efficiency tables.
- `mo_nco.ablation`: theory-module ablation runner.
- `mo_nco.ablation_suite`: cross-instance theory-module ablation runner.
- `mo_nco.benchmark`: multi-seed runner, summary statistics, and SVG plots.

The current code separates two claims. Fast `ips-heuristic-adaptive` /
`ips-efficient` family
members are theory-aware zero/low-temperature heuristics with changing context
and possible batched replacement.  Only `ips-theory-certified` implements the
mechanically auditable frozen-context, single-coordinate, positive-temperature
MH control.  A learned scalar prior is endpoint-only; source/action features are
reserved for the separate proposal policy.

## Algorithms

- `annealed-pareto-smc` / `pareto-smc-feynman-kac`: manifest-bound typed
  Pareto-SMC implementation. Its strongest built-in claim level is
  `pareto_smc_mechanical`; a scientific coverage claim additionally requires a
  source-bound finite-particle error constant and positive Pareto-cell mass
  certificate.
- `ips-heuristic-adaptive`: explicit name for the historical low-temperature
  heuristic with reference-direction
  particles, frozen scalar potential, Boltzmann replacement, archive stopping
  epochs, exact counted 2-opt delta evaluation, archive-HV scalar shaping,
  neural scalar distillation, neural-guided candidate proposal selection,
  scalar-greedy initialization, finite-depth scalar 2-opt descent, and
  archive-parent proposals.
- `ips-theory` / `theory-ips`: deprecated compatibility aliases for
  `ips-heuristic-adaptive`; runtime metadata marks them as heuristic descent,
  never as a certified theory implementation.
- `ips-theory-certified`: strict control using a frozen typed augmented-
  Tchebycheff Hamiltonian, uniform coordinate selection, uniform symmetric
  2-opt, constant positive temperature, and exact single-site Metropolis
  acceptance.  Its archive is reporting-only and does not feed back into the
  kernel.  Use `mo_nco.run_kernel_certificates` to audit the runtime contract.
- `ips-neural-mv-jitgreedy-targetflow-theory-optimized`: conservative
  high-performance candidate that requires an `endpoint_state_v1` scalar prior
  and a target-only move prior.  It remains a batched zero-temperature heuristic
  and must not be reported as the certified kernel.
- `ips-theory-heavy-no-prior`, `ips-theory-endpoint-only`, and
  `ips-theory-move-only`: matched-search controls for attributing gains to the
  endpoint scalar and target-only move priors independently. Run them with the
  pipeline `-Ablation` switch; the screen then forcibly uses 512 evaluations
  even when suite cases store a different default.
- `ips-theory-legacy`: previous theory-aligned configuration without
  scalar-greedy initialization or learned candidate proposal selection. This is
  retained for method-ablation and regression checks.
- `ips-offline-neural`: uses a frozen cross-instance neural prior from
  `MO_NCO_NEURAL_PRIOR_PATH`; test instances do not update the network.
- `ips-scalar-greedy`: keeps scalar-greedy initialization and archive shaping
  but disables neural proposal scoring, isolating the neural contribution.
- `ips-quality` / `ips-descent-quality`: quality-prioritized theory-aligned
  configuration using all reference-direction scalar-greedy initialization and
  deeper finite-depth scalar 2-opt descent. It is intended for LKH-derived
  quality pressure tests, not as the fastest configuration.
- `ips-neural-quality`: same quality-prioritized configuration with neural
  candidate proposal scoring enabled. Current evidence shows a very small and
  not yet decisive final-HV gain over `ips-quality`; treat it as an ablation
  claim until it is stable on the full 35+ case, 10-seed suite.
- `ips-quality-relocate` / `ips-neural-quality-relocate`: quality-pressure
  variants that add finite-depth scalar relocate/insert descent to the same
  reference-direction scalar potential. These are the current strongest local
  LKH-pressure configurations, but they are still not a settled SOTA claim.
- `ips-theory-core`: fast low-temperature core without nonzero archive/neural
  shaping, useful for speed-focused ablations.
- `ips`: theory-guided IPS-Metropolis with a frozen archive-conditioned
  hypervolume scalar potential, mixed 2-opt/swap proposals, crossover proposals,
  archive resampling, and short scalarized archive intensification at stopping
  times.
- `ips-efficient`: fast archive-conditioned zero-temperature IPS variant with
  reference-direction particles, scalar-potential neighbor replacement, delayed
  archive flushing, and a shared fast 2D Pareto archive update.

The complete endpoint-only training, target-only proposal, screen, confirmation,
and certificate commands are in `THEORY_OPTIMIZED_EXPERIMENT_COMMANDS.md`; the
one-command PowerShell entrypoint is `RUN_THEORY_OPTIMIZED_PIPELINE.ps1`.
- `ips-neural`: same IPS sampler, but the scalar single-energy component is a
  tiny trainable MLP implemented in pure Python. It is retrained at archive
  update stopping times and kept frozen between updates.
- `nsga2`: compact permutation-coded NSGA-II using order crossover and 2-opt
  mutation.
- `moead`: compact MOEA/D-style decomposition baseline with 2-opt variation.
- `motsp-pls` / `tpls` / `mogls`: MOTSP-specialized Pareto local search
  baseline with exact counted 2-opt expansion and scalar-guided archive parent
  selection.
- `lkh-scalar` / `elkai-lkh` / `lkh-derived`: optional LKH/Lin-Kernighan-style
  scalarization baseline backed by `elkai`. This is a strong external-oracle
  pressure test; its hidden scalar TSP search effort is not the same unit as
  multi-objective objective evaluations, so report wall-clock and quality
  separately.
- `random2opt`: random-walk 2-opt baseline with a Pareto archive.

## Current Local Result

The strictest local pressure test currently available is a 3-case held-out
MOTSP subset with 3 seeds, budget 512, population 32, and an `elkai`/LKH-derived
weighted-scalar TSP baseline:

```powershell
$env:MO_NCO_LKH_RUNS="1"
C:\miniconda3\python.exe -m mo_nco.run_budget_sweep `
  --suite outputs\neural_prior_split70_seed0\splits\public_pair_motsp_suite_test_lkh3.json `
  --algorithms ips-quality-relocate,ips-neural-quality-relocate,lkh-scalar,pymoo-moead `
  --seeds 0,1,2 `
  --budgets 512 `
  --output-dir outputs\relocate_quality_vs_lkh3_3cases_3seeds_eval512 `
  --log-period 256 `
  --archive-update-period 64
```

The summary is in
`outputs/relocate_quality_vs_lkh3_3cases_3seeds_eval512/budget_summary.md`.
Case-relative means:

| algorithm | rel HV | rel eval-AUC | rel time-AUC | rel HV/sec |
|---|---:|---:|---:|---:|
| ips-neural-quality-relocate | 0.9935 | 0.9991 | 0.9824 | 0.5923 |
| ips-quality-relocate | 0.9931 | 0.9990 | 0.9758 | 0.6680 |
| lkh-scalar | 1.0000 | 0.9578 | 0.9928 | 0.6418 |
| pymoo-moead | 0.1714 | 0.1087 | 0.1127 | 0.8693 |

Strict interpretation: the relocate quality IPS is now close to the
LKH-derived baseline in final HV, significantly stronger on HV-vs-evaluation
AUC, and no longer significantly worse on HV-vs-time AUC in this small pressure
test. It still loses final HV against LKH-derived scalarization on 9/9 matched
pairs, and the run covers only 3 cases and 3 seeds. The strict gate report in
`outputs/relocate_quality_vs_lkh3_3cases_3seeds_eval512/strict_sota_audit.md`
therefore still fails the SOTA claim.

## IJOC-oriented computational mainline (v20)

The manuscript-facing computational algorithm is separated from the historical
flow/drifting theory.  The formal alias is:

```text
ijoc-pareto-smc
```

It composes two phases under one exact objective-evaluation budget:

1. a frozen, typed annealed-SMC core with within-type ESS resampling and
   symmetric Metropolis mutation; and
2. a post-core search tail using either deterministic balanced allocation or a
   **stratified EXP3** policy.  The stratified policy first gives every
   reference type a frozen minimum quota, then applies EXP3 only to the
   remaining suffix.

The tail reward is a predeclared convex combination of fixed-reference
hypervolume gain, first nondominated occupancy of a frozen objective cell, and
positive typed scalar-energy improvement.  Selection draws and per-type
counterfactual proposal tapes use separately derived mutable RNG states.  The
EXP3 claim is only expected external regret for the realized-state one-step
counterfactual reward sequence.  For a quota of `q` pulls per type and suffix
length `T'`, the authorized bound is the trivial `R*q` prefix regret plus the
classical EXP3 suffix bound.  It is not policy regret and is not a final-HV or
IGD regret theorem.

Three output objects are kept distinct:

- the **competitive search archive**, an exact-tolerance, unbounded
  nondominated archive of every evaluated candidate;
- the **certificate snapshot/cell ledger**, frozen before the adaptive tail;
- the optional **deployment archive**, which may be crowding-capped and is not
  used for competitive HV or evaluation-AUC.

A frozen IJOC v2 algorithm specification is loaded from
`MO_NCO_IJOC_PARETO_SMC_SPEC`.  It SHA-binds the base SMC specification, reward
weights, tail length, allocation policy, minimum per-type tail quota, and
deployment cap.  Formal biobjective runs reject nondivisible anytime grids,
insufficient core budgets, non-symmetric proposal transitions, reused optimizer
objects, or any archive tolerance other than exact zero on the manuscript alias.

`mo_nco.pareto_ijoc_preflight` validates the v3 study packet before launch.  A
formal packet requires at least two problem families, at least ten cases per
family and thirty cases in total, at least ten seeds, at least three budgets,
three frozen strong baselines per family, an exact Cartesian configuration
matrix, case-indexed instance artifacts, actual calibration-source artifacts
for every metric reference, bound baseline wrappers/executables, a dependency
lock, and a source archive.  All bound paths must remain inside the artifact
directory.  A preflight PASS still reports `evidence_status = NOT_RUN`; it is
only a launch-readiness result.

The shared algorithmic skeleton is implemented against
`MultiObjectiveCombinatorialProblem` and exercised on fixed-origin MOTSP and
multiobjective 0--1 knapsack.  This is a software-generality bridge only until
both families have frozen matched experiments with strong problem-specific
baselines.  The single-instance adaptive-tail smoke is a development check,
not competitive evidence; both unstratified and stratified EXP3 remain
predeclared ablations until the formal matrix is complete.
