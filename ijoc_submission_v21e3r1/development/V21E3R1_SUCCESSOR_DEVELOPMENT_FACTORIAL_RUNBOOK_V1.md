# V21e3r1 successor development factorial V1

Status: prospective development-only design. This workflow does not generate or
accept selection, confirmation, or formal cases and does not authorize a
scientific or IJOC claim.

The exact design is derived from the hash-pinned V7 504-row plan: six exposed
MOKP cases and six exposed MOTSP cases, seeds 31051/31057/31059, budget 2000,
and checkpoint period 200. MOKP uses the four frozen 2x2 arms
`MOKP_LEGACY`, `MOKP_ANCHOR_ONLY`, `MOKP_NOVELTY_ONLY`, and `MOKP_BOTH`
(72 rows). MOTSP uses `MOTSP_LEGACY` and `MOTSP_ANCHOR` (36 rows). The total
is exactly 108 rows.

The prospective inference contract is
`V21E3R1_SUCCESSOR_DEVELOPMENT_FACTORIAL_INFERENCE_V1.json`. It freezes the
one-sided observed-SE max-t paired-case-cluster bootstrap, alpha 0.05,
9999 replicates, seed 2026082301, and all five hypotheses and practical
thresholds before execution.

The selection/confirmation evaluator cannot be reused for this gate because it
requires a balanced C0--C3 matrix and one score metric. This development gate
has a four-arm MOKP factorial, a two-arm MOTSP contrast, and both EAUC and
cache-hit-rate effects. The dedicated evaluator uses the same bootstrap method,
but all independence and later-phase authority fields remain false.

## Commands

Run only after the V7 exact-504 receipt is sealed and after a successor-source
freeze receipt has been created for the final implementation. A plan-only pass
is available for inspection. These programs use package imports, so direct
execution of their `.py` paths is unsupported and can fail with
`ModuleNotFoundError`. Run every entry point from the repository root through
the pinned environment and Python's `-m` module mode.

First verify both module entry points:

```powershell
Set-Location -LiteralPath 'D:\MO_NCO'
$V21Python = 'C:\miniconda3\envs\ssm_env\python.exe'

& $V21Python -m ijoc_submission_v21e3r1.scripts.run_v21e3r1_successor_development_factorial --help
if ($LASTEXITCODE -ne 0) { throw "Factorial runner module preflight failed: $LASTEXITCODE" }

& $V21Python -m ijoc_submission_v21e3r1.scripts.evaluate_v21e3r1_successor_development_factorial --help
if ($LASTEXITCODE -ne 0) { throw "Factorial evaluator module preflight failed: $LASTEXITCODE" }
```

Then create the inspectable plan:

```powershell
Set-Location -LiteralPath 'D:\MO_NCO'
$V21Python = 'C:\miniconda3\envs\ssm_env\python.exe'

& $V21Python -m ijoc_submission_v21e3r1.scripts.run_v21e3r1_successor_development_factorial `
  --project-root D:\MO_NCO `
  --output-directory D:\MO_NCO\outputs\v21e3r1_successor_factorial_20260823 `
  --v7-diagnostic-plan D:\MO_NCO\outputs\v21e3r1_v7_exposed_development_diagnostics_20260823\diagnostic.plan.json `
  --source-freeze-receipt D:\MO_NCO\outputs\v21e3r1_successor_source_freeze_20260823\successor-source.freeze.receipt.json `
  --plan-only
```

Start from that frozen plan with `--resume` (or omit both `--plan-only` and
`--resume` when creating and executing a new output directory in one call):

```powershell
Set-Location -LiteralPath 'D:\MO_NCO'
$V21Python = 'C:\miniconda3\envs\ssm_env\python.exe'

& $V21Python -m ijoc_submission_v21e3r1.scripts.run_v21e3r1_successor_development_factorial `
  --project-root D:\MO_NCO `
  --output-directory D:\MO_NCO\outputs\v21e3r1_successor_factorial_20260823 `
  --v7-diagnostic-plan D:\MO_NCO\outputs\v21e3r1_v7_exposed_development_diagnostics_20260823\diagnostic.plan.json `
  --source-freeze-receipt D:\MO_NCO\outputs\v21e3r1_successor_source_freeze_20260823\successor-source.freeze.receipt.json `
  --resume
```

An interrupted run uses the same second command. Existing attempt directories
are never overwritten; a retry receives the next `attempt-NNNN` directory.
Resume is fail-closed: every completed row is re-bound to its exact plan row,
attempt directory, trace, terminal receipt, independent metric receipt, source
freeze, and frozen semantic configuration before it is reused. The finalized
matrix uses the V2 plan/row/completed/aggregate/receipt contracts and revalidates
the sealed V7 exact-504 parent plus the live successor-source freeze both before
and after finalization.

After the exact-108 matrix receipt is sealed, evaluate the development
promotion gate:

```powershell
Set-Location -LiteralPath 'D:\MO_NCO'
$V21Python = 'C:\miniconda3\envs\ssm_env\python.exe'

& $V21Python -m ijoc_submission_v21e3r1.scripts.evaluate_v21e3r1_successor_development_factorial `
  --project-root D:\MO_NCO `
  --matrix-directory D:\MO_NCO\outputs\v21e3r1_successor_factorial_20260823 `
  --output D:\MO_NCO\outputs\v21e3r1_successor_factorial_20260823\development-promotion.evaluation.json
```

Exit code 0 means all five simultaneous lower bounds strictly exceed their
frozen thresholds and authorizes development promotion only. Exit code 2 is a
valid HOLD (threshold failure or zero standard error). Exit code 3 is an
integrity/execution HOLD. No exit code from these programs authorizes selection,
confirmation, formal materialization, a scientific claim, or IJOC submission.

Evaluation does not trust row-level cached metrics. For all 108 rows it freshly
verifies the trace database, reruns the stdlib independent metric recomputation
in a temporary directory, derives cache hits from the validated terminal
receipt, and then checks the frozen inference contract. Its V2 evaluation
receipt binds the study/candidate identity, successor source and configuration,
source-freeze and source-manifest receipts, exact matrix plan/receipt, frozen
inference specification, and an ordered 108-row replay witness. A statistically
valid threshold or zero-standard-error HOLD remains a canonical V2 receipt and
may be carried forward as a blocking prerequisite. An integrity failure instead
returns exit code 3 and, when the exclusive output path is safely available,
writes only the separate `evaluation_integrity_hold_v1` evidence schema; that
schema is never a promotion authorization input.

The exact V2 `promotion_scope` is
`SUCCESSOR_DEVELOPMENT_PROMOTION_ONLY_HASH_BOUND_PRODUCER_RECEIPT_NO_PROSPECTIVE_108_ROW_RECOMPUTATION_NO_SCIENTIFIC_CLAIM`.
Accordingly, the later prospective gate verifies the bound producer receipt and
its cross-stage identities; it does not independently repeat the 108 row
replays. This is not implementation-independent evidence. Selection remains
HOLD unless its separate external-baseline, independent-producer, custody, and
all other prospective prerequisites pass.

## Focused verification

```powershell
Set-Location -LiteralPath 'D:\MO_NCO'
$V21Python = 'C:\miniconda3\envs\ssm_env\python.exe'

& $V21Python -m pytest -q tests\test_v21e3r1_successor_development_factorial.py
```
