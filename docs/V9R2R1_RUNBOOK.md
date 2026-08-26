# V9R2R1 reproducibility entry points

The complete historical command reference remains
[`V21E3R1_V9R2R1_RUNBOOK.md`](V21E3R1_V9R2R1_RUNBOOK.md). For a fresh Git
checkout, begin with the repository-level fail-closed entry point:

Create two isolated environments and install the checked version closures
before running validation. These locks are not artifact-hashed cross-platform
locks, so this step does not satisfy the scientific environment-lock gate.

```powershell
Set-Location '<MO_NCO checkout>'
$Base311 = 'C:\miniconda3\envs\ssm_env\python.exe'
$Base313 = 'C:\miniconda3\python.exe'
$Env311 = Join-Path (Get-Location).Path '.venv-v9r2r1-py311'
$Env313 = Join-Path (Get-Location).Path '.venv-v9r2r1-py313'

& $Base311 -m venv $Env311
& $Base313 -m venv $Env313
$Py311 = Join-Path $Env311 'Scripts\python.exe'
$Py313 = Join-Path $Env313 'Scripts\python.exe'

& $Py311 -m pip install --requirement .\requirements\base.lock
& $Py313 -m pip install --requirement .\requirements\optional-pymoo.lock
& $Py311 -m pip check
& $Py313 -m pip check

& $Py311 -m pytest -q .\tests\test_v9r2r1_dependency_contract.py
& $Py313 -m pytest -q .\tests\test_v9r2r1_dependency_contract.py

& $Py313 scripts\verify_v9r2r1_current_source.py `
  --root . `
  --manifest .\V21E3R1_V9R2R1_SOURCE_MANIFEST.json `
  --require-git-tracked `
  --require-pass

powershell -NoProfile -ExecutionPolicy Bypass -File `
  .\scripts\run_v9r2r1_engineering_validation.ps1 `
  -Mode EvidenceReplay `
  -ProjectRoot (Get-Location).Path `
  -Python311 $Py311 `
  -Python313 $Py313 `
  -OutputDirectory (Join-Path (Get-Location).Path 'tmp\v9r2r1-evidence-replay')
```

The output directory must not already exist. `EvidenceReplay` verifies the
checked-in historical evidence byte-for-byte; it does not rerun tests. To rerun
the scoped 271-test engineering regression, use a different nonexistent output
directory and replace `-Mode EvidenceReplay` with `-Mode ScopedLive`.

`FullLive` also executes the complete public-checkout suite and accepts only the
eight exact versioned historical V8 node-ids. The currently published checkout
is expected to fail closed in this mode because redistribution authority is
absent for required historical fixture/evidence bytes. A nonzero exit must not
be weakened, xfailed, skipped, or reinterpreted as a scientific result.

Exit `0` means only that the selected scoped engineering contract passed. It
does not mean repository-wide green, a complete environment lock, scientific
independence, or scientific PASS.

Only the already exposed development cases may be used for the four-arm smoke.
Do not materialize selection, confirmation, or formal case bytes from this
runbook.
