# V21E3R1 V9R2R1 完整运行手册

本手册使用 PowerShell。V9R2R1 是工程维护身份，不授权 full development
matrix。所有输出使用新目录；不得覆盖 V8、V9R1 或 V9R2 历史制品。

## 1. 初始化与身份检查

```powershell
$ErrorActionPreference = 'Stop'
Set-Location 'D:\MO_NCO'

$Py311 = 'C:\miniconda3\envs\ssm_env\python.exe'
$Py313 = 'C:\miniconda3\python.exe'
$RunRoot = 'D:\MO_NCO\v9r2r1_validation_20260825_001'

if (Test-Path -LiteralPath $RunRoot) {
    throw "output already exists: $RunRoot"
}
New-Item -ItemType Directory -Path $RunRoot | Out-Null

& $Py311 -c "import mo_nco; assert mo_nco.__version__ == '0.21.3.14'"
if ($LASTEXITCODE -ne 0) { throw 'version check failed' }
```

## 2. runpy 警告与兼容 API

```powershell
& $Py311 -W error::RuntimeWarning `
  -m mo_nco.pareto_v21e3r1_v9_gate --help
if ($LASTEXITCODE -ne 0) { throw 'gate runpy check failed' }

& $Py311 -W error::RuntimeWarning `
  -m mo_nco.pareto_v21e3r1_v9_diagnostics --help
if ($LASTEXITCODE -ne 0) { throw 'diagnostic runpy check failed' }

& $Py311 -c @'
import sys
import mo_nco
assert "mo_nco.pareto_v21e3r1_v9_gate" not in sys.modules
assert "mo_nco.pareto_v21e3r1_v9_diagnostics" not in sys.modules
from mo_nco import (
    analyze_v9_trace_database,
    evaluate_v9_predevelopment_readiness,
    write_v9_predevelopment_readiness_receipt,
)
assert callable(analyze_v9_trace_database)
assert callable(evaluate_v9_predevelopment_readiness)
assert callable(write_v9_predevelopment_readiness_receipt)
'@
if ($LASTEXITCODE -ne 0) { throw 'lazy public API check failed' }
```

## 3. 定向回归与编译

```powershell
$TargetedJUnit = Join-Path $RunRoot 'targeted.junit.xml'

& $Py311 -m pytest -q -p no:cacheprovider --tb=short `
  --junitxml=$TargetedJUnit `
  'tests\test_pareto_v21e3r1_v9_information_search.py' `
  'tests\test_pareto_v21e3r1_v9_strict_regressions.py' `
  'tests\test_pareto_v21e3r1_v9r1_theory_strict.py' `
  'tests\test_pareto_v21e3r1_v9_diagnostics_strict.py' `
  'tests\test_pareto_v21e3r1_branch_replay.py' `
  'tests\test_pareto_v21e3r1_v9_runner.py' `
  'tests\test_pareto_v21e3_trace.py' `
  'tests\test_pareto_v21e3_trace_verify.py' `
  'tests\test_pareto_v21e3_trace_chunks.py' `
  'tests\test_pareto_v21e3r1_v9_protocol.py' `
  'tests\test_pareto_v21e3r1_v9_gate.py' `
  'tests\test_pareto_v21e3r1_v9_packaging.py' `
  'tests\test_build_v9r2r1_engineering_source_bundle.py' `
  'tests\test_check_v9r2r1_full_suite_environment.py' `
  'tests\test_pareto_v21e3r1_development_diagnostic_runner.py' `
  'tests\test_v21e3r1_same_implementation_branch_replay_coverage.py'
if ($LASTEXITCODE -ne 0) { throw 'targeted regression failed' }

& $Py311 -m compileall -q `
  'mo_nco' `
  'scripts\build_v9r2r1_engineering_source_bundle.py' `
  'scripts\check_v9r2r1_full_suite_environment.py' `
  'ijoc_submission_v21e3r1\scripts\run_v21e3r1_development_diagnostics.py' `
  'ijoc_submission_v21e3r1\scripts\run_v21e3r1_same_implementation_branch_replay_coverage.py'
if ($LASTEXITCODE -ne 0) { throw 'compileall failed' }
```

## 4. 双 source bundle 与独立 verifier

source builder 要求输出目录已经存在且为空。

```powershell
$SourceA = Join-Path $RunRoot 'source_A'
$SourceB = Join-Path $RunRoot 'source_B'
New-Item -ItemType Directory -Path $SourceA, $SourceB | Out-Null

& $Py313 'scripts\build_v9r2r1_engineering_source_bundle.py' build `
  --root 'D:\MO_NCO' `
  --output-directory $SourceA
if ($LASTEXITCODE -ne 0) { throw 'source build A failed' }

& $Py313 'scripts\build_v9r2r1_engineering_source_bundle.py' build `
  --root 'D:\MO_NCO' `
  --output-directory $SourceB
if ($LASTEXITCODE -ne 0) { throw 'source build B failed' }

$SourceZipA = Join-Path $SourceA 'mo_nco-0.21.3.14-v9r2r1-source.zip'
$SourceZipB = Join-Path $SourceB 'mo_nco-0.21.3.14-v9r2r1-source.zip'
$SourceManifestA = Join-Path $SourceA 'V21E3R1_V9R2R1_SOURCE_MANIFEST.json'
$SourceManifestB = Join-Path $SourceB 'V21E3R1_V9R2R1_SOURCE_MANIFEST.json'
$SourceHashA = (Get-FileHash -Algorithm SHA256 -LiteralPath $SourceZipA).Hash.ToLowerInvariant()
$SourceHashB = (Get-FileHash -Algorithm SHA256 -LiteralPath $SourceZipB).Hash.ToLowerInvariant()
if ($SourceHashA -ne $SourceHashB) { throw "source ZIP mismatch: $SourceHashA != $SourceHashB" }
if (-not ((Get-FileHash -Algorithm SHA256 -LiteralPath $SourceManifestA).Hash -eq
          (Get-FileHash -Algorithm SHA256 -LiteralPath $SourceManifestB).Hash)) {
    throw 'source manifests differ'
}

& $Py313 'scripts\build_v9r2r1_engineering_source_bundle.py' verify `
  --manifest $SourceManifestA `
  --archive $SourceZipA `
  --root 'D:\MO_NCO'
if ($LASTEXITCODE -ne 0) { throw 'source verification failed' }
```

该 PASS 仍是 source-freeze candidate；manifest 必须保持
`full_source_freeze_requirement_satisfied=false`。

## 5. 双 wheel 精确复建

```powershell
$WheelA = Join-Path $RunRoot 'wheel_A'
$WheelB = Join-Path $RunRoot 'wheel_B'
New-Item -ItemType Directory -Path $WheelA, $WheelB | Out-Null

$env:SOURCE_DATE_EPOCH = '1700000000'
& $Py313 -m pip wheel . --no-deps --no-build-isolation --wheel-dir $WheelA
if ($LASTEXITCODE -ne 0) { throw 'wheel build A failed' }
& $Py313 -m pip wheel . --no-deps --no-build-isolation --wheel-dir $WheelB
if ($LASTEXITCODE -ne 0) { throw 'wheel build B failed' }

$WheelPathA = Join-Path $WheelA 'mo_nco-0.21.3.14-py3-none-any.whl'
$WheelPathB = Join-Path $WheelB 'mo_nco-0.21.3.14-py3-none-any.whl'
$WheelHashA = (Get-FileHash -Algorithm SHA256 -LiteralPath $WheelPathA).Hash.ToLowerInvariant()
$WheelHashB = (Get-FileHash -Algorithm SHA256 -LiteralPath $WheelPathB).Hash.ToLowerInvariant()
if ($WheelHashA -ne $WheelHashB) { throw "wheel mismatch: $WheelHashA != $WheelHashB" }
```

## 6. 仓库外 clean install 与 canonical HOLD gate

```powershell
$CleanVenv = Join-Path $RunRoot 'clean_venv'
& $Py313 -m venv $CleanVenv
if ($LASTEXITCODE -ne 0) { throw 'clean venv creation failed' }
$CleanPy = Join-Path $CleanVenv 'Scripts\python.exe'

& $CleanPy -m pip install --no-deps $WheelPathA
if ($LASTEXITCODE -ne 0) { throw 'clean install failed' }
& $CleanPy -m pip check
if ($LASTEXITCODE -ne 0) { throw 'pip check failed' }
& $CleanPy -m compileall -q (Join-Path $CleanVenv 'Lib\site-packages\mo_nco')
if ($LASTEXITCODE -ne 0) { throw 'installed compileall failed' }

$Outside = Join-Path $RunRoot 'outside_repo'
New-Item -ItemType Directory -Path $Outside | Out-Null
Push-Location $Outside
try {
    & $CleanPy -W error::RuntimeWarning -c @'
import pathlib
import mo_nco
assert mo_nco.__version__ == "0.21.3.14"
assert "site-packages" in pathlib.Path(mo_nco.__file__).as_posix()
'@
    if ($LASTEXITCODE -ne 0) { throw 'outside-repository import failed' }

    $GateOut = Join-Path $RunRoot 'installed_gate_receipt.json'
    & $CleanPy -W error::RuntimeWarning `
      -m mo_nco.pareto_v21e3r1_v9_gate `
      --expected-protocol-file-sha256 '112bbe405c64fbe598275f27a0ae7262a4e68e85469052e559de722066ef15ad' `
      --output $GateOut
    $GateExit = $LASTEXITCODE
    if ($GateExit -ne 2) { throw "expected gate HOLD exit 2, got $GateExit" }
    $Gate = Get-Content -Raw -LiteralPath $GateOut | ConvertFrom-Json
    if ($Gate.status -ne 'PRE_DEVELOPMENT_HOLD' -or
        $Gate.development_rows_materialized -ne 0 -or
        $Gate.gates.full_development_matrix_authorized -ne $false -or
        $Gate.selection_authorized -ne $false -or
        $Gate.confirmation_authorized -ne $false -or
        $Gate.formal_authorized -ne $false -or
        $Gate.ijoc_submission_authorized -ne $false) {
        throw 'installed gate authorization boundary drifted'
    }
} finally {
    Pop-Location
}
```

## 7. installed MOKP/MOTSP 四臂 B=8 工程 smoke

```powershell
$Directions = '[[0.25,0.75],[0.75,0.25]]'
$Cases = @(
    @{
        Path = 'D:\MO_NCO\ijoc_submission_v21e3\development_partitions_v1\instances\v21e3-mokp-development-n100-s00.json'
        Out  = (Join-Path $RunRoot 'installed_smoke_mokp')
    },
    @{
        Path = 'D:\MO_NCO\ijoc_submission_v21e3\development_partitions_v1\instances\v21e3-motsp-development-n100-s00.json'
        Out  = (Join-Path $RunRoot 'installed_smoke_motsp')
    }
)

Push-Location $Outside
try {
    foreach ($Item in $Cases) {
        if (Test-Path -LiteralPath $Item.Out) { throw "exists: $($Item.Out)" }
        & $CleanPy -m mo_nco.pareto_v21e3r1_v9_runner `
          --case $Item.Path `
          --outdir $Item.Out `
          --seed 1701 `
          --directions $Directions `
          --charged-evaluations 8 `
          --attempt-cap 64 `
          --structural-screening-cap 10000 `
          --wall-time-cap-seconds 120 `
          --candidate-screening-cap 4 `
          --archive-tradeoff-lambda 0.5 `
          --checkpoint-period 4 `
          --expected-protocol-file-sha256 '112bbe405c64fbe598275f27a0ae7262a4e68e85469052e559de722066ef15ad' `
          --acknowledge-exposed-development-only
        if ($LASTEXITCODE -ne 0) { throw "installed smoke failed: $($Item.Path)" }
    }
} finally {
    Pop-Location
}
```

这仍是相同实现、已暴露 case、单 seed、B=8 的工程 smoke；不得生成 CI95、
W/T/L 或 scientific promotion。

## 8. full-suite native environment fail-fast preflight

全仓的 pymoo tests 会实际加载 moocore native PYD。顶层 `import pymoo` 成功不
足以证明 backend 可用；必须运行以下真实 import preflight。当前这台机器已观察到
企业 Code Integrity policy 拒绝未签名 `_libmoocore.pyd`，此时预期 exit 2，必须
停止，不能把测试改成 skip。

```powershell
$EnvironmentReceipt = Join-Path $RunRoot 'full_suite_environment_preflight.json'
if (Test-Path -LiteralPath $EnvironmentReceipt) {
    throw "exists: $EnvironmentReceipt"
}

& $Py313 'scripts\check_v9r2r1_full_suite_environment.py' `
  --expected-python-executable 'C:\miniconda3\python.exe' `
  --expected-python-version-prefix '3.13.12' `
  --expected-pymoo-version '0.6.1.6' `
  --expected-moocore-version '0.3.1' `
  --output $EnvironmentReceipt
$EnvironmentExit = $LASTEXITCODE

if ($EnvironmentExit -eq 2) {
    Get-WinEvent -LogName 'Microsoft-Windows-CodeIntegrity/Operational' -MaxEvents 80 |
      Where-Object { $_.Message -match 'moocore|_libmoocore' } |
      Select-Object TimeCreated, Id, LevelDisplayName, Message |
      Format-List
    throw 'HOLD_FULL_SUITE_ENVIRONMENT: obtain an administrator-approved signed/frozen pymoo runtime; do not run the full suite'
}
if ($EnvironmentExit -ne 0) {
    throw "environment preflight integrity error: exit $EnvironmentExit"
}
```

不得使用 `Unblock-File`、关闭 WDAC、复制 DLL 到受信路径、旧的
`consumer_use_authorized=false` durable runtime 或源内替代实现绕过该门禁。应由
有权管理员提供符合 policy 的 native artifacts，并形成新的 environment lock。

管理员修复后，先只跑原先受影响的 pymoo 回归：

```powershell
$PymooJUnit = Join-Path $RunRoot 'pymoo_environment_recovery.junit.xml'
& $Py313 -m pytest -q -p no:cacheprovider --tb=short `
  --junitxml=$PymooJUnit `
  'tests\test_external_pymoo_baseline.py' `
  'tests\test_ijoc_algorithm_adapters.py' `
  -k 'pymoo'
if ($LASTEXITCODE -ne 0) { throw 'pymoo environment recovery regression failed' }
```

## 9. 最终全仓回归与冻结 V8 失败集合核验

最终全仓必须使用冻结历史所绑定的 Python 3.13 解释器。pytest 的 exit 1 是
当前预期，但只能在第 8 节 preflight 和 pymoo recovery 均 PASS 后运行，并在
失败数和消息族精确核验后接受。

```powershell
$FullJUnit = Join-Path $RunRoot 'full_repository.junit.xml'
& $Py313 -m pytest -q -p no:cacheprovider --tb=short --junitxml=$FullJUnit
$FullExit = $LASTEXITCODE
if ($FullExit -ne 1) { throw "expected full-suite exit 1, got $FullExit" }

[xml]$FullXml = Get-Content -Raw -LiteralPath $FullJUnit
$Failures = @($FullXml.SelectNodes('//testcase/failure'))
$Errors = @($FullXml.SelectNodes('//testcase/error'))
$Skipped = @($FullXml.SelectNodes('//testcase/skipped'))
$CasesSeen = @($FullXml.SelectNodes('//testcase'))
$WrongFailures = @($Failures | Where-Object {
    $_.InnerText -notmatch 'Frozen diagnostic source manifest drifted'
})

if ($Failures.Count -ne 8) { throw "expected 8 frozen failures, got $($Failures.Count)" }
if ($Errors.Count -ne 0) { throw "expected 0 errors, got $($Errors.Count)" }
if ($Skipped.Count -ne 4) { throw "expected 4 skipped, got $($Skipped.Count)" }
if ($WrongFailures.Count -ne 0) { throw 'unexpected failure family detected' }

[pscustomobject]@{
    JUnitCases = $CasesSeen.Count
    Failures = $Failures.Count
    Errors = $Errors.Count
    Skipped = $Skipped.Count
    JUnitSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $FullJUnit).Hash.ToLowerInvariant()
    WheelSha256 = $WheelHashA
    SourceZipSha256 = $SourceHashA
}
```

## 10. 停止条件

上述全部通过后只允许裁决 `V9R2R1 scoped engineering maintenance=PASS`。
canonical gate 仍须返回 exit 2；不允许运行 full matrix、selection、confirmation、
formal study 或 IJOC submission。下一工作流应先创建新的 external evidence
envelope，重新封口 source/test/environment 候选，再由有权主体冻结算法、metric、
rights、strong baselines、target-scale capacity 和 independent replay。
