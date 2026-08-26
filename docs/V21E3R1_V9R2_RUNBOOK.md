# V21E3R1 V9R2 工程验证运行手册

以下命令使用 PowerShell。输出目录必须尚不存在；不要覆盖历史 V8/V9R1 artifact。

## 1. 运行 230 项定向回归

```powershell
Set-Location 'D:\MO_NCO'
$Py = 'C:\miniconda3\envs\ssm_env\python.exe'

& $Py -m pytest -q -p no:cacheprovider --tb=short `
  --junitxml='D:\MO_NCO\v9r2_targeted.junit.xml' `
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
  'tests\test_pareto_v21e3r1_v9_gate.py'
```

预期：`230 passed`。

## 2. 构建并比较两个 wheel

```powershell
Set-Location 'D:\MO_NCO'
$BuildPy = 'C:\miniconda3\python.exe'
$BuildA = 'D:\MO_NCO\v9r2_build_A'
$BuildB = 'D:\MO_NCO\v9r2_build_B'
if (Test-Path -LiteralPath $BuildA) { throw "exists: $BuildA" }
if (Test-Path -LiteralPath $BuildB) { throw "exists: $BuildB" }
New-Item -ItemType Directory -Path $BuildA, $BuildB | Out-Null

$env:SOURCE_DATE_EPOCH = '1700000000'
& $BuildPy -m pip wheel . --no-deps --no-build-isolation --wheel-dir $BuildA
& $BuildPy -m pip wheel . --no-deps --no-build-isolation --wheel-dir $BuildB

$WheelA = Join-Path $BuildA 'mo_nco-0.21.3.13-py3-none-any.whl'
$WheelB = Join-Path $BuildB 'mo_nco-0.21.3.13-py3-none-any.whl'
$HashA = (Get-FileHash -Algorithm SHA256 -LiteralPath $WheelA).Hash.ToLowerInvariant()
$HashB = (Get-FileHash -Algorithm SHA256 -LiteralPath $WheelB).Hash.ToLowerInvariant()
if ($HashA -ne $HashB) { throw "wheel mismatch: $HashA != $HashB" }
Write-Output $HashA
```

本交付件预期 SHA-256：`589dc10657fd14c65008da6da8bc1111d24fe05e866d7fba8200ff031f50df6e`。

## 3. 执行 pre-development readiness gate

```powershell
$GateOut = 'D:\MO_NCO\v9r2_predevelopment_gate.json'
if (Test-Path -LiteralPath $GateOut) { throw "exists: $GateOut" }

& 'C:\miniconda3\envs\ssm_env\python.exe' `
  -m mo_nco.pareto_v21e3r1_v9_gate `
  --expected-protocol-file-sha256 112bbe405c64fbe598275f27a0ae7262a4e68e85469052e559de722066ef15ad `
  --output $GateOut

if ($LASTEXITCODE -ne 2) { throw "expected HOLD exit 2, got $LASTEXITCODE" }
```

退出码 2 是当前 canonical protocol 的预期 fail-closed 结果。收据必须报告 `PRE_DEVELOPMENT_HOLD`、0 个 development rows 和所有后续阶段未授权。不要把退出码改为 0 来绕过前置项。

## 4. 运行单个已暴露 case 的四臂工程 smoke

```powershell
Set-Location 'D:\MO_NCO'
$Py = 'C:\miniconda3\envs\ssm_env\python.exe'
$Case = 'D:\MO_NCO\ijoc_submission_v21e3\development_partitions_v1\instances\v21e3-mokp-development-n100-s00.json'
$Out = 'D:\MO_NCO\v9r2_smoke_mokp'
if (Test-Path -LiteralPath $Out) { throw "exists: $Out" }

& $Py -m mo_nco.pareto_v21e3r1_v9_runner `
  --case $Case `
  --outdir $Out `
  --seed 1701 `
  --directions '[[0.25,0.75],[0.75,0.25]]' `
  --charged-evaluations 8 `
  --attempt-cap 64 `
  --structural-screening-cap 10000 `
  --wall-time-cap-seconds 120 `
  --candidate-screening-cap 4 `
  --archive-tradeoff-lambda 0.5 `
  --checkpoint-period 4 `
  --expected-protocol-file-sha256 112bbe405c64fbe598275f27a0ae7262a4e68e85469052e559de722066ef15ad `
  --acknowledge-exposed-development-only
```

MOTSP 只替换 case 和新的输出目录：

```powershell
$Case = 'D:\MO_NCO\ijoc_submission_v21e3\development_partitions_v1\instances\v21e3-motsp-development-n100-s00.json'
$Out = 'D:\MO_NCO\v9r2_smoke_motsp'
```

成功后必须有四个臂，每臂均包含 `trace.sqlite3`、`terminal.json`、`diagnostic.json` 和 `branch_replay.json`。顶层 `summary.json` 必须继续声明 `scientific_independence=false`，并禁止 selection、confirmation、formal study 和 IJOC submission。

## 5. 单独运行外部绑定的只读诊断

```powershell
$Arm = Join-Path $Out 'BOTH'
$Terminal = Join-Path $Arm 'terminal.json'
$TerminalSha = (Get-FileHash -Algorithm SHA256 -LiteralPath $Terminal).Hash.ToLowerInvariant()
$Diagnostic = Join-Path $Arm 'diagnostic_cli.json'
if (Test-Path -LiteralPath $Diagnostic) { throw "exists: $Diagnostic" }

& $Py -m mo_nco.pareto_v21e3r1_v9_diagnostics `
  --trace (Join-Path $Arm 'trace.sqlite3') `
  --terminal-receipt $Terminal `
  --expected-terminal-receipt-sha256 $TerminalSha `
  --output $Diagnostic
```

诊断中的 `full_algorithm_decision_replay=NOT_IMPLEMENTED` 是边界声明，不是 PASS。完整独立算法重放作为 full development matrix 的必需 artifact 仍然缺失。

## 6. 当前禁止的动作

当前 protocol 没有可执行的 full-matrix/promotion 通道。它机器列出 10 项当前阻塞 artifact，且明确不是 later scientific/submission 要求的穷尽清单。只有全部适用要求由有权主体提供、hash 绑定，并生成新的 canonical protocol 身份后，才可另行实现和审计下一阶段。不得修改当前 protocol 的 false 字段、阈值、case menu 或 gate 退出语义来产生授权。
