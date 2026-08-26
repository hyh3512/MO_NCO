[CmdletBinding()]
param(
    [ValidateSet('EvidenceReplay', 'ScopedLive', 'FullLive')]
    [string]$Mode = 'EvidenceReplay',
    [string]$ProjectRoot,
    [string]$Python311 = 'C:\miniconda3\envs\ssm_env\python.exe',
    [string]$Python313 = 'C:\miniconda3\python.exe',
    [string]$OutputDirectory
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
    $ProjectRoot = Split-Path -Parent $PSScriptRoot
}
$ProjectRoot = [System.IO.Path]::GetFullPath($ProjectRoot)
if (-not (Test-Path -LiteralPath (Join-Path $ProjectRoot 'pyproject.toml') -PathType Leaf)) {
    throw "not a project root: $ProjectRoot"
}

if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
    $stamp = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ')
    $OutputDirectory = Join-Path $ProjectRoot "v9r2r1_engineering_validation_$stamp"
}
$OutputDirectory = [System.IO.Path]::GetFullPath($OutputDirectory)
$rootPrefix = $ProjectRoot.TrimEnd('\') + '\'
if (-not $OutputDirectory.StartsWith($rootPrefix, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'output directory must remain inside the declared project root'
}
if (Test-Path -LiteralPath $OutputDirectory) {
    throw "refusing to overwrite existing output: $OutputDirectory"
}
New-Item -ItemType Directory -Path $OutputDirectory | Out-Null

function Invoke-PythonExpectedExit {
    param(
        [Parameter(Mandatory = $true)][string]$Executable,
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [Parameter(Mandatory = $true)][int[]]$ExpectedExitCodes
    )
    if (-not (Test-Path -LiteralPath $Executable -PathType Leaf)) {
        throw "Python executable missing: $Executable"
    }
    & $Executable @Arguments
    $exitCode = $LASTEXITCODE
    if ($ExpectedExitCodes -notcontains $exitCode) {
        throw "unexpected exit $exitCode from: $Executable $($Arguments -join ' ')"
    }
    return $exitCode
}

$EvidenceRoot = Join-Path $ProjectRoot 'evidence\v9r2r1_environment_recovery_20260825_002'
$Registry = Join-Path $ProjectRoot 'provenance\V9R2R1_EXPECTED_HISTORICAL_V8_FAILURE_SET.json'
$SourceBuilder = Join-Path $ProjectRoot 'scripts\build_v9r2r1_engineering_source_bundle.py'
$FailureVerifier = Join-Path $ProjectRoot 'scripts\verify_expected_historical_failure_set.py'
$EnvelopeBuilder = Join-Path $ProjectRoot 'scripts\build_v9r2r1_engineering_envelope.py'
$EnvelopeVerifier = Join-Path $ProjectRoot 'scripts\verify_v9r2r1_engineering_envelope.py'
$EnvironmentPreflightScript = Join-Path $ProjectRoot 'scripts\check_v9r2r1_full_suite_environment.py'

$SourceDirectory = Join-Path $OutputDirectory 'source_bundle'
New-Item -ItemType Directory -Path $SourceDirectory | Out-Null
Invoke-PythonExpectedExit -Executable $Python313 -ExpectedExitCodes @(0) -Arguments @(
    $SourceBuilder,
    'build',
    '--root', $ProjectRoot,
    '--output-directory', $SourceDirectory
) | Out-Null
$SourceManifest = Join-Path $SourceDirectory 'V21E3R1_V9R2R1_SOURCE_MANIFEST.json'

$TargetedJUnit = Join-Path $EvidenceRoot 'targeted_final.junit.xml'
$PymooJUnit = Join-Path $EvidenceRoot 'pymoo_environment_recovery.junit.xml'
$FullJUnit = Join-Path $EvidenceRoot 'full_repository.junit.xml'
$EnvironmentPreflight = Join-Path $EvidenceRoot 'full_suite_environment_preflight.json'

$TargetedTests = @(
    'tests\test_pareto_v21e3r1_v9_information_search.py',
    'tests\test_pareto_v21e3r1_v9_strict_regressions.py',
    'tests\test_pareto_v21e3r1_v9r1_theory_strict.py',
    'tests\test_pareto_v21e3r1_v9_diagnostics_strict.py',
    'tests\test_pareto_v21e3r1_branch_replay.py',
    'tests\test_pareto_v21e3r1_v9_runner.py',
    'tests\test_pareto_v21e3_trace.py',
    'tests\test_pareto_v21e3_trace_verify.py',
    'tests\test_pareto_v21e3_trace_chunks.py',
    'tests\test_pareto_v21e3r1_v9_protocol.py',
    'tests\test_pareto_v21e3r1_v9_gate.py',
    'tests\test_pareto_v21e3r1_v9_packaging.py',
    'tests\test_build_v9r2r1_engineering_source_bundle.py',
    'tests\test_check_v9r2r1_full_suite_environment.py',
    'tests\test_pareto_v21e3r1_development_diagnostic_runner.py',
    'tests\test_v21e3r1_same_implementation_branch_replay_coverage.py'
)

Push-Location $ProjectRoot
try {
    if ($Mode -in @('ScopedLive', 'FullLive')) {
        $TargetedJUnit = Join-Path $OutputDirectory 'targeted.junit.xml'
        $targetedArguments = @(
            '-m', 'pytest', '-q', '-p', 'no:cacheprovider', '--tb=short',
            "--junitxml=$TargetedJUnit"
        ) + $TargetedTests
        Invoke-PythonExpectedExit -Executable $Python311 `
            -Arguments $targetedArguments -ExpectedExitCodes @(0) | Out-Null
    }

    if ($Mode -eq 'FullLive') {
        $EnvironmentPreflight = Join-Path $OutputDirectory 'full_suite_environment_preflight.json'
        Invoke-PythonExpectedExit -Executable $Python313 -ExpectedExitCodes @(0) -Arguments @(
            $EnvironmentPreflightScript,
            '--expected-python-executable', $Python313,
            '--expected-python-version-prefix', '3.13.12',
            '--expected-pymoo-version', '0.6.1.6',
            '--expected-moocore-version', '0.3.1',
            '--output', $EnvironmentPreflight
        ) | Out-Null

        $PymooJUnit = Join-Path $OutputDirectory 'pymoo_environment_recovery.junit.xml'
        $requiredPymooTests = @(
            'tests\test_external_pymoo_baseline.py',
            'tests\test_ijoc_algorithm_adapters.py'
        )
        foreach ($testPath in $requiredPymooTests) {
            if (-not (Test-Path -LiteralPath (Join-Path $ProjectRoot $testPath) -PathType Leaf)) {
                throw "full-live source closure missing required test: $testPath"
            }
        }
        $pymooArguments = @(
            '-m', 'pytest', '-q', '-p', 'no:cacheprovider', '--tb=short',
            "--junitxml=$PymooJUnit"
        ) + $requiredPymooTests + @('-k', 'pymoo')
        Invoke-PythonExpectedExit -Executable $Python313 `
            -ExpectedExitCodes @(0) -Arguments $pymooArguments | Out-Null

        $FullJUnit = Join-Path $OutputDirectory 'full_repository.junit.xml'
        Invoke-PythonExpectedExit -Executable $Python313 -ExpectedExitCodes @(1) -Arguments @(
            '-m', 'pytest', '-q', '-p', 'no:cacheprovider', '--tb=short',
            "--junitxml=$FullJUnit"
        ) | Out-Null
    }
} finally {
    Pop-Location
}

$FailureReceipt = Join-Path $OutputDirectory 'expected_historical_v8_failure_set.receipt.json'
$failureArguments = @(
    $FailureVerifier,
    '--registry', $Registry,
    '--junit', $FullJUnit,
    '--output', $FailureReceipt
)
if ($Mode -ne 'FullLive') {
    $failureArguments += '--require-reference-sha256'
}
Invoke-PythonExpectedExit -Executable $Python313 `
    -Arguments $failureArguments -ExpectedExitCodes @(0) | Out-Null

$Envelope = Join-Path $OutputDirectory 'V9R2R1_ENGINEERING_RECOVERY_ENVELOPE.json'
Invoke-PythonExpectedExit -Executable $Python313 -ExpectedExitCodes @(0) -Arguments @(
    $EnvelopeBuilder,
    '--root', $ProjectRoot,
    '--source-manifest', $SourceManifest,
    '--environment-preflight', $EnvironmentPreflight,
    '--pymoo-junit', $PymooJUnit,
    '--targeted-junit', $TargetedJUnit,
    '--full-repository-junit', $FullJUnit,
    '--expected-failure-registry', $Registry,
    '--expected-failure-receipt', $FailureReceipt,
    '--output', $Envelope
) | Out-Null

$EnvelopeVerification = Join-Path $OutputDirectory 'V9R2R1_ENGINEERING_RECOVERY_ENVELOPE.verify.json'
Invoke-PythonExpectedExit -Executable $Python313 -ExpectedExitCodes @(0) -Arguments @(
    $EnvelopeVerifier,
    '--root', $ProjectRoot,
    '--envelope', $Envelope,
    '--output', $EnvelopeVerification
) | Out-Null

[pscustomobject]@{
    status = 'PASS_VERIFIED_SCOPED_ENGINEERING_RECOVERY_ENVELOPE_ONLY'
    mode = $Mode
    output_directory = $OutputDirectory
    repository_wide_green = $false
    environment_lock_satisfied = $false
    scientific_stage_authorized = $false
    full_development_matrix_authorized = $false
    selection_authorized = $false
    confirmation_authorized = $false
    formal_authorized = $false
    ijoc_submission_authorized = $false
} | ConvertTo-Json -Compress
