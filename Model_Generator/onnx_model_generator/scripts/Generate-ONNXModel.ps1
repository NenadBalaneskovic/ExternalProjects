<#
Generate-ONNXModel.ps1
----------------------
PowerShell orchestrator for the ONNX Model Generator.

Responsibilities:
- Accept parameters from the GUI (model folder, entry point, backend, output folder)
- Call the appropriate backend converter
- Emit detailed logs
- Exit non-zero on failure
#>

param(
    [Parameter(Mandatory = $true)]
    [string]$ModelFolder,

    [Parameter(Mandatory = $true)]
    [string]$EntryPoint,

    [Parameter(Mandatory = $true)]
    [ValidateSet("torch", "sklearn", "custom", "mlserver", "triton")]
    [string]$Backend,

    [Parameter(Mandatory = $true)]
    [string]$OutputFolder
)

# Optional: keep console UTF-8, but it's not enough alone
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [Console]::OutputEncoding

function Write-Log {
    param([string]$Message)
    $timestamp = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
    Write-Output "$timestamp  $Message"
}

Write-Log "=== ONNX Model Generation Orchestrator Started ==="

Write-Log "Model folder      : $ModelFolder"
Write-Log "Entry point       : $EntryPoint"
Write-Log "Backend           : $Backend"
Write-Log "Output folder     : $OutputFolder"

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
if (-not (Test-Path -Path $ModelFolder)) {
    Write-Log "ERROR: Model folder does not exist: $ModelFolder"
    exit 1
}

$entryPath = Join-Path $ModelFolder $EntryPoint
if (-not (Test-Path -Path $entryPath)) {
    Write-Log "ERROR: Entry point file not found: $entryPath"
    exit 1
}

if (-not (Test-Path -Path $OutputFolder)) {
    Write-Log "Output folder does not exist. Creating: $OutputFolder"
    New-Item -ItemType Directory -Path $OutputFolder | Out-Null
}

# ---------------------------------------------------------------------------
# Determine converter path
# ---------------------------------------------------------------------------
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$ConvertersDir = Join-Path $ProjectRoot "gui\converters"

Write-Log "Script directory   : $ScriptDir"
Write-Log "Project root       : $ProjectRoot"
Write-Log "Converters folder  : $ConvertersDir"

if (-not (Test-Path -Path $ConvertersDir)) {
    Write-Log "ERROR: Converters folder not found: $ConvertersDir"
    exit 1
}

$ConverterModule = switch ($Backend) {
    "torch"    { "convert_torch.py" }
    "sklearn"  { "convert_sklearn.py" }
    "custom"   { "convert_custom.py" }
    "mlserver" { "convert_mlserver.py" }
    "triton"   { "convert_triton.py" }
}

if (-not $ConverterModule) {
    Write-Log "ERROR: No converter module mapped for backend: $Backend"
    exit 1
}

$ConverterPath = Join-Path $ConvertersDir $ConverterModule

if (-not (Test-Path -Path $ConverterPath)) {
    Write-Log "ERROR: Converter module not found: $ConverterPath"
    exit 1
}

Write-Log "Selected converter : $ConverterModule"
Write-Log "Converter path     : $ConverterPath"

# ---------------------------------------------------------------------------
# Escape paths for Python
# ---------------------------------------------------------------------------
$PyConvertersDir = $ConvertersDir -replace '\\', '\\\\'
$PyModelFolder   = $ModelFolder -replace '\\', '\\\\'
$PyEntryPoint    = $EntryPoint -replace '\\', '\\\\'
$PyOutputFolder  = $OutputFolder -replace '\\', '\\\\'
$PyModuleName    = [System.IO.Path]::GetFileNameWithoutExtension($ConverterModule)

# ---------------------------------------------------------------------------
# Build Python script content
# ---------------------------------------------------------------------------
$PythonScript = @"
import sys
import os

sys.path.insert(0, r"$PyConvertersDir")

from $PyModuleName import convert

def log(msg):
    print(msg, flush=True)

log(f"> Python executable: {sys.executable}")

model_folder = r"$PyModelFolder"
entry_point  = r"$PyEntryPoint"
output_folder = r"$PyOutputFolder"

log("> Python: starting backend conversion...")
onnx_path = convert(model_folder, entry_point, output_folder, log_callback=log)
log(f"> Python: conversion finished. ONNX path: {onnx_path}")
"@

# ---------------------------------------------------------------------------
# Write script to temp file
# ---------------------------------------------------------------------------
$TempPy = Join-Path $env:TEMP "onnx_convert_temp.py"
Set-Content -Path $TempPy -Value $PythonScript -Encoding UTF8

# ---------------------------------------------------------------------------
# Execute Python script
# ---------------------------------------------------------------------------
Write-Log "Invoking Python converter..."
Write-Log "Backend            : $Backend"
Write-Log "Entry point        : $EntryPoint"
Write-Log "Model folder       : $ModelFolder"
Write-Log "Output folder      : $OutputFolder"

$psi = New-Object System.Diagnostics.ProcessStartInfo

# ⭐ FORCE THE CORRECT PYTHON INTERPRETER ⭐
$psi.FileName = "C:\ProgramData\miniforge3\envs\jupyter-env\python.exe"

$psi.Arguments = "`"$TempPy`""
$psi.RedirectStandardOutput = $true
$psi.RedirectStandardError = $true
$psi.UseShellExecute = $false
$psi.CreateNoWindow = $true

# ⭐ CRUCIAL FIX: force Python to use UTF‑8 for stdout/stderr
$psi.EnvironmentVariables["PYTHONIOENCODING"] = "utf-8"

$process = New-Object System.Diagnostics.Process
$process.StartInfo = $psi
$null = $process.Start()

while (-not $process.HasExited) {
    $line = $process.StandardOutput.ReadLine()
    if ($line -ne $null) {
        Write-Output $line
    }
}

while (-not $process.StandardOutput.EndOfStream) {
    $line = $process.StandardOutput.ReadLine()
    if ($line -ne $null) {
        Write-Output $line
    }
}

$stderr = $process.StandardError.ReadToEnd()
if ($stderr -and $stderr.Trim().Length -gt 0) {
    Write-Log "Python STDERR:"
    Write-Output $stderr
}

$exitCode = $process.ExitCode

if ($exitCode -ne 0) {
    Write-Log "ERROR: Python converter exited with code $exitCode"
    Write-Log "=== ONNX Model Generation FAILED ==="
    exit $exitCode
}

Write-Log "=== ONNX Model Generation COMPLETED SUCCESSFULLY ==="
exit 0
