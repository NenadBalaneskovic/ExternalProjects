<#
Validate-Environment.ps1
------------------------
Validates the system environment for the ONNX Model Generator.
#>

$OutputEncoding = [Console]::OutputEncoding = [System.Text.Encoding]::UTF8

function Write-Log {
    param([string]$Message)
    $timestamp = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
    Write-Output "$timestamp  $Message"
}

function Test-Command {
    param([string]$Command)
    $result = (Get-Command $Command -ErrorAction SilentlyContinue)
    return ($result -ne $null)
}

function Test-PipPackage {
    param([string]$PackageName)
    pip show $PackageName 2>$null | Out-Null
    return ($LASTEXITCODE -eq 0)
}

Write-Log "=== Environment Validation Started ==="
$Errors = 0

# ---------------------------------------------------------------------------
# Fix PATH for Git and Podman (critical!)
# ---------------------------------------------------------------------------

$GitPath = "C:\Program Files\Git\cmd"
$PodmanPath = "C:\Program Files\RedHat\Podman"

if (Test-Path $GitPath) {
    Write-Log "Adding Git to PATH: $GitPath"
    $env:PATH = "$GitPath;$env:PATH"
}

if (Test-Path $PodmanPath) {
    Write-Log "Adding Podman to PATH: $PodmanPath"
    $env:PATH = "$PodmanPath;$env:PATH"
}

# ---------------------------------------------------------------------------
# Python
# ---------------------------------------------------------------------------
Write-Log "Checking Python availability..."
if (Test-Command "python") {
    Write-Log "Python found."

    $pyVersion = python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
    Write-Log "Python version detected: $pyVersion"

    if ([version]$pyVersion -lt [version]"3.10") {
        Write-Log "ERROR: Python version must be >= 3.10."
        $Errors++
    } else {
        Write-Log "Python version OK."
    }
} else {
    Write-Log "ERROR: Python not found on PATH."
    $Errors++
}

# ---------------------------------------------------------------------------
# pip
# ---------------------------------------------------------------------------
Write-Log "Checking pip availability..."
if (Test-Command "pip") {
    Write-Log "pip found."
} else {
    Write-Log "ERROR: pip not found."
    $Errors++
}

# ---------------------------------------------------------------------------
# Python packages
# ---------------------------------------------------------------------------
$packages = @(
    "onnx",
    "onnxruntime",
    "torch",
    "scikit-learn",
    "tensorflow",
    "mlserver",
    "tritonclient"
)

Write-Log "Checking Python packages..."

foreach ($pkg in $packages) {
    if (Test-PipPackage $pkg) {
        Write-Log "Package '$pkg' : INSTALLED"
    } else {
        Write-Log "Package '$pkg' : MISSING"
        $Errors++
    }
}

# ---------------------------------------------------------------------------
# Git
# ---------------------------------------------------------------------------
Write-Log "Checking Git..."
if (Test-Command "git") {
    Write-Log "Git found."
} else {
    Write-Log "ERROR: Git not found."
    $Errors++
}

# ---------------------------------------------------------------------------
# Podman
# ---------------------------------------------------------------------------
Write-Log "Checking Podman..."
if (Test-Command "podman") {
    Write-Log "Podman found."
} else {
    Write-Log "ERROR: Podman not found."
    $Errors++
}

# ---------------------------------------------------------------------------
# Final result
# ---------------------------------------------------------------------------
if ($Errors -eq 0) {
    Write-Log "=== Environment Validation PASSED ==="
    exit 0
} else {
    Write-Log "=== Environment Validation FAILED ==="
    Write-Log "Errors detected: $Errors"
    exit 1
}
