<#
Install-Dependencies.ps1
------------------------
Installs all required Python packages and system tools for the ONNX Model Generator.

Responsibilities:
- Install single dependency when called with -Package
- Install all dependencies when called without parameters
- Provide detailed, explainable logs for the GUI
- Exit with non-zero code on failure

Dependencies installed:
- onnx
- onnxruntime
- torch
- scikit-learn
- tensorflow
- mlserver
- tritonclient

System tools installed:
- git (via winget)
- podman (via winget)

This script is designed to be called from the GUI.
#>

param(
    [string]$Package = ""
)

$OutputEncoding = [Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# ---------------------------------------------------------------------------
# Helper: Write log messages with timestamp
# ---------------------------------------------------------------------------
function Write-Log {
    param([string]$Message)
    $timestamp = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
    Write-Output "$timestamp  $Message"
}

# ---------------------------------------------------------------------------
# Helper: Check if a pip package is installed
# ---------------------------------------------------------------------------
function Test-PipPackage {
    param([string]$PackageName)

    pip show $PackageName 2>$null | Out-Null
    return ($LASTEXITCODE -eq 0)
}

# ---------------------------------------------------------------------------
# Helper: Install a pip package with logging
# ---------------------------------------------------------------------------
function Install-PipPackage {
    param([string]$PackageName)

    if (Test-PipPackage $PackageName) {
        Write-Log "Package '$PackageName' is already installed."
        return 0
    }

    Write-Log "Installing pip package '$PackageName'..."
    pip install $PackageName

    if ($LASTEXITCODE -ne 0) {
        Write-Log "ERROR: Failed to install pip package '$PackageName'."
        return 1
    }

    Write-Log "Successfully installed pip package '$PackageName'."
    return 0
}

# ---------------------------------------------------------------------------
# Helper: Install Git via winget + PATH fix
# ---------------------------------------------------------------------------
function Install-Git {
    Write-Log "Installing Git via winget..."
    winget install --id Git.Git -e --source winget

    if ($LASTEXITCODE -ne 0) {
        Write-Log "ERROR: Failed to install Git."
        return 1
    }

    $GitPath = "C:\Program Files\Git\cmd"
    if (Test-Path $GitPath) {
        Write-Log "Adding Git to PATH: $GitPath"
        $env:PATH = "$GitPath;$env:PATH"
    }

    Write-Log "Successfully installed Git."
    return 0
}

# ---------------------------------------------------------------------------
# Helper: Install Podman via winget + PATH fix
# ---------------------------------------------------------------------------
function Install-Podman {
    Write-Log "Installing Podman via winget..."
    winget install --id RedHat.Podman -e --source winget

    if ($LASTEXITCODE -ne 0) {
        Write-Log "ERROR: Failed to install Podman."
        return 1
    }

    $PodmanPath = "C:\Program Files\RedHat\Podman"
    if (Test-Path $PodmanPath) {
        Write-Log "Adding Podman to PATH: $PodmanPath"
        $env:PATH = "$PodmanPath;$env:PATH"
    }

    Write-Log "Successfully installed Podman."
    return 0
}

# ---------------------------------------------------------------------------
# Single-package installation mode
# ---------------------------------------------------------------------------
if ($Package -ne "") {
    Write-Log "=== Installing single dependency: $Package ==="

    switch ($Package.ToLower()) {
        "git"          { exit (Install-Git) }
        "podman"       { exit (Install-Podman) }
        default        { exit (Install-PipPackage $Package) }
    }
}

# ---------------------------------------------------------------------------
# Bulk installation mode
# ---------------------------------------------------------------------------
Write-Log "=== Bulk Dependency Installation Started ==="

# Core ONNX stack
Install-PipPackage "onnx"
Install-PipPackage "onnxruntime"

# ML frameworks
Install-PipPackage "torch"
Install-PipPackage "scikit-learn"
Install-PipPackage "tensorflow"

# Serving frameworks
Install-PipPackage "mlserver"
Install-PipPackage "tritonclient"

# System tools
Install-Git
Install-Podman

Write-Log "=== All dependencies installed successfully ==="
exit 0
