$ErrorActionPreference = "Stop"

# Always run from the repo root
$scriptDir = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $PSCommandPath }
Set-Location $scriptDir

# Timestamp for log file (Windows-safe format)
$time = Get-Date -Format "yyyy-MM-dd-HH-mm-ss"
$logFile = "$time.log"

# Activate virtual environment if present
$venvCandidates = @(
    (Join-Path $scriptDir ".venv/Scripts/Activate.ps1"),
    (Join-Path $scriptDir "env/Scripts/Activate.ps1")
)
$activated = $false
foreach ($candidate in $venvCandidates) {
    if (Test-Path $candidate) {
        Write-Host "Activating venv: $candidate"
        & $candidate
        $activated = $true
        break
    }
}
if (-not $activated) {
    Write-Warning "未找到虚拟环境，尝试使用系统 python"
}

# Choose python executable
$pythonExe = Join-Path $scriptDir ".venv/Scripts/python.exe"
if (-not (Test-Path $pythonExe)) {
    $pythonExe = Join-Path $scriptDir "env/Scripts/python.exe"
}
if (-not (Test-Path $pythonExe)) {
    $pythonExe = "python"
}

# Set PYTHONPATH
$env:PYTHONPATH = "$scriptDir;$env:PYTHONPATH"

function Run-Step {
    param(
        [string]$Message,
        [string[]]$CommandArgs
    )
    Write-Host "=============================="
    Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $Message"
    Write-Host "Command: $pythonExe $($CommandArgs -join ' ')"
    & $pythonExe @CommandArgs
}

# Preprocess
Run-Step "处理数据 (online)" @("preprocess/data.py", "--mode", "online", "--logfile", $logFile)

# Recall
Run-Step "itemcf 召回 (online)" @("recall/recall_itemcf.py", "--mode", "online", "--logfile", $logFile)
Run-Step "binetwork 召回 (online)" @("recall/recall_binetwork.py", "--mode", "online", "--logfile", $logFile)
Run-Step "w2v 召回 (online)" @("recall/recall_w2v.py", "--mode", "online", "--logfile", $logFile)
Run-Step "召回合并 (online)" @("recall/recall.py", "--mode", "online", "--logfile", $logFile)

# Pointwise
Run-Step "pointwise 特征 (online)" @("rank/pointwise/rank_feature.py", "--mode", "online", "--logfile", $logFile)
Run-Step "pointwise 预测 (online)" @("rank/pointwise/rank_lgb.py", "--mode", "online", "--logfile", $logFile)

# Listwise (optional but kept for completeness; requires trained listwise models)
Run-Step "listwise 特征 (online)" @("rank/listwise/rank_feature_listwise.py", "--mode", "online", "--logfile", $logFile)
Run-Step "listwise 预测 (online)" @("rank/listwise/rank_lambdamart.py", "--mode", "online", "--logfile", $logFile)

Write-Host "完成，提交文件位于 prediction_result/"
