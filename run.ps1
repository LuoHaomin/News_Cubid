$ErrorActionPreference = "Stop"

# Always run from the repo root
$scriptDir = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $PSCommandPath }
Set-Location $scriptDir

# Timestamp for log file (Windows-safe format)
$time = Get-Date -Format "yyyy-MM-dd-HH-mm-ss"

# Activate virtual environment if present
$venvCandidates = @(
    (Join-Path $scriptDir ".venv/Scripts/Activate.ps1")
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
        [string]$message,
        [string]$script,
        [string[]]$extraArgs = @()
    )
    Write-Host $message
    & $pythonExe $script --mode valid --logfile "$time.log" @extraArgs
}

Run-Step "处理数据" "preprocess/data.py"
Run-Step "itemcf 召回" "recall/recall_itemcf.py"
Run-Step "binetwork 召回" "recall/recall_binetwork.py"
Run-Step "w2v 召回" "recall/recall_w2v.py"
Run-Step "召回合并" "recall/recall.py"
Run-Step "排序特征" "rank/pointwise/rank_feature.py"
Run-Step "lgb 模型训练" "rank/pointwise/rank_lgb.py"
Run-Step "listwise 特征工程" "rank/listwise/rank_feature_listwise.py"
Run-Step "listwise 模型训练" "rank/listwise/rank_lambdamart.py"
# Run-Step "listwise 在线预测" "rank/listwise/rank_lambdamart.py" @("--mode", "online")
