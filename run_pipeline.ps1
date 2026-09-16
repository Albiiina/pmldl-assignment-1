Set-Location $PSScriptRoot

$logFile = Join-Path $PSScriptRoot "pipeline.log"

$venvScripts = Join-Path $PSScriptRoot ".venv\Scripts"
$env:Path = "$venvScripts;$env:Path"

$dvc = Join-Path $venvScripts "dvc.exe"

"========================================" | Out-File $logFile -Append
"$(Get-Date) Starting MLOps pipeline" | Out-File $logFile -Append

& $dvc repro --force >> $logFile 2>&1

$exitCode = $LASTEXITCODE

if ($exitCode -eq 0) {
    "$(Get-Date) Pipeline completed successfully" | Out-File $logFile -Append
    exit 0
}
else {
    "$(Get-Date) Pipeline failed with exit code $exitCode" | Out-File $logFile -Append
    exit $exitCode
}