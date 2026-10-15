param(
    [string]$Python = 'python',
    [ValidateSet('unittest', 'pytest')][string]$Framework = 'unittest',
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$TestArguments
)

$ErrorActionPreference = 'Stop'
$taskRepo = Split-Path -Parent $PSScriptRoot
$taskTemporary = Join-Path $PSScriptRoot ('temporary\session_' + $PID + '_' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Force -Path $taskTemporary | Out-Null
$taskOldTemp = $env:TEMP
$taskOldTmp = $env:TMP
$taskOldBytecode = $env:PYTHONDONTWRITEBYTECODE
$taskOldLocation = (Get-Location).Path
try {
    $env:TEMP = $taskTemporary
    $env:TMP = $taskTemporary
    $env:PYTHONDONTWRITEBYTECODE = '1'
    Set-Location -LiteralPath $taskRepo
    & $Python -m $Framework @TestArguments
    $taskExitCode = $LASTEXITCODE
}
finally {
    $env:TEMP = $taskOldTemp
    $env:TMP = $taskOldTmp
    $env:PYTHONDONTWRITEBYTECODE = $taskOldBytecode
    Set-Location -LiteralPath $taskOldLocation
}
exit $taskExitCode
