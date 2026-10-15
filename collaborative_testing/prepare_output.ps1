param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[A-Za-z0-9_]+$')]
    [string]$Experiment,
    [ValidateSet('results_archive', 'results', 'outputs', '__pycache__')]
    [string]$Kind = 'results_archive'
)
$ErrorActionPreference = 'Stop'
$taskRepo = Split-Path -Parent $PSScriptRoot
$taskSource = Join-Path (Join-Path $taskRepo 'collaborative_test') $Experiment
if (-not (Test-Path -LiteralPath $taskSource -PathType Container)) {
    throw "Experiment source directory is absent: $taskSource"
}
$taskLogical = [IO.Path]::GetFullPath((Join-Path $taskSource $Kind))
$taskPhysical = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ("generated\collaborative_test\" + $Experiment + '\' + $Kind)))
if (-not $taskLogical.StartsWith($taskRepo + '\', [StringComparison]::OrdinalIgnoreCase) -or
    -not $taskPhysical.StartsWith($PSScriptRoot + '\', [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Output path escaped the intended repository'
}
if (Test-Path -LiteralPath $taskLogical) {
    $taskItem = Get-Item -LiteralPath $taskLogical
    if ($taskItem.LinkType -ne 'Junction' -or @($taskItem.Target)[0] -ne $taskPhysical) {
        throw "Existing output needs an explicit migration: $taskLogical"
    }
} else {
    New-Item -ItemType Directory -Force -Path $taskPhysical | Out-Null
    New-Item -ItemType Junction -Path $taskLogical -Target $taskPhysical | Out-Null
}
Write-Output $taskPhysical
