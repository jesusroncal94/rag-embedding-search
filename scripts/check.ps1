# The one check agents and CI run before calling a change done (Windows PowerShell wrapper).
$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')

& bash scripts/check.sh
exit $LASTEXITCODE
