$PSScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $PSScriptRoot

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (Test-Path $python) {
    & $python server.py
    exit $LASTEXITCODE
}

uv run server.py
exit $LASTEXITCODE