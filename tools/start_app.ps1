param([switch]$NoBrowser)
$ErrorActionPreference = 'Stop'
$project = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$python = Join-Path $project '.venv\Scripts\python.exe'
$address = 'http://127.0.0.1:7860'
function Get-Ready {
    try {
        $response = Invoke-RestMethod "$address/api/health" -TimeoutSec 2
        return $response.app -eq 'neuroscope-mri' -and $response.ready -eq $true
    } catch { return $false }
}
if (Get-Ready) {
    Write-Host 'NeuroScope is already running.'
    if (-not $NoBrowser) { Start-Process $address }
    exit 0
}
if (-not (Test-Path $python)) { throw "Project environment missing at $python. See README.md." }
& $python -c 'import fastapi, uvicorn, tensorflow, keras, gradio, PIL, h5py'
if ($LASTEXITCODE -ne 0) { throw 'Essential Python dependencies are missing. Install requirements.txt.' }
$listener = Get-NetTCPConnection -LocalPort 7860 -State Listen -ErrorAction SilentlyContinue
if ($listener) { throw "Port 7860 belongs to process $($listener.OwningProcess). Stop it or use another port." }
Write-Host 'Starting NeuroScope and loading the model...'
$service = Start-Process -FilePath $python -ArgumentList 'app.py' -WorkingDirectory $project -PassThru -WindowStyle Hidden -RedirectStandardOutput (Join-Path $project 'tmp_launcher.log') -RedirectStandardError (Join-Path $project 'tmp_launcher_err.log')
for ($attempt = 0; $attempt -lt 60; $attempt++) {
    if (Get-Ready) {
        if (-not $NoBrowser) { Start-Process $address }
        Write-Host "Application ready at $address"
        exit 0
    }
    if ($service.HasExited) { throw "Server exited with code $($service.ExitCode). Read tmp_launcher_err.log." }
    Start-Sleep -Seconds 1
}
throw 'Startup timed out. Read tmp_launcher_err.log.'


