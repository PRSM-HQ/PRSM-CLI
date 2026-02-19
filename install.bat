@echo off
setlocal

set "UPDATE_MODE=0"
set "START_SERVER=1"
set "UNINSTALL=0"

:parse_args
if "%~1"=="" goto :args_done
if /I "%~1"=="--update" (
  set "UPDATE_MODE=1"
  shift
  goto :parse_args
)
if /I "%~1"=="--start-server" (
  set "START_SERVER=1"
  shift
  goto :parse_args
)
if /I "%~1"=="--no-start" (
  set "START_SERVER=0"
  shift
  goto :parse_args
)
if /I "%~1"=="--uninstall" (
  set "UNINSTALL=1"
  shift
  goto :parse_args
)
if /I "%~1"=="--help" goto :show_help
if /I "%~1"=="-h" goto :show_help

echo Unknown arg: %~1
goto :show_help

:args_done
set "PS_SCRIPT=%TEMP%\prsm-install-%RANDOM%.ps1"

> "%PS_SCRIPT%" (
  echo param(
  echo   [int]$UpdateMode = 0,
  echo   [int]$StartServer = 1,
  echo   [int]$Uninstall = 0
  echo )
  echo $ErrorActionPreference = "Stop"
  echo $ProgressPreference = "SilentlyContinue"
  echo.
  echo $repo = "prsm/prism-CLI"
  echo $releaseApi = "https://api.github.com/repos/$repo/releases/latest"
  echo $userAgent = "prsm-installer"
  echo $installRoot = Join-Path $env:USERPROFILE ".prsm"
  echo $venvDir = Join-Path $installRoot "venv"
  echo $binDir = Join-Path $env:USERPROFILE ".local\bin"
  echo $linkPath = Join-Path $binDir "prsm.cmd"
  echo $stateFile = Join-Path $installRoot "install-state.json"
  echo $logDir = Join-Path $installRoot "logs"
  echo $tempDir = Join-Path $env:TEMP "prsm-install"
  echo.
  echo function Write-Info($message) {
  echo   Write-Host "=> $message" -ForegroundColor Cyan
  echo }
  echo function Write-Warn($message) {
  echo   Write-Host "=> $message" -ForegroundColor Yellow
  echo }
  echo function Write-Err($message) {
  echo   Write-Host "=> $message" -ForegroundColor Red
  echo }
  echo.
  echo function Ensure-Python {
  echo   $python = Get-Command py -ErrorAction SilentlyContinue
  echo   if (-not $python) { $python = Get-Command python -ErrorAction SilentlyContinue }
  echo   if (-not $python) { $python = Get-Command python3 -ErrorAction SilentlyContinue }
  echo   if (-not $python) { throw "Python 3 is required." }
  echo   return $python.Source
  echo }
  echo.
  echo function Fetch-Release([string]$python) {
  echo   $headers = @{
  echo     Accept = "application/vnd.github+json"
  echo     "User-Agent" = $userAgent
  echo   }
  echo   return Invoke-RestMethod -Uri $releaseApi -Headers $headers -UseBasicParsing
  echo }
  echo.
  echo function Select-Asset([object]$release, [string]$suffix) {
  echo   foreach ($asset in @($release.assets)) {
  echo     if (($asset.name -as [string]).ToLower().EndsWith($suffix)) {
  echo       return $asset
  echo     }
  echo   }
  echo   return $null
  echo }
  echo.
  echo function Ensure-Directories {
  echo   New-Item -ItemType Directory -Path $installRoot, $binDir, $logDir -Force | Out-Null
  echo   if (-not (Test-Path $tempDir)) { New-Item -ItemType Directory -Path $tempDir | Out-Null }
  echo   return $tempDir
  echo }
  echo.
  echo function Ensure-GlobalModels {
  echo   param([string]$modelsPath)
  echo   if (Test-Path $modelsPath) {
  echo     Write-Info "Global model config already exists at $modelsPath"
  echo     return
  echo   }
  echo   Write-Info "Initializing global model config at $modelsPath"
  echo   $templateUrl = "https://raw.githubusercontent.com/$repo/main/.prism/models.yaml"
  echo   $tmpModels = Join-Path $tempDir "models.yaml"
  echo   try {
  echo     Invoke-WebRequest -Uri $templateUrl -OutFile $tmpModels -ErrorAction Stop
  echo     Copy-Item $tmpModels $modelsPath -Force
  echo     Write-Info "Seeded global model config from repository template."
  echo     return
  echo   } catch {
  echo     Write-Warn "Could not fetch model template from GitHub: $($_.Exception.Message)"
  echo   }
  echo   @"
echo models: {}
echo model_registry: {}
echo "@ | Set-Content -Path $modelsPath -Encoding UTF8
  echo   Write-Warn "Created minimal global models file at $modelsPath"
  echo }
  echo.
  echo function Ensure-Venv([string]$python) {
  echo   $venvPython = Join-Path $venvDir "Scripts\python.exe"
  echo   if (-not (Test-Path $venvPython)) {
  echo     Write-Info "Creating virtual environment in $venvDir"
  echo     & $python -m venv $venvDir
  echo   }
  echo   & $venvPython -m pip install --upgrade pip | Out-Null
  echo   return $venvPython
  echo }
  echo.
  echo function Install-Wheel([string]$url, [string]$python) {
  echo   if (-not $url) { throw "No wheel asset found in release." }
  echo   $wheelFile = Join-Path $tempDir "prsm.whl"
  echo   Write-Info "Downloading wheel"
  echo   Invoke-WebRequest -Uri $url -OutFile $wheelFile
  echo   Write-Info "Installing prsm wheel"
  echo   & $python -m pip install --disable-pip-version-check --no-input $wheelFile | Out-Null
  echo }
  echo.
  echo function Install-Extension([object]$asset) {
  echo   if (-not $asset) {
  echo     Write-Warn "No VSIX asset found in release; skipping extension install."
  echo     return
  echo   }
  echo   $candidates = @("code", "code-insiders", "codium")
  echo   $codeCmd = $null
  echo   foreach ($candidate in $candidates) {
  echo     $cmd = Get-Command $candidate -ErrorAction SilentlyContinue
  echo     if ($cmd) { $codeCmd = $cmd.Source; break }
  echo   }
  echo   if (-not $codeCmd) {
  echo     Write-Warn "VS Code CLI not found; skipping extension install."
  echo     return
  echo   }
  echo   $vsixPath = Join-Path $tempDir "prsm.vsix"
  echo   Invoke-WebRequest -Uri $asset.browser_download_url -OutFile $vsixPath
  echo   Write-Info "Installing VSIX"
  echo   & $codeCmd --install-extension $vsixPath --force | Out-Null
  echo }
  echo.
  echo function Link-Executable {
  echo   if (-not (Test-Path $venvDir)) { throw "Virtual environment not found." }
  echo   $venvPrsm = Join-Path $venvDir "Scripts\prsm.exe"
  echo   if (-not (Test-Path $venvPrsm)) { throw "Installed prsm executable not found." }
  echo   New-Item -ItemType Directory -Path $binDir -Force | Out-Null
  echo   $shim = "@echo off`r`n`"$venvPrsm`" %*"
  echo   Set-Content -Path $linkPath -Value $shim -Encoding ASCII
  echo   return $venvPrsm
  echo }
  echo.
  echo function Find-PrsmServerProcesses {
  echo   $commandPattern = "*--server*"
  echo   try {
  echo     Get-CimInstance Win32_Process -Filter "Name='prsm.exe'" |
  echo     Where-Object { $_.CommandLine -like $commandPattern }
  echo   } catch {
  echo     @()
  echo   }
  echo }
  echo.
  echo function Stop-PrsmServer {
  echo   foreach ($proc in Find-PrsmServerProcesses) {
  echo     try {
  echo       Stop-Process -Id $proc.ProcessId -Force -ErrorAction SilentlyContinue
  echo     } catch {
  echo       # ignore
  echo     }
  echo   }
  echo }
  echo.
  echo function Start-PrsmServer([string]$exePath) {
  echo   if ($StartServer -ne 1) { return }
  echo   if (-not (Test-Path $exePath)) {
  echo     Write-Warn "prsm executable not available; skipping server start."
  echo     return
  echo   }
  echo   $running = Find-PrsmServerProcesses
  echo   if ($running) {
  echo     Write-Info "prsm server already running."
  echo     return
  echo   }
  echo   $outLog = Join-Path $logDir "server-install.log"
  echo   $errLog = Join-Path $logDir "server-install.err.log"
  echo   Start-Process -FilePath $linkPath -ArgumentList "--server" -WindowStyle Hidden -RedirectStandardOutput $outLog -RedirectStandardError $errLog
  echo   Write-Info "prsm server started."
  echo }
  echo.
  echo function Write-State([string]$tag, [string]$wheelUrl, [string]$vsixUrl, [string]$releaseUrl) {
  echo   $state = [ordered]@{
  echo     source = "release"
  echo     repo = $repo
  echo     channel = "stable"
  echo     install_root = $installRoot
  echo     installed_version = $tag
  echo     tag = $tag
  echo     wheel_asset_url = $wheelUrl
  echo     vsix_asset_url = $vsixUrl
  echo     release_url = $releaseUrl
  echo     prsm_command = $linkPath
  echo     venv = $venvDir
  echo     installed_at = (Get-Date).ToUniversalTime().ToString("o")
  echo     last_update_check_at = (Get-Date).ToUniversalTime().ToString("o")
  echo   }
  echo   $state | ConvertTo-Json -Depth 4 | Set-Content -Path $stateFile -Encoding UTF8
  echo }
  echo.
  echo function Remove-Install {
  echo   Write-Warn "Uninstall mode"
  echo   if (Test-Path $linkPath) { Remove-Item $linkPath -Force }
  echo   Stop-PrsmServer
  echo   if (Test-Path $stateFile) { Remove-Item $stateFile -Force }
  echo   Write-Info "Uninstall complete."
  echo   exit 0
  echo }
  echo.
  echo $python = Ensure-Python
  echo Ensure-Directories | Out-Null
  echo if ($Uninstall -eq 1) { Remove-Install }
  echo
  echo $release = Fetch-Release $python
  echo if (-not $release) { throw "Failed to fetch release payload." }
  echo $rawTag = [string]$release.tag_name
  echo $tag = if ($rawTag -like "v*") { $rawTag.Substring(1) } else { $rawTag }
  echo if (-not $tag) { throw "Failed to read release tag." }
  echo Write-Info "Latest release: $tag"
  echo
  echo $wheelAsset = Select-Asset $release ".whl"
  echo $vsixAsset = Select-Asset $release ".vsix"
  echo if (-not $wheelAsset) { throw "No wheel asset found in release payload." }
  echo
  echo if (($UpdateMode -eq 1) -or (Test-Path $linkPath)) { Stop-PrsmServer }
  echo
  echo $venvPython = Ensure-Venv $python
  echo Install-Wheel $wheelAsset.browser_download_url $venvPython
  echo Link-Executable | Out-Null
  echo Ensure-GlobalModels (Join-Path $installRoot "models.yaml")
  echo Install-Extension $vsixAsset
  echo Write-State $tag $wheelAsset.browser_download_url ($vsixAsset.browser_download_url -as [string]) $release.html_url
  echo Start-PrsmServer $venvPython
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%PS_SCRIPT%" %UPDATE_MODE% %START_SERVER% %UNINSTALL%
set "PSRC=%ERRORLEVEL%"

if exist "%PS_SCRIPT%" del /q "%PS_SCRIPT%"
exit /b %PSRC%

:show_help
echo Install / update PRSM from GitHub release assets (wheel + VSIX).
echo.
echo Usage:
echo   install.bat --update --start-server
echo.
echo Options:
echo   --update        Re-run install/update flow (idempotent)
echo   --start-server  Start prsm --server after install (default on)
echo   --no-start      Do not start prsm --server
echo   --uninstall     Remove installed symlink/state
echo   --help          Show this help text
