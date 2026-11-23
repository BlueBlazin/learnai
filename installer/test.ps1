# The installer's own test on Windows, run by .github/workflows/release.yml:
#
#   installer\test.ps1 -Files <folder with the release files> [-Shots <folder for screenshots>]
#
# It installs from the folder, checks the Python environment and judges one problem,
# opens the app and waits for it to load the catalog, updates, uninstalls, and then
# fails if anything new is left in the user profile or in the Python registry key. It
# works on %USERPROFILE%, so run it on a machine that can be thrown away.

param(
    [Parameter(Mandatory = $true)][string]$Files,
    [string]$Shots
)

$ErrorActionPreference = 'Stop'
$Files = (Resolve-Path $Files).Path
$repo = Split-Path -Parent $PSScriptRoot
$dest = Join-Path $env:USERPROFILE '.learnai'
$appId = 'dev.learnai.app'
$exe = Join-Path $dest 'app\learnai.exe'
$log = Join-Path $env:LOCALAPPDATA "$appId\logs\learnai.log"
$shortcut = Join-Path $env:APPDATA 'Microsoft\Windows\Start Menu\Programs\Learn AI.lnk'
$systemDirs = @((Join-Path $env:APPDATA $appId), (Join-Path $env:LOCALAPPDATA $appId))

# Not ours: what Windows and its libraries write for any program that opens a window.
$ignore = @(
    '\\AppData\\Local\\Temp(\\|$)',
    '\\AppData\\Local\\Microsoft(\\|$)',
    '\\AppData\\Local\\D3DSCache(\\|$)',
    '\\AppData\\LocalLow\\Microsoft(\\|$)',
    '\\AppData\\Roaming\\Microsoft\\Windows\\Recent(\\|$)'
)

function Step($text) { Write-Host ''; Write-Host "=== $text" }
function Fail($text) { throw "FAILED: $text" }
function Must-Exist($path) { if (-not (Test-Path -LiteralPath $path)) { Fail "$path is missing" } }
function Must-Not-Exist($path) { if (Test-Path -LiteralPath $path) { Fail "$path is still there" } }
function Check-Exit($what) { if ($LASTEXITCODE -ne 0) { Fail "$what exited with $LASTEXITCODE" } }

# The user profile, four levels deep, and the registry key Python installs announce
# themselves in.
function Snapshot {
    $paths = Get-ChildItem -LiteralPath $env:USERPROFILE -Force -Recurse -Depth 3 -ErrorAction SilentlyContinue |
        ForEach-Object { $_.FullName }
    $registry = & reg query 'HKCU\Software\Python' /s 2>$null | Where-Object { $_ -match '^HKEY' }
    $global:LASTEXITCODE = 0
    return @($paths) + @($registry) | Sort-Object -Unique
}

function Run-Install {
    $env:LEARNAI_DOWNLOAD_BASE = $Files
    # As `irm ... | iex` runs it.
    Invoke-Expression (Get-Content -Raw (Join-Path $repo 'install.ps1'))
}

# uv as the app runs it (`private_envs` in crates/core/src/uv.rs).
$env:UV_CACHE_DIR = Join-Path $dest 'uv\cache'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $dest 'uv\python'
$env:UV_MANAGED_PYTHON = '1'
$env:UV_PYTHON_INSTALL_BIN = '0'
$env:UV_PYTHON_INSTALL_REGISTRY = '0'
$env:HF_HOME = Join-Path $dest '.learnai\cache\huggingface'
$env:TORCH_HOME = Join-Path $dest '.learnai\cache\torch'
$env:LEARNAI_ROOT = $dest
$uv = Join-Path $dest 'bin\uv.exe'
$project = Join-Path $dest 'python'

$before = Snapshot

Step 'Install'
Run-Install
foreach ($path in @('version', 'bin\uv.exe', 'python\.venv', 'python\uv.lock', 'problems', 'areas', 'workspace', 'uninstall.ps1', 'uninstall.cmd', 'LICENSE', 'app\learnai.exe')) {
    Must-Exist (Join-Path $dest $path)
}
Must-Exist $shortcut
Must-Not-Exist (Join-Path $dest '.staging')

Step 'The Python environment'
& $uv run --project $project --no-sync -- python -c "import sys, torch, jax, numpy; print(sys.version); print('torch', torch.__version__, 'jax', jax.__version__)"
Check-Exit 'importing the libraries'
& $uv run --project $project --no-sync -- learnai doctor --json
Check-Exit 'learnai doctor'
$python = Join-Path $dest 'python\.venv\Scripts\python.exe'
$base = & $python -c 'import sys; print(sys.base_prefix)'
if (-not $base.StartsWith((Join-Path $dest 'uv\python'), [StringComparison]::OrdinalIgnoreCase)) {
    Fail "the environment's Python is $base, not one inside $dest"
}

Step 'Judge one problem'
$problem = Get-ChildItem -Recurse -Filter problem.toml (Join-Path $dest 'problems') |
    ForEach-Object { $_.Directory } | Sort-Object FullName | Select-Object -First 1
& $uv run --project $project --no-sync -- learnai validate $problem.Name
Check-Exit 'learnai validate'

Step 'Open the app'
$problems = @(Get-ChildItem -Recurse -Filter problem.toml (Join-Path $dest 'problems')).Count
$env:LEARNAI_LOG = 'debug'
$app = Start-Process -FilePath $exe -PassThru
$waited = 0
while (-not ((Test-Path $log) -and (Select-String -Path $log -Pattern 'catalog loaded' -Quiet))) {
    if ($app.HasExited) { Fail 'the app exited before it loaded the catalog' }
    if ($waited -ge 90) { Fail 'the app did not load the catalog in 90 seconds' }
    Start-Sleep -Seconds 1
    $waited++
}
$root = (Select-String -Path $log -Pattern 'content root' | Select-Object -Last 1).Line
$loaded = (Select-String -Path $log -Pattern 'catalog loaded' | Select-Object -Last 1).Line
Write-Host $root
Write-Host $loaded
if ($root -notmatch 'Installed') { Fail 'the app did not take its content from the installed folder' }
if ($loaded -notmatch "problems=$problems ") { Fail "the app did not load all $problems problems" }
# Time for the window to draw and ask the core for its first screen.
Start-Sleep -Seconds 20
if ($app.HasExited) { Fail 'the app exited after it loaded the catalog' }
if ($Shots) {
    New-Item -ItemType Directory -Force -Path $Shots | Out-Null
    try {
        Add-Type -AssemblyName System.Windows.Forms, System.Drawing
        $bounds = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds
        $bitmap = New-Object System.Drawing.Bitmap $bounds.Width, $bounds.Height
        $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
        $graphics.CopyFromScreen($bounds.Location, [System.Drawing.Point]::Empty, $bounds.Size)
        $bitmap.Save((Join-Path $Shots 'Windows-x64.png'))
    } catch {
        Write-Host "no screenshot: $_"
    }
    Copy-Item $log (Join-Path $Shots 'Windows-x64.log')
}
& taskkill /PID $app.Id /T /F | Out-Null
$global:LASTEXITCODE = 0
Start-Sleep -Seconds 5
# The app must not have removed the libraries it did not know were installed.
& $python -c 'import torch, jax'
Check-Exit 'importing the libraries after the app ran'

Step 'Update'
$kept = Join-Path $dest 'workspace\kept\torch.py'
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $kept) | Out-Null
Set-Content -Path $kept -Value 'x = 1'
Run-Install
Must-Exist $kept
Must-Exist (Join-Path $dest 'version')
Must-Exist (Join-Path $dest '.learnai\state.db')

Step 'Uninstall, keeping the work'
& cmd /c (Join-Path $dest 'uninstall.cmd') -KeepWork
Check-Exit 'uninstall.cmd -KeepWork'
Must-Exist $kept
Must-Exist (Join-Path $dest '.learnai\state.db')
foreach ($path in @('version', 'bin', 'python', 'problems', 'areas', 'uv', 'app', 'uninstall.ps1', 'uninstall.cmd')) {
    Must-Not-Exist (Join-Path $dest $path)
}
Must-Not-Exist $shortcut
foreach ($dir in $systemDirs) { Must-Not-Exist $dir }

Step 'Uninstall, deleting the work'
& powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $repo 'installer\uninstall.ps1') -DeleteWork
Check-Exit 'uninstall.ps1 -DeleteWork'
Must-Not-Exist $dest

Step 'What is left in the user profile'
$after = Snapshot
$seen = New-Object 'System.Collections.Generic.HashSet[string]' ([StringComparer]::OrdinalIgnoreCase)
foreach ($entry in $before) { [void]$seen.Add($entry) }
$left = $after | Where-Object { -not $seen.Contains($_) } | Where-Object {
    $path = $_
    -not ($ignore | Where-Object { $path -match $_ })
}
if ($left) {
    $left | ForEach-Object { Write-Host $_ }
    Fail 'the paths above are new in the user profile'
}
Write-Host 'nothing'
Write-Host ''
Write-Host 'PASSED'
