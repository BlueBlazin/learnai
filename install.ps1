# Installs or updates Learn AI on Windows:
#
#   irm https://raw.githubusercontent.com/BlueBlazin/learnai/master/install.ps1 | iex
#
# Everything goes into %USERPROFILE%\.learnai: the app, the problems, a private uv, the
# Python it downloads, and the libraries. The only things outside it are a Start Menu
# shortcut and the folders Windows keeps for the app (settings, logs, web cache).
# %USERPROFILE%\.learnai\uninstall.cmd removes all of it. No administrator rights, no
# change to PATH or to the registry.
#
# Running it again updates the app and the problems and keeps your solutions and
# progress.
#
# Environment:
#   LEARNAI_VERSION        a release tag to install instead of the latest, like v0.1.0
#   LEARNAI_DOWNLOAD_BASE  where the release files are: a URL, or a folder (the
#                          installer's own test passes a folder)

function Install-LearnAI {
    $ErrorActionPreference = 'Stop'
    $ProgressPreference = 'SilentlyContinue'
    [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

    $repo = 'BlueBlazin/learnai'
    $appName = 'Learn AI'
    $archive = 'learnai-x86_64-pc-windows-msvc.zip'
    # The extras the app can use. `mlx` installs nothing off Apple silicon.
    $extras = @('--extra', 'editor', '--extra', 'jax', '--extra', 'mlx', '--extra', 'models')

    function Step($text) { Write-Host ''; Write-Host "==> $text" }

    $arch = $env:PROCESSOR_ARCHITEW6432
    if (-not $arch) { $arch = $env:PROCESSOR_ARCHITECTURE }
    if ($arch -ne 'AMD64') {
        throw "Windows on $arch is not supported: PyTorch publishes a build for x64 only."
    }
    if (-not $env:USERPROFILE) { throw 'USERPROFILE is not set.' }
    $dest = Join-Path $env:USERPROFILE '.learnai'

    # The window is a WebView2 web view. Windows 11 includes the runtime; it is the one
    # thing this script cannot put in the folder, so it stops and says where to get it.
    $webview = '{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}'
    $found = $false
    foreach ($key in @(
            "HKLM:\SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\$webview",
            "HKLM:\SOFTWARE\Microsoft\EdgeUpdate\Clients\$webview",
            "HKCU:\Software\Microsoft\EdgeUpdate\Clients\$webview")) {
        $pv = (Get-ItemProperty -Path $key -Name pv -ErrorAction SilentlyContinue).pv
        if ($pv -and $pv -ne '0.0.0.0') { $found = $true }
    }
    if (-not $found) {
        throw "Learn AI's window needs the Microsoft Edge WebView2 Runtime, which is not installed. Install it from https://developer.microsoft.com/microsoft-edge/webview2/ and run this installer again."
    }

    $exe = Join-Path $dest 'app\learnai.exe'
    $running = Get-Process -Name learnai -ErrorAction SilentlyContinue | Where-Object { $_.Path -eq $exe }
    if ($running) { throw 'Learn AI is running: close it, then run this installer again.' }

    if ($env:LEARNAI_DOWNLOAD_BASE) {
        $base = $env:LEARNAI_DOWNLOAD_BASE
    } elseif ($env:LEARNAI_VERSION) {
        $base = "https://github.com/$repo/releases/download/$($env:LEARNAI_VERSION)"
    } else {
        $base = "https://github.com/$repo/releases/latest/download"
    }

    $versionFile = Join-Path $dest 'version'
    if (Test-Path $versionFile) {
        Step "Updating Learn AI in $dest (installed: $((Get-Content $versionFile -Raw).Trim()))"
    } else {
        Step "Installing Learn AI into $dest"
    }

    $stage = Join-Path $dest '.staging'
    if (Test-Path $stage) { Remove-Item -Recurse -Force $stage }
    New-Item -ItemType Directory -Force -Path $stage | Out-Null

    function Fetch($name) {
        $out = Join-Path $stage $name
        if (Test-Path -PathType Container $base) {
            Copy-Item (Join-Path $base $name) $out
        } else {
            Invoke-WebRequest -UseBasicParsing -Uri "$base/$name" -OutFile $out
        }
        return $out
    }

    # Replaces "$to\<name>" with "$from\<name>" for each name.
    function Replace-Entries($from, $to, $names) {
        New-Item -ItemType Directory -Force -Path $to | Out-Null
        foreach ($name in $names) {
            $source = Join-Path $from $name
            if (-not (Test-Path $source)) { continue }
            $target = Join-Path $to $name
            if (Test-Path $target) { Remove-Item -Recurse -Force $target }
            Move-Item $source $target
        }
    }

    $saved = @{}
    $private = @{
        UV_CACHE_DIR          = Join-Path $dest 'uv\cache'
        UV_PYTHON_INSTALL_DIR = Join-Path $dest 'uv\python'
        UV_MANAGED_PYTHON     = '1'
        UV_PYTHON_INSTALL_BIN = '0'
        UV_PYTHON_INSTALL_REGISTRY = '0'
    }
    try {
        Step "Downloading $archive"
        $zip = Fetch $archive
        $sums = Fetch 'SHA256SUMS'
        $expected = $null
        foreach ($line in Get-Content $sums) {
            $parts = $line -split '\s+', 2
            if ($parts.Count -eq 2 -and $parts[1].TrimStart('*') -eq $archive) { $expected = $parts[0].ToLower() }
        }
        if (-not $expected) { throw "SHA256SUMS has no entry for $archive." }
        $actual = (Get-FileHash -Algorithm SHA256 $zip).Hash.ToLower()
        if ($actual -ne $expected) {
            throw "The download is damaged: its checksum is $actual, expected $expected."
        }

        Step 'Unpacking'
        $tar = Join-Path $env:SystemRoot 'System32\tar.exe'
        if (Test-Path $tar) {
            & $tar -xf $zip -C $stage
            if ($LASTEXITCODE -ne 0) { throw 'The download could not be unpacked.' }
        } else {
            Expand-Archive -Path $zip -DestinationPath $stage
        }
        $new = Join-Path $stage 'learnai'
        if (-not (Test-Path (Join-Path $new 'version'))) { throw 'The download is not a Learn AI release.' }

        # No version on disk while the files are being replaced: a version means a
        # complete install.
        if (Test-Path $versionFile) { Remove-Item -Force $versionFile }
        Replace-Entries $new $dest @('app')
        # `python\.venv` stays where it is: uv brings it up to date below.
        Replace-Entries (Join-Path $new 'python') (Join-Path $dest 'python') @('src', 'pyproject.toml', 'uv.lock', '.python-version', 'README.md')
        Replace-Entries $new $dest @('areas', 'problems', 'bin', 'licenses', 'LICENSE', 'uninstall.ps1', 'uninstall.cmd')
        New-Item -ItemType Directory -Force -Path (Join-Path $dest 'workspace') | Out-Null

        $programs = Join-Path $env:APPDATA 'Microsoft\Windows\Start Menu\Programs'
        New-Item -ItemType Directory -Force -Path $programs | Out-Null
        $shortcut = (New-Object -ComObject WScript.Shell).CreateShortcut((Join-Path $programs "$appName.lnk"))
        $shortcut.TargetPath = $exe
        $shortcut.WorkingDirectory = $dest
        $shortcut.Save()

        Step 'Installing Python and the libraries (a large download the first time)'
        # The same variables the app runs uv with: `private_envs` in crates/core/src/uv.rs.
        foreach ($name in $private.Keys) {
            $saved[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
            [Environment]::SetEnvironmentVariable($name, $private[$name], 'Process')
        }
        $uv = Join-Path $dest 'bin\uv.exe'
        & $uv sync --project (Join-Path $dest 'python') --locked @extras
        if ($LASTEXITCODE -ne 0) {
            throw 'The libraries could not be installed; run this installer again to retry.'
        }

        # Last, so that a version on disk means a complete install.
        Replace-Entries $new $dest @('version')

        Step "Learn AI $((Get-Content $versionFile -Raw).Trim()) is installed"
        Write-Host "Open it:    Learn AI in the Start Menu   (or $exe)"
        Write-Host 'Update:     run this installer again'
        Write-Host "Uninstall:  $(Join-Path $dest 'uninstall.cmd')"
    } finally {
        foreach ($name in $saved.Keys) {
            [Environment]::SetEnvironmentVariable($name, $saved[$name], 'Process')
        }
        if (Test-Path $stage) { Remove-Item -Recurse -Force $stage -ErrorAction SilentlyContinue }
    }
}

Install-LearnAI
