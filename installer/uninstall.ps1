# Removes Learn AI from Windows. Run it through uninstall.cmd next to it, which reads
# this file into memory and leaves its own batch context first (`(goto) 2>nul`), so that
# both files can be deleted while they run:
#
#   %USERPROFILE%\.learnai\uninstall.cmd [-KeepWork | -DeleteWork]
#
# It removes %USERPROFILE%\.learnai, the Start Menu shortcut, and the folders Windows
# keeps for the app (settings, logs, web cache). Your solutions and progress
# (.learnai\workspace and .learnai\.learnai\state.db) are deleted only if you say so:
# with neither option it asks, and when it cannot ask it keeps them.

param(
    [switch]$KeepWork,
    [switch]$DeleteWork
)

$ErrorActionPreference = 'Stop'
$appId = 'dev.learnai.app'
$appName = 'Learn AI'

if (-not $env:USERPROFILE) { throw 'USERPROFILE is not set.' }
$dest = Join-Path $env:USERPROFILE '.learnai'
$exe = Join-Path $dest 'app\learnai.exe'

$running = Get-Process -Name learnai -ErrorAction SilentlyContinue | Where-Object { $_.Path -eq $exe }
if ($running) { throw 'Learn AI is running: close it, then run this again.' }

$workspace = Join-Path $dest 'workspace'
$state = Join-Path $dest '.learnai'
$hasWork = (Test-Path (Join-Path $state 'state.db')) -or
    ((Test-Path $workspace) -and (Get-ChildItem -Recurse -File -Force $workspace | Select-Object -First 1))

$work = 'ask'
if ($KeepWork) { $work = 'keep' }
if ($DeleteWork) { $work = 'delete' }
if (-not $hasWork) {
    $work = 'delete'
} elseif ($work -eq 'ask') {
    $work = 'keep'
    if ([Environment]::UserInteractive -and -not [Console]::IsInputRedirected) {
        $answer = Read-Host 'Delete your solutions and progress too? [y/N]'
        if ($answer -match '^(y|yes)$') { $work = 'delete' }
    }
}

# Nothing can be removed from a folder that is the current one.
Set-Location $env:USERPROFILE

$shortcut = Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs\$appName.lnk"
foreach ($path in @($shortcut, (Join-Path $env:APPDATA $appId), (Join-Path $env:LOCALAPPDATA $appId))) {
    if (Test-Path $path) { Remove-Item -Recurse -Force $path }
}

if ($work -eq 'delete') {
    if (Test-Path $dest) { Remove-Item -Recurse -Force $dest }
    Write-Host 'Learn AI is removed.'
    return
}

# Keep the solutions and the progress database; remove everything else.
foreach ($entry in Get-ChildItem -Force $dest) {
    if ($entry.Name -ne 'workspace' -and $entry.Name -ne '.learnai') {
        Remove-Item -Recurse -Force $entry.FullName
    }
}
if (Test-Path $state) {
    foreach ($entry in Get-ChildItem -Force $state) {
        if ($entry.Name -notlike 'state.db*') { Remove-Item -Recurse -Force $entry.FullName }
    }
}
Write-Host "Learn AI is removed. Your solutions and progress are kept in $dest."
Write-Host 'Installing again picks them up. To delete them, delete that folder.'
