@cd /d "%USERPROFILE%" & powershell -NoProfile -ExecutionPolicy Bypass -Command "& ([scriptblock]::Create((Get-Content -Raw -LiteralPath '%~dp0uninstall.ps1'))) %*" & exit /b
