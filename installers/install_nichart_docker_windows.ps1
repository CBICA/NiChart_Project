<# 
  windows_install.ps1

  Usage:
    .\windows_install.ps1 C:\DATAPATH [--distro Ubuntu]

  Notes:
    - Requires WSL installed.
    - Creates a Desktop shortcut that invokes a WSL script.
#>

[CmdletBinding()]
param(
  [Parameter(Mandatory=$true, Position=0)]
  [string]$DataPath,

  [Parameter(Mandatory=$false)]
  [string]$distro
)

function Fail($msg, [int]$code=1) {
  Write-Error $msg
  exit $code
}

function Assert-WSLPresent {
  $wslExe = Join-Path $env:WINDIR 'System32\wsl.exe'
  if (-not (Test-Path $wslExe)) { Fail "WSL not found at $wslExe. Please install WSL (wsl --install) and try again." }
  return $wslExe
}

function Get-DefaultDistro {
  # 'wsl -l -v' marks default with a '*'
  $out = & $script:wslExe -l -v 2>$null
  if ($LASTEXITCODE -ne 0 -or -not $out) {
    # fallback: 'wsl -l' may include '(Default)'
    $out = & $script:wslExe -l 2>$null
  }
  if (-not $out) { Fail "Unable to query WSL distros. Is WSL installed and initialized?" }
  foreach ($line in $out) {
    if ($line -match '^\*?\s*([^\s].*?)(\s+\(Default\))?\s+(\d+)?' -or
        $line -match '^\*\s+(.+)$') {
      $name = $Matches[1]
      if ($line -like '*Default*' -or $line -like '*`**') { return $name.Trim() }
    }
  }
  # If no marker found, pick the first non-header line from `wsl -l -q`
  $q = & $script:wslExe -l -q 2>$null | Where-Object { $_ -and $_.Trim() -ne '' }
  if (-not $q) { Fail "No WSL distributions found. Install a distro from the Store and retry." }
  return $q[0].Trim()
}

function Convert-ToWslPath([string]$winPath) {
  try {
    $full = (Resolve-Path -LiteralPath $winPath).Path
  } catch {
    Fail "Provided path '$winPath' does not exist. Create it first or provide an existing path."
  }
  # Handle UNC or local drive paths
  if ($full -match '^[A-Za-z]:(\\.*)?$') {
    $drive = $full.Substring(0,1).ToLower()
    $rest  = $full.Substring(2).Replace('\','/').TrimStart('/')
    return "/mnt/$drive/$rest"
  } elseif ($full -like '\\*') {
    # UNC: //server/share/path -> /mnt/unc/server/share/path (convention)
    $clean = $full.TrimStart('\').Replace('\','/')
    return "/mnt/unc/$clean"
  } else {
    Fail "Unsupported path format: $full"
  }
}

# --- Main ---
$ErrorActionPreference = 'Stop'
$wslExe = Assert-WSLPresent
$selectedDistro = if ($PSBoundParameters.ContainsKey('distro') -and $distro) { $distro } else { Get-DefaultDistro }
Write-Host "Using WSL distro: $selectedDistro"

$wslDataPath = Convert-ToWslPath $DataPath
Write-Host "Windows path: $DataPath"
Write-Host "WSL path:     $wslDataPath"

# Where app/script will live inside WSL
$AppDir         = "/home/\$USER/NiChart"
$RunScriptPath  = "$AppDir/run_nichart.sh"
$InstallLogPath = "$AppDir/install.log"

# Compose the bash payload to run inside the distro
$bashBody = @"
set -euo pipefail

# Example: prepare app dir and a tiny demo run script if it doesn't exist
mkdir -p "$AppDir"

# Persist the provided data path for your app to read later
printf '%s\n' "$wslDataPath" > "$AppDir/DATA_PATH"

### INSTALL STEPS ###
# Get current installer from NiChart_Project and run it
cd "$AppDir"
wget https://raw.githubusercontent.com/CBICA/NiChart_Project/main/installers/install_nichart_docker_linux.sh
chmod +x install_nichart_docker_linux.sh
./install_nichart_docker_linux.sh "$wslDataPath"
chmod +x run_nichart.sh
######################

echo "\$(date -Is) Install completed. Data path: $wslDataPath" >> "$InstallLogPath"
"@

# Invoke bash -lc "<payload>" in the chosen distro
Write-Host "Running installer steps inside WSL..."
& $wslExe -d $selectedDistro -- bash -lc $bashBody
if ($LASTEXITCODE -ne 0) { Fail "Installer failed inside WSL (exit code $LASTEXITCODE)." }

# Verify the runtime script exists (and is executable if possible)
$verifyCmd = "[ -f '$RunScriptPath' ] || exit 99; [ -x '$RunScriptPath' ] || exit 98"
& $wslExe -d $selectedDistro -- bash -lc $verifyCmd
switch ($LASTEXITCODE) {
  0   { }  # ok
  98  { Write-Warning "Runtime script exists but is not executable: $RunScriptPath. The shortcut may still work if invoked via bash." }
  99  { Fail "Your installer did not create the runtime script at: $RunScriptPath" }
  default { Fail "Unexpected error checking runtime script (exit $LASTEXITCODE)." }
}

# Create a Windows shortcut on the Desktop that launches the WSL script
$desktop = [Environment]::GetFolderPath('Desktop')
$shortcutName = "NiChart.lnk"
$shortcutPath = Join-Path $desktop $shortcutName

$WshShell = New-Object -ComObject WScript.Shell
$shortcut = $WshShell.CreateShortcut($shortcutPath)

# Target: wsl.exe
$shortcut.TargetPath = (Join-Path $env:WINDIR 'System32\wsl.exe')

# Arguments: optionally include -d "<distro>", then invoke /bin/bash -lc '<script>'
# Use single quotes around the script path to handle spaces safely.
$argList = @()
if ($selectedDistro) { $argList += @('-d', $selectedDistro) }
$argList += @('--','/bin/bash','-lc', "'$RunScriptPath'")

# Join with spaces and proper quoting
# (WScript will pass this verbatim to CreateProcess; wsl handles the rest.)
$shortcut.Arguments = ($argList -join ' ')
$shortcut.WorkingDirectory = $desktop
$shortcut.WindowStyle = 1
$shortcut.Description = "Launch NiChart inside $selectedDistro"
$shortcut.IconLocation = "$env:SystemRoot\System32\wsl.exe,0"
$shortcut.Save()

if (-not (Test-Path $shortcutPath)) { Fail "Failed to create shortcut at $shortcutPath." }

Write-Host ""
Write-Host "✅ Install complete."
Write-Host "   WSL distro:     $selectedDistro"
Write-Host "   WSL data path:  $wslDataPath"
Write-Host "   Run script:     $RunScriptPath (inside WSL)"
Write-Host "   Shortcut:       $shortcutPath"
Write-Host ""
Write-Host "Tip: double-click the shortcut on the desktop to launch the NiChart application."
