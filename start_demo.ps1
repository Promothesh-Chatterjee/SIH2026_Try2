# PowerShell quick launcher for Cognitive EW SmartScan SIH Demo
param (
    [string]$scenario = "config_29",
    [double]$speed = 15.0,
    [switch]$noBrowser
)

$cmdArgs = @("start_sih_demo.py", "--scenario", $scenario, "--speed", $speed)
if ($noBrowser) {
    $cmdArgs += "--no-browser"
}

Write-Host "[*] Launching Cognitive EW SmartScan Demonstration with Gate-25k-R4.2-alpha020..." -ForegroundColor Cyan
python @cmdArgs
