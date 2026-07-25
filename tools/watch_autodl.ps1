[CmdletBinding()]
param(
    [ValidateSet("E1", "E2", "E3", "E4", "M6", "M7", "M8", "Extensions")]
    [string]$Task = "E1",

    [ValidateRange(2, 3600)]
    [int]$IntervalSeconds = 10,

    [switch]$Once,

    [string]$WslDistribution = "Ubuntu",
    [string]$WslProjectPath = "/home/mountainatic/modeling_school/zhangsproject",
    [string]$RemoteHost = "connect.westd.seetacloud.com",
    [int]$RemotePort = 17623,
    [string]$RemoteUser = "root"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$expectedHostKey = "SHA256:liZ36vNCsNcNdXeWs4f+g5ZIhPM/ZihP834vxs8Ulqc"
$statusScript = switch ($Task) {
    "E1" { "deploy/autodl/status_e1.sh" }
    "E2" { "deploy/autodl/status.sh" }
    "E3" { "deploy/autodl/status_e3.sh" }
    "E4" { "deploy/autodl/status_e4.sh" }
    "M6" { "deploy/autodl/status_m6.sh" }
    "M7" { "deploy/autodl/status_m7.sh" }
    "M8" { "deploy/autodl/status_m8.sh" }
    "Extensions" { "deploy/autodl/status_phase1_extensions.sh" }
}
$remoteCommand = "cd /root/AR_RAPHU_AUTODL; bash $statusScript"
$passwordWasAlreadySet = Test-Path Env:AR_RAPHU_DEPLOY_PASSWORD
$previousWslEnv = $env:WSLENV

if (-not $passwordWasAlreadySet) {
    $securePassword = Read-Host "AutoDL SSH password" -AsSecureString
    $passwordPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR(
        $securePassword
    )
    try {
        $env:AR_RAPHU_DEPLOY_PASSWORD =
            [Runtime.InteropServices.Marshal]::PtrToStringBSTR($passwordPointer)
    }
    finally {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($passwordPointer)
    }
}

$wslEnvEntries = @()
if ($previousWslEnv) {
    $wslEnvEntries = @($previousWslEnv -split ":")
}
if ($wslEnvEntries -notcontains "AR_RAPHU_DEPLOY_PASSWORD") {
    $wslEnvEntries += "AR_RAPHU_DEPLOY_PASSWORD"
}
$env:WSLENV = $wslEnvEntries -join ":"

try {
    do {
        if (-not $Once) {
            Clear-Host
        }
        Write-Host (
            "AutoDL {0} status at {1}" -f
            $Task, (Get-Date -Format "yyyy-MM-dd HH:mm:ss")
        )
        Write-Host (
            "Refreshing every {0}s; press Ctrl+C to stop.`n" -f
            $IntervalSeconds
        )

        & wsl.exe `
            -d $WslDistribution `
            --cd $WslProjectPath `
            python3 tools/_autodl_remote.py `
            --host $RemoteHost `
            --port $RemotePort `
            --user $RemoteUser `
            --expected-host-key $expectedHostKey `
            exec $remoteCommand

        if ($LASTEXITCODE -ne 0) {
            throw "Remote status command failed with exit code $LASTEXITCODE."
        }
        if (-not $Once) {
            Start-Sleep -Seconds $IntervalSeconds
        }
    } while (-not $Once)
}
finally {
    if ($null -eq $previousWslEnv) {
        Remove-Item Env:WSLENV -ErrorAction SilentlyContinue
    }
    else {
        $env:WSLENV = $previousWslEnv
    }
    if (-not $passwordWasAlreadySet) {
        Remove-Item Env:AR_RAPHU_DEPLOY_PASSWORD -ErrorAction SilentlyContinue
    }
}
