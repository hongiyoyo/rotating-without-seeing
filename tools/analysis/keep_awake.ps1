# Keeps Windows from sleeping while the training (train_full.sh in WSL) is running.
# Does not change power settings; the request ends when this script exits.
Add-Type -Name Power -Namespace Win32 -MemberDefinition @'
[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);
'@
$ES_CONTINUOUS = [uint32]2147483648
$ES_SYSTEM_REQUIRED = [uint32]1
[Win32.Power]::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED) | Out-Null
while ($true) {
    $running = Get-CimInstance Win32_Process -Filter "Name='wsl.exe'" |
        Where-Object { $_.CommandLine -like '*train_full.sh*' }
    if (-not $running) { break }
    Start-Sleep -Seconds 60
}
[Win32.Power]::SetThreadExecutionState($ES_CONTINUOUS) | Out-Null
