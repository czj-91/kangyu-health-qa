# Fetch GitHub repo stats (stars / watching / forks / open issues) via REST API
# and append a daily row to logs/stats_log.csv.
# NOTE: keep this file ASCII-only - Windows PowerShell 5.1 misparses BOM-less
# UTF-8 non-ASCII comments.
# Run daily via Task Scheduler (see command below):
#   schtasks /Create /SC DAILY /ST 09:00 /TN KangyuRepoStats /F ^
#     /TR "powershell -NoProfile -ExecutionPolicy Bypass -File \"C:\Users\15893\Desktop\AI_QA_Project\scripts\fetch_repo_stats.ps1\""

$ErrorActionPreference = 'Stop'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$repo   = 'czj-91/kangyu-health-qa'
$outCsv = Join-Path $PSScriptRoot '..\logs\stats_log.csv'   # logs/ is gitignored

New-Item -ItemType Directory -Force -Path (Split-Path $outCsv) | Out-Null

$headers = @{
    'User-Agent' = 'kangyu-stats-script'
    'Accept'     = 'application/vnd.github+json'
}

# Optional: personal access token raises rate limit and unlocks nothing extra here,
# unauthenticated 60 req/h is plenty for a daily job. Env var GITHUB_TOKEN honored.
if ($env:GITHUB_TOKEN) { $headers['Authorization'] = "Bearer $env:GITHUB_TOKEN" }

$uri = "https://api.github.com/repos/$repo"

try {
    $r = Invoke-RestMethod -Uri $uri -UseBasicParsing -TimeoutSec 30 -Headers $headers
}
catch {
    Write-Error "API request failed: $_"
    exit 1
}

$row = [pscustomobject]@{
    date     = (Get-Date).ToUniversalTime().ToString('yyyy-MM-dd')
    stars    = $r.stargazers_count
    watching = $r.subscribers_count
    forks    = $r.forks_count
    issues   = $r.open_issues_count
}

# Upsert by date: replace same-day row if present, else append
if (Test-Path $outCsv) {
    $all = @(Import-Csv $outCsv | Where-Object { $_.date -ne $row.date })
}
else {
    $all = @()
}
$all += $row
$all | Export-Csv -Path $outCsv -NoTypeInformation

Write-Output ("OK {0}  stars={1} watching={2} forks={3}" -f $row.date, $row.stars, $row.watching, $row.forks)
