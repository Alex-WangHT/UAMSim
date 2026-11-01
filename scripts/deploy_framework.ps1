param(
    [string]$Container = "ros2-learning",
    [string]$Destination = '/opt/uamsim/project'
)
$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$Running = & docker inspect --format '{{.State.Running}}' $Container
if ($LASTEXITCODE -ne 0) { throw "Cannot inspect container $Container" }
if ($Running -ne "true") {
    & docker start $Container
    if ($LASTEXITCODE -ne 0) { throw "Cannot start container $Container" }
}
& docker exec $Container mkdir -p $Destination
if ($LASTEXITCODE -ne 0) { throw "Cannot create deployment directory" }
foreach ($Item in @('UAMSim', 'examples', 'tests', 'scripts', 'docs', 'pyproject.toml', 'README.md')) {
    & docker cp (Join-Path $RepoRoot $Item) "${Container}:$Destination/"
    if ($LASTEXITCODE -ne 0) { throw "Cannot deploy $Item" }
}
Write-Output "Validation framework and tests staged in ${Container}:$Destination"
