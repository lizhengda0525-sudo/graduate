param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('Extract', 'Preprocess', 'Train')]
    [string]$Stage,
    [string]$ConfigPath = 'configs\hup116.yaml',
    [string]$PythonPath = 'E:\Anaconda\envs\qwen3tts-cpu\python.exe',
    [string]$ExtractionPythonPath = 'E:\Anaconda\envs\epilepsy_ekf_data\python.exe'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path

if (-not [System.IO.Path]::IsPathRooted($ConfigPath)) {
    $ConfigPath = Join-Path $projectRoot $ConfigPath
}
$configFile = (Resolve-Path -LiteralPath $ConfigPath).Path

function Invoke-HupCommand {
    param([string]$CommandName)

    if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
        throw "Python 程序不存在：$PythonPath"
    }
    Push-Location -LiteralPath (Join-Path $projectRoot 'src')
    $previousPythonEncoding = [Environment]::GetEnvironmentVariable('PYTHONIOENCODING', 'Process')
    $env:PYTHONIOENCODING = 'utf-8'
    try {
        & $PythonPath -m tc_cmlp.cli $CommandName --config $configFile
        if ($LASTEXITCODE -ne 0) {
            throw "HUP 命令执行失败，退出代码：$LASTEXITCODE"
        }
    }
    finally {
        Pop-Location
        if ($null -eq $previousPythonEncoding) {
            Remove-Item Env:PYTHONIOENCODING
        }
        else {
            $env:PYTHONIOENCODING = $previousPythonEncoding
        }
    }
}

switch ($Stage) {
    'Extract' {
        if (-not (Test-Path -LiteralPath $ExtractionPythonPath -PathType Leaf)) {
            throw "提取信号使用的 Python 程序不存在：$ExtractionPythonPath"
        }
        $extractor = Join-Path $projectRoot 'scripts\extract_hup_segment.py'
        & $ExtractionPythonPath $extractor --config $configFile
        if ($LASTEXITCODE -ne 0) {
            throw "信号提取失败，退出代码：$LASTEXITCODE"
        }
    }
    'Preprocess' {
        Invoke-HupCommand 'preprocess-hup'
    }
    'Train' {
        Invoke-HupCommand 'hup'
    }
}
