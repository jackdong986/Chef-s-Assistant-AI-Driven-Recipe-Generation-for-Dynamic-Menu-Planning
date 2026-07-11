$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $venvPython)) {
    throw "The virtual environment is missing. Create .venv and install requirements-gradio.txt first."
}

if (-not $env:RECIPE_DATASET_PATH) {
    $env:RECIPE_DATASET_PATH = "C:\Users\Jack\Downloads\RAW_recipes_with_amount.csv"
}

& $venvPython (Join-Path $projectRoot "gradio_app.py") @args
