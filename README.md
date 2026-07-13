# Chef's Assistant: AI-Driven Recipe Generation and Menu Planning

Chef's Assistant is a private, local kitchen workspace for chefs, home cooks, and
food teams. It combines a complete indexed recipe catalog with one locally hosted
Llama 3.2 3B Instruct model for recipe discovery, custom recipe creation, and
multi-day menu planning.

## Chef-facing features

- Search the complete recipe collection with natural-language queries, dietary and
  allergen filters, preparation-time limits, and Llama relevance ranking.
- Discover random Chinese, Western, or all-category recipes.
- Browse the full catalog alphabetically with pagination.
- Generate structured recipes with servings, quantities, cuisine requirements,
  deterministic quality scoring, and conditional AI correction.
- Plan menus by days, service periods, people, cuisine, dietary needs, ingredients
  already available, and pantry staples.
- Avoid repeated dishes or intentionally reuse leftovers with storage and reheating
  reminders.
- Download generated recipes as Markdown and menu plans as kitchen CSV or printable
  Markdown files.
- Cancel long AI operations directly from the responsive Gradio interface.

## How it works

The application uses one AI model: Llama 3.2 3B Instruct with the fine-tuned Chef's
Assistant LoRA adapter. SQLite full-text search is used for fast, deterministic
catalog access; it is a database index, not an additional AI model.

```text
Recipe CSV -> SQLite full-text candidates -> Llama ranking/selection
Chef brief -> Llama recipe draft -> deterministic checks -> conditional Llama audit
```

Deterministic checks cover JSON structure, servings, ingredient quantities, cuisine
identity, dietary/allergen violations, duplicates, and ingredient-step consistency.
The second model pass runs only when a check finds a problem.

## Current local configuration

- Dataset: `C:\Users\Jack\Downloads\RAW_recipes_with_amount.csv`
- Base model: `C:\Users\Jack\Downloads\aistackphison\aistackphison\Llama-3.2-3B-Instruct`
- LoRA adapter: `models\chef-llama-3.2-3b-lora`
- Recipe index: `.cache\recipes.sqlite3`
- Local server: `http://127.0.0.1:7860`

All paths can be changed with the variables in `.env.example`. The dataset, index,
model outputs, virtual environment, exports, caches, and logs are ignored by Git.

## Setup on Windows

From PowerShell in the repository directory:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-torch-cuda.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-gradio.txt
```

For development and automated tests:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

## Build the complete recipe index

The application builds the index automatically on the first dataset operation. To
build or refresh it manually:

```powershell
.\.venv\Scripts\python.exe .\scripts\build_recipe_index.py
```

Force a rebuild after changing index logic:

```powershell
.\.venv\Scripts\python.exe .\scripts\build_recipe_index.py --force
```

The index automatically rebuilds when the CSV path, size, or modification time
changes.

## Run the server

```powershell
.\run_gradio.ps1
```

Open `http://127.0.0.1:7860`. See [`SERVER_GUIDE.md`](SERVER_GUIDE.md) for start,
stop, restart, port recovery, and temporary public-link instructions.

## Tests and model evaluation

Run the deterministic test suite:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

GitHub Actions runs these tests without downloading the private dataset or model.

Run one fixed model benchmark case:

```powershell
.\.venv\Scripts\python.exe .\evaluation\run_benchmark.py --limit 1
```

Run the full benchmark by removing `--limit`. Reports are written to the Git-ignored
`.cache\evaluation\latest.json` and measure pass rate, quality score, audit use, and
latency.

## Fine-tuning

The training pipeline uses stratified sampling so less common cuisine and dietary
labels are represented:

```powershell
.\.venv\Scripts\python.exe .\fine-tuning\fine_tune_single_model.py `
  --dataset "C:\Users\Jack\Downloads\RAW_recipes_with_amount.csv" `
  --base-model "C:\Users\Jack\Downloads\aistackphison\aistackphison\Llama-3.2-3B-Instruct" `
  --max-rows 10000 `
  --epochs 2
```

Training output is saved under the Git-ignored `models\` directory. Use
`fine-tuning\evaluate_adapter.py` to compare adapter strengths.

## Project structure

```text
gradio_app.py                         Gradio UI and application workflows
recipe_store.py                       Complete SQLite/FTS recipe catalog
recipe_quality.py                     Deterministic recipe validation
scripts/build_recipe_index.py         Repeatable catalog index builder
evaluation/                           Fixed model benchmark cases and runner
tests/                                Fast deterministic unit tests
fine-tuning/                           LoRA training and adapter evaluation
run_gradio.ps1                        Windows server launcher
SERVER_GUIDE.md                       Start, stop, and restart reference
```

## Data and safety limitations

- The CSV's `amount` field is not used as an ingredient quantity because it stores
  one scalar per recipe instead of one quantity per ingredient.
- Budget is treated as a planning target because the dataset contains no verified
  ingredient prices. The application does not invent cost estimates.
- Dataset recipes and AI-generated recipes require professional review. Verify
  allergens, halal certification, storage, cooking temperatures, nutrition, and
  local food-safety requirements before service.

## Contributor

Jack Dong
