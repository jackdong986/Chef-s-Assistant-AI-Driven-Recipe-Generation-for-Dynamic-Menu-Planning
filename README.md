# Chef's Assistant: AI-Driven Recipe Generation for Dynamic Menu Planning

Chef's Assistant is a local Gradio application for searching, browsing, generating,
and planning recipes. It keeps the original project objectives while replacing the
old GPT-2 plus SentenceTransformer setup with one Llama 3.2 3B Instruct model.

The recipe CSV supplies factual recipe records. The single Llama model reranks
search results, creates custom recipes, and selects recipes for menu plans. A LoRA
adapter fine-tuned on the supplied dataset improves recipe structure while a second
audit pass through the same loaded model enforces cuisine, dietary, ingredient, and
step consistency.

## Features

- AI recipe search with dataset candidate filtering and Llama relevance ranking.
- Random recipe discovery for all, Chinese, or Western recipes.
- Alphabetical and category-based recipe browsing with pagination.
- Custom recipe generation with servings and dietary requirements.
- Dynamic menu planning by days, meals, budget, dietary needs, cuisine, and
  available ingredients, including a consolidated shopping list.
- A temporary local Gradio web interface with lazy model loading.

## Current local configuration

This checkout is configured for the following local resources:

- Dataset: `C:\Users\Jack\Downloads\RAW_recipes_with_amount.csv`
- Base model: `C:\Users\Jack\Downloads\aistackphison\aistackphison\Llama-3.2-3B-Instruct`
- Fine-tuned LoRA adapter: `models\chef-llama-3.2-3b-lora`
- Default server: `http://127.0.0.1:7860`

These paths can be changed with the environment variables documented in
`.env.example`. The CSV, model files, virtual environment, caches, checkpoints, and
logs are excluded by `.gitignore` and should not be pushed to GitHub.

## Setup on Windows

From PowerShell in the repository directory:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-torch-cuda.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-gradio.txt
```

The current machine already has this `.venv` configured with a CUDA-enabled PyTorch
build. A compatible NVIDIA GPU is strongly recommended; CPU generation with a 3B
model will be slow.

## Run the Gradio server

For start, stop, restart, health-check, and port-recovery instructions, see
[`SERVER_GUIDE.md`](SERVER_GUIDE.md).

Start the local-only server:

```powershell
.\run_gradio.ps1
```

Then open `http://127.0.0.1:7860`. The application loads the CSV when a dataset
feature is first used and loads the Llama model when an AI feature is first used.

To create a temporary public Gradio URL, run:

```powershell
.\run_gradio.ps1 --share
```

Only use `--share` when public access is intended. Do not expose private data or an
unattended model server.

The legacy command remains available as a compatibility entry point:

```powershell
.\.venv\Scripts\python.exe .\backend\chef.py
```

## Fine-tuning the single model

The active training pipeline is `fine-tuning\fine_tune_single_model.py`. It selects
a stratified sample so less common cuisine and dietary labels are represented, then
LoRA fine-tunes the local Llama checkpoint.

Example:

```powershell
.\.venv\Scripts\python.exe .\fine-tuning\fine_tune_single_model.py `
  --dataset "C:\Users\Jack\Downloads\RAW_recipes_with_amount.csv" `
  --base-model "C:\Users\Jack\Downloads\aistackphison\aistackphison\Llama-3.2-3B-Instruct" `
  --max-rows 10000 `
  --epochs 2
```

Training output is saved under the Git-ignored `models\` directory. The application
currently uses the adapter at scale `0.5`, which can be changed with
`RECIPE_LORA_SCALE`.

The CSV's `amount` field is not used as an ingredient quantity because it contains
one scalar per recipe rather than an amount for each ingredient. Generated recipes
receive explicit ingredient quantities during the model audit pass.

## Project structure

```text
gradio_app.py                         Active Gradio application
run_gradio.ps1                        Windows launcher
backend/chef.py                       Compatibility launcher
fine-tuning/fine_tune_single_model.py Single-model LoRA training
fine-tuning/evaluate_adapter.py       Adapter-strength evaluation
ui/                                   Original HTML interface retained for reference
models/                               Local outputs; ignored by Git
```

## Accuracy and safety notes

- Search and menu planning ground results in the dataset instead of asking the
  language model to invent every record.
- Mandatory cuisine and dietary terms are checked before results are shown. If the
  sampled dataset does not contain enough exact matches, the menu planner reports
  that honestly rather than silently weakening the constraints.
- Generated recipes still require human review. Verify allergens, halal status,
  cooking temperatures, food safety, nutrition, and local prices before use.

## Possible next improvements

- Add a held-out evaluation set with cuisine accuracy, dietary violation rate,
  ingredient-step consistency, and human taste ratings.
- Add persistent user accounts, favorites, feedback, and saved menu plans.
- Index the full CSV in a lightweight database for faster and more complete search.
- Add ingredient substitution and nutrition data from verified sources.
- Containerize the application and add automated tests and GitHub Actions.
- Build a mobile-friendly production UI after the model workflow is stable.

## Contributor

Jack Dong
