# Single-model fine-tuning

The active training pipeline is `fine_tune_single_model.py`. It LoRA fine-tunes the
local Llama 3.2 3B Instruct checkpoint on a stratified sample of the recipe CSV and
saves both an adapter and a merged model under the Git-ignored `models/` directory.

`evaluate_adapter.py` tests different adapter strengths before deployment. The Gradio
application uses a 0.5 adapter scale for structured recipe drafts and the same base
model with reduced or disabled adapter influence for constraint auditing.

The training prompts preserve cuisine, style, and dietary labels from the dataset.
Generated recipes are evaluated again at runtime so mandatory constraints remain
explicit in the final result.
