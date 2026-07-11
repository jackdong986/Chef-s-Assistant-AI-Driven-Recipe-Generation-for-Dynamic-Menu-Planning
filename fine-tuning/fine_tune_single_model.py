"""LoRA fine-tune one Llama model for structured recipe generation."""

from __future__ import annotations

import argparse
import ast
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = Path.home() / "Downloads" / "RAW_recipes_with_amount.csv"
DEFAULT_BASE_MODEL = Path(
    r"C:\Users\Jack\Downloads\aistackphison\aistackphison\Llama-3.2-3B-Instruct"
)
DEFAULT_ADAPTER_OUTPUT = PROJECT_ROOT / "models" / "chef-llama-3.2-3b-lora"
DEFAULT_MERGED_OUTPUT = PROJECT_ROOT / "models" / "chef-llama-3.2-3b"

os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface"))

import pandas as pd
import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

SYSTEM_MESSAGE = (
    "You are Chef's Assistant. Create realistic, internally consistent recipes. "
    "The requested cuisine, dish style, and dietary requirements are mandatory. "
    "Respect restrictions, use sensible ingredients, and provide safe, "
    "ordered cooking instructions. Return only valid JSON with the keys name, description, "
    "minutes, ingredients, and steps. Ingredients and steps must be JSON arrays of strings."
)


def parse_list(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple)):
            return [str(item).strip() for item in parsed if str(item).strip()]
    except (SyntaxError, ValueError):
        pass
    return [part.strip(" '[]\"") for part in text.split(",") if part.strip(" '[]\"")]


def load_training_frame(dataset_path: Path, max_rows: int) -> pd.DataFrame:
    columns = ["name", "description", "tags", "minutes", "ingredients", "steps"]
    frame = pd.read_csv(
        dataset_path,
        usecols=columns,
        encoding="ISO-8859-1",
        low_memory=False,
    )
    frame = frame.dropna(subset=["name", "ingredients", "steps"]).copy()
    frame["ingredients_list"] = frame["ingredients"].apply(parse_list)
    frame["steps_list"] = frame["steps"].apply(parse_list)
    frame["tags_list"] = frame["tags"].apply(parse_list)
    frame = frame[
        frame["ingredients_list"].map(len).between(2, 25)
        & frame["steps_list"].map(len).between(2, 25)
    ]
    frame = frame.drop_duplicates(subset=["name", "description"])
    sample_size = min(max_rows, len(frame))
    # Guarantee representation for less common cuisines instead of allowing popular
    # Western tags to dominate a purely random sample.
    focus_terms = [
        "african", "caribbean", "chinese", "filipino", "french", "greek",
        "indian", "indonesian", "italian", "japanese", "korean", "malaysian",
        "mediterranean", "mexican", "middle-eastern", "spanish", "thai", "vietnamese",
        "gluten-free", "vegan", "vegetarian",
    ]
    per_term = max(20, sample_size // (len(focus_terms) * 3))
    selected_indices: set[int] = set()
    lowered_tags = frame["tags"].fillna("").str.lower()
    for term in focus_terms:
        matching = frame[lowered_tags.str.contains(term, regex=False)]
        if not matching.empty:
            chosen = matching.sample(
                n=min(per_term, len(matching)), random_state=42
            ).index
            selected_indices.update(int(index) for index in chosen)

    selected = frame.loc[sorted(selected_indices)]
    remaining_count = max(0, sample_size - len(selected))
    remaining = frame.drop(index=selected.index).sample(
        n=min(remaining_count, len(frame) - len(selected)), random_state=42
    )
    return pd.concat([selected, remaining]).sample(frac=1, random_state=42).reset_index(drop=True)


class RecipeDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, tokenizer: Any, max_length: int) -> None:
        self.frame = frame.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        row = self.frame.iloc[index]
        description = str(row.get("description") or "").strip()[:500]
        tags = ", ".join(row["tags_list"][:10])
        name = str(row["name"]).strip()
        ingredients_hint = ", ".join(row["ingredients_list"][:8])
        templates = [
            f"Create a recipe matching this request. Dish or idea: {name}. "
            f"Description: {description or 'not provided'}. Mandatory style and dietary tags: {tags or 'none' }.",
            f"Develop a recipe called {name}. It must respect these cuisine, style, and dietary requirements: "
            f"{tags or 'none'}. The intended dish is: {description or name}.",
            f"Create a coherent recipe from this idea: {description or name}. Suggested dish style: {name}. "
            f"Useful core ingredients are {ingredients_hint}. Requirements: {tags or 'none'}.",
        ]
        user_message = templates[index % len(templates)]
        target = json.dumps(
            {
                "name": str(row["name"]).strip(),
                "description": description,
                "minutes": int(row["minutes"]) if pd.notna(row["minutes"]) else None,
                "ingredients": row["ingredients_list"][:20],
                "steps": row["steps_list"][:20],
            },
            ensure_ascii=False,
        )

        prompt_messages = [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": user_message},
        ]
        full_messages = prompt_messages + [{"role": "assistant", "content": target}]
        prompt_ids = self.tokenizer.apply_chat_template(
            prompt_messages, tokenize=True, add_generation_prompt=True
        )
        input_ids = self.tokenizer.apply_chat_template(
            full_messages, tokenize=True, add_generation_prompt=False
        )[: self.max_length]
        prompt_length = min(len(prompt_ids), len(input_ids))
        labels = [-100] * prompt_length + input_ids[prompt_length:]
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
        }


@dataclass
class CausalRecipeCollator:
    tokenizer: Any

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        model_features = [
            {"input_ids": item["input_ids"], "attention_mask": item["attention_mask"]}
            for item in features
        ]
        batch = self.tokenizer.pad(model_features, padding=True, return_tensors="pt")
        labels = torch.full(batch["input_ids"].shape, -100, dtype=torch.long)
        for row_index, item in enumerate(features):
            item_labels = torch.tensor(item["labels"], dtype=torch.long)
            labels[row_index, : len(item_labels)] = item_labels
        batch["labels"] = labels
        return batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter-output", type=Path, default=DEFAULT_ADAPTER_OUTPUT)
    parser.add_argument("--merged-output", type=Path, default=DEFAULT_MERGED_OUTPUT)
    parser.add_argument("--max-rows", type=int, default=10000)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument(
        "--skip-merge",
        action="store_true",
        help="Save only the LoRA adapter; useful for a short benchmark run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this fine-tuning configuration.")
    if not args.dataset.is_file():
        raise FileNotFoundError(f"Dataset not found: {args.dataset}")
    if not (args.base_model / "config.json").is_file():
        raise FileNotFoundError(f"Transformers model not found: {args.base_model}")

    frame = load_training_frame(args.dataset, args.max_rows)
    if len(frame) < 20:
        raise ValueError("At least 20 valid recipe records are required.")
    validation_size = max(1, int(len(frame) * 0.05))
    validation_frame = frame.iloc[:validation_size]
    training_frame = frame.iloc[validation_size:]

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.enable_input_require_grads()

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_dataset = RecipeDataset(training_frame, tokenizer, args.max_length)
    validation_dataset = RecipeDataset(validation_frame, tokenizer, args.max_length)
    use_bf16 = dtype == torch.bfloat16
    training_arguments = TrainingArguments(
        output_dir=str(args.adapter_output / "checkpoints"),
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=args.learning_rate,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        bf16=use_bf16,
        fp16=not use_bf16,
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=CausalRecipeCollator(tokenizer),
    )
    trainer.train()

    args.adapter_output.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(args.adapter_output, safe_serialization=True)
    tokenizer.save_pretrained(args.adapter_output)

    if args.skip_merge:
        print(f"LoRA adapter saved to {args.adapter_output}")
        return

    merged_model = trainer.model.merge_and_unload()
    merged_model.config.use_cache = True
    args.merged_output.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(
        args.merged_output,
        safe_serialization=True,
        max_shard_size="4GB",
    )
    tokenizer.save_pretrained(args.merged_output)
    (args.merged_output / "training_summary.json").write_text(
        json.dumps(
            {
                "base_model": str(args.base_model),
                "dataset": str(args.dataset),
                "training_examples": len(training_frame),
                "validation_examples": len(validation_frame),
                "epochs": args.epochs,
                "max_length": args.max_length,
                "learning_rate": args.learning_rate,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Merged model saved to {args.merged_output}")


if __name__ == "__main__":
    main()
