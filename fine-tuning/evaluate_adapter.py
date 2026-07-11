"""Evaluate a LoRA adapter at a configurable strength before deployment."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_MODEL = Path(
    r"C:\Users\Jack\Downloads\aistackphison\aistackphison\Llama-3.2-3B-Instruct"
)
DEFAULT_ADAPTER = PROJECT_ROOT / "models" / "chef-llama-3.2-3b-lora"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--scale", type=float, default=0.25)
    parser.add_argument(
        "--prompt",
        default=(
            "Create a Malaysian-style spicy chicken and rice dinner for 4. "
            "It must be halal and have no peanuts. Do not change the cuisine."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to("cuda")
    model = PeftModel.from_pretrained(model, args.adapter)
    for module in model.modules():
        scaling = getattr(module, "scaling", None)
        if isinstance(scaling, dict) and "default" in scaling:
            scaling["default"] *= args.scale
    model.eval()

    messages = [
        {
            "role": "system",
            "content": (
                "You are Chef's Assistant. The requested cuisine and dietary requirements "
                "are mandatory. Return only valid JSON with name, description, minutes, "
                "ingredients, and steps."
            ),
        },
        {"role": "user", "content": args.prompt},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=600,
            do_sample=True,
            temperature=0.2,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )
    print(
        tokenizer.decode(
            output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
    )


if __name__ == "__main__":
    main()
