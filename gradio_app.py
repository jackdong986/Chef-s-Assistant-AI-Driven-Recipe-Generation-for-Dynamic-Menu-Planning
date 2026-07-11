"""Temporary Gradio server for recipe search, discovery, and generation."""

from __future__ import annotations

import argparse
import ast
import html
import json
import os
import random
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_PATH = Path.home() / "Downloads" / "RAW_recipes_with_amount.csv"
DEFAULT_BASE_MODEL_PATH = Path(
    r"C:\Users\Jack\Downloads\aistackphison\aistackphison\Llama-3.2-3B-Instruct"
)
DEFAULT_FINETUNED_MODEL_PATH = PROJECT_ROOT / "models" / "chef-llama-3.2-3b"
DEFAULT_ADAPTER_PATH = PROJECT_ROOT / "models" / "chef-llama-3.2-3b-lora"

# Keep downloaded Hugging Face models with this checkout. The directory is ignored by Git.
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface"))

import gradio as gr
import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

DATASET_PATH = Path(os.getenv("RECIPE_DATASET_PATH", str(DEFAULT_DATASET_PATH))).expanduser()
MAX_ROWS = max(1000, int(os.getenv("RECIPE_MAX_ROWS", "100000")))
GENERATION_MODEL = os.getenv("RECIPE_GENERATION_MODEL", "").strip()
BASE_MODEL = os.getenv("RECIPE_BASE_MODEL", str(DEFAULT_BASE_MODEL_PATH))
LORA_ADAPTER = os.getenv("RECIPE_LORA_ADAPTER", str(DEFAULT_ADAPTER_PATH))
LORA_SCALE = float(os.getenv("RECIPE_LORA_SCALE", "0.5"))

CHINESE_KEYWORDS = (
    "baozi",
    "char siu",
    "chow mein",
    "dim sum",
    "dumpling",
    "fried rice",
    "kung pao",
    "mapo tofu",
    "peking duck",
    "szechuan",
    "wonton",
)
WESTERN_KEYWORDS = (
    "burger",
    "caesar salad",
    "fish and chips",
    "lasagna",
    "pasta",
    "pizza",
    "roast",
    "sandwich",
    "steak",
)


def _safe_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def _parse_list(value: Any) -> list[str]:
    text = _safe_text(value)
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple)):
            return [_safe_text(item) for item in parsed if _safe_text(item)]
    except (SyntaxError, ValueError):
        pass
    return [part.strip(" '[]\"") for part in text.split(",") if part.strip(" '[]\"")]


@lru_cache(maxsize=1)
def load_recipes() -> pd.DataFrame:
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(
            f"Dataset not found at {DATASET_PATH}. Set RECIPE_DATASET_PATH to the CSV file."
        )

    columns = ["name", "description", "ingredients", "steps", "tags", "minutes"]
    frame = pd.read_csv(
        DATASET_PATH,
        usecols=columns,
        encoding="ISO-8859-1",
        low_memory=False,
    )
    frame = frame.dropna(subset=["name"]).copy()
    for column in ("name", "description", "ingredients", "steps", "tags"):
        frame[column] = frame[column].fillna("").astype(str)
    frame = frame.drop_duplicates(subset=["name", "description"])
    if len(frame) > MAX_ROWS:
        frame = frame.sample(n=MAX_ROWS, random_state=42)

    frame["search_text"] = (
        frame["name"]
        + ". "
        + frame["description"]
        + ". Ingredients: "
        + frame["ingredients"]
    ).str.slice(0, 1200)
    return frame.reset_index(drop=True)


@lru_cache(maxsize=1)
def load_generator() -> tuple[Any, Any]:
    model_source = GENERATION_MODEL or BASE_MODEL
    tokenizer = AutoTokenizer.from_pretrained(model_source)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        device = torch.device("cuda")
    else:
        dtype = torch.float32
        device = torch.device("cpu")

    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    adapter_path = Path(LORA_ADAPTER)
    if not GENERATION_MODEL and (adapter_path / "adapter_config.json").is_file():
        model = PeftModel.from_pretrained(model, adapter_path)
        for module in model.modules():
            scaling = getattr(module, "scaling", None)
            if isinstance(scaling, dict) and "default" in scaling:
                scaling["default"] *= LORA_SCALE
    model.eval()
    return tokenizer, model


@torch.inference_mode()
def _generate_with_model(
    system_message: str,
    user_message: str,
    *,
    max_new_tokens: int,
    temperature: float = 0.0,
    adapter_multiplier: float = 1.0,
) -> str:
    tokenizer, model = load_generator()
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    generation_options: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "repetition_penalty": 1.08,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if temperature > 0:
        generation_options.update(
            {"do_sample": True, "temperature": temperature, "top_p": 0.9}
        )
    else:
        generation_options.update(
            {"do_sample": False, "temperature": None, "top_p": None}
        )

    adjusted_layers: list[tuple[dict[str, float], float]] = []
    if adapter_multiplier != 1.0:
        for module in model.modules():
            scaling = getattr(module, "scaling", None)
            if isinstance(scaling, dict) and "default" in scaling:
                original_scale = float(scaling["default"])
                adjusted_layers.append((scaling, original_scale))
                scaling["default"] = original_scale * adapter_multiplier
    try:
        outputs = model.generate(**inputs, **generation_options)
    finally:
        for scaling, original_scale in adjusted_layers:
            scaling["default"] = original_scale
    generated_tokens = outputs[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


def _recipe_markdown(row: pd.Series, score: float | None = None) -> str:
    name = html.escape(_safe_text(row.get("name")) or "Untitled recipe")
    description = html.escape(_safe_text(row.get("description")) or "No description available.")
    ingredients = _parse_list(row.get("ingredients"))
    steps = _parse_list(row.get("steps"))
    minutes = _safe_text(row.get("minutes"))

    lines = [f"### {name}"]
    if score is not None:
        lines.append(f"Dataset match score: **{score:.0f}**")
    if minutes:
        lines.append(f"Preparation time: **{html.escape(minutes)} minutes**")
    lines.extend(["", description, "", "**Ingredients**"])
    lines.extend(f"- {html.escape(item)}" for item in ingredients[:20])
    lines.extend(["", "**Steps**"])
    lines.extend(f"{index}. {html.escape(step)}" for index, step in enumerate(steps[:20], 1))
    return "\n".join(lines)


def _query_tokens(query: str) -> list[str]:
    ignored = {"a", "an", "and", "for", "in", "of", "the", "to", "with"}
    return [
        token
        for token in re.findall(r"[a-z0-9]+", query.lower())
        if len(token) > 1 and token not in ignored
    ]


def _ingredient_contains_core(value: Any, token: str) -> bool:
    exclusions = {
        "chicken": {"bouillon", "broth", "flavor", "seasoning", "soup", "stock"},
        "beef": {"bouillon", "broth", "flavor", "seasoning", "soup", "stock"},
        "rice": {"flour", "noodle", "paper", "vermicelli", "wrapper"},
    }
    for ingredient in _parse_list(value):
        lowered = ingredient.lower()
        if token in lowered and not any(
            excluded in lowered for excluded in exclusions.get(token, set())
        ):
            return True
    return False


def _rank_recipes(
    query: str,
    count: int,
    source_frame: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    frame = source_frame.copy() if source_frame is not None else load_recipes()
    normalized_query = query.lower().strip()
    tokens = _query_tokens(normalized_query)
    if not tokens:
        return frame.iloc[0:0], np.array([], dtype=float)

    names = frame["name"].str.lower()
    descriptions = frame["description"].str.lower()
    ingredients = frame["ingredients"].str.lower()
    tags = frame["tags"].str.lower()
    scores = np.zeros(len(frame), dtype=float)

    scores += names.str.contains(normalized_query, regex=False).to_numpy(dtype=float) * 10
    scores += descriptions.str.contains(normalized_query, regex=False).to_numpy(dtype=float) * 4
    for token in tokens:
        scores += names.str.contains(token, regex=False).to_numpy(dtype=float) * 4
        scores += ingredients.str.contains(token, regex=False).to_numpy(dtype=float) * 2
        scores += tags.str.contains(token, regex=False).to_numpy(dtype=float)
        scores += descriptions.str.contains(token, regex=False).to_numpy(dtype=float)

    matching = np.flatnonzero(scores > 0)
    if not len(matching):
        return frame.iloc[0:0], np.array([], dtype=float)
    ordered = matching[np.argsort(scores[matching])[::-1]][:count]
    return frame.iloc[ordered], scores[ordered]


def ai_search(query: str, result_count: int) -> str:
    query = _safe_text(query)
    if not query:
        return "Enter a dish, ingredient, cuisine, or description to search."

    try:
        count = min(max(int(result_count), 1), 10)
        requested_cuisines = _requested_cuisines(query)
        source_frame = None
        if requested_cuisines:
            frame = load_recipes()
            cuisine_mask = pd.Series(True, index=frame.index)
            for cuisine in requested_cuisines:
                cuisine_mask &= frame["search_text"].str.contains(
                    cuisine, case=False, regex=False
                )
            source_frame = frame[cuisine_mask]
        candidates, lexical_scores = _rank_recipes(
            query, max(20, count), source_frame=source_frame
        )
        if candidates.empty:
            return "No matching recipe was found in the loaded dataset sample."

        candidates = candidates.reset_index(drop=True)
        candidate_lines = []
        for candidate_id, row in candidates.iterrows():
            candidate_lines.append(
                f"ID {candidate_id}: {_safe_text(row['name'])}. "
                f"Description: {_safe_text(row['description'])[:220]}. "
                f"Ingredients: {', '.join(_parse_list(row['ingredients'])[:8])}."
            )

        response = _generate_with_model(
            "You rank recipe search results. Judge intent, ingredients, cuisine, and dish type. "
            "Return only valid JSON in the form "
            '{"ranking":[{"id":0,"score":95}]}. Scores are integers from 0 to 100.',
            f"Search request: {query}\n\nCandidates:\n" + "\n".join(candidate_lines),
            max_new_tokens=300,
        )

        ranked: list[tuple[int, float]] = []
        try:
            parsed = json.loads(response[response.find("{") : response.rfind("}") + 1])
            seen: set[int] = set()
            for item in parsed.get("ranking", []):
                candidate_id = int(item["id"])
                if 0 <= candidate_id < len(candidates) and candidate_id not in seen:
                    ranked.append((candidate_id, min(max(float(item["score"]), 0), 100)))
                    seen.add(candidate_id)
        except (ValueError, TypeError, KeyError, json.JSONDecodeError):
            ranked = []

        used = {candidate_id for candidate_id, _ in ranked}
        for candidate_id, lexical_score in enumerate(lexical_scores):
            if candidate_id not in used:
                fallback_score = min(99.0, 40.0 + float(lexical_score) * 3.0)
                ranked.append((candidate_id, fallback_score))
        ranked = sorted(ranked, key=lambda item: item[1], reverse=True)[:count]

        return "\n\n---\n\n".join(
            _recipe_markdown(candidates.iloc[candidate_id], score)
            for candidate_id, score in ranked
        )
    except Exception as exc:  # Gradio should show a helpful message instead of a traceback.
        return f"### Search unavailable\n\n{html.escape(str(exc))}"


def random_recipe(category: str) -> str:
    try:
        frame = load_recipes()
        if category != "All":
            keywords = CHINESE_KEYWORDS if category == "Chinese" else WESTERN_KEYWORDS
            candidates = frame[
                frame["search_text"].str.lower().apply(
                    lambda text: any(keyword in text for keyword in keywords)
                )
            ]
        else:
            candidates = frame

        if candidates.empty:
            return f"No {category.lower()} recipe was found in the loaded sample."
        return _recipe_markdown(candidates.iloc[random.randrange(len(candidates))])
    except Exception as exc:
        return f"### Random recipe unavailable\n\n{html.escape(str(exc))}"


def _blocked_dietary_terms(dietary_notes: str) -> set[str]:
    notes = dietary_notes.lower()
    blocked_terms: set[str] = set()
    if "halal" in notes:
        blocked_terms.update({"pork", "bacon", "ham", "lard", "wine", "beer", "brandy", "rum"})
    if "vegetarian" in notes:
        blocked_terms.update({"beef", "chicken", "duck", "fish", "lamb", "pork", "shrimp", "turkey"})
    if "vegan" in notes:
        blocked_terms.update(
            {"beef", "butter", "cheese", "chicken", "cream", "egg", "fish", "honey", "lamb", "milk", "pork", "shrimp"}
        )
    if "no peanut" in notes or "peanut allergy" in notes:
        blocked_terms.update({"peanut", "groundnut"})
    if "gluten-free" in notes or "gluten free" in notes:
        blocked_terms.update({"barley", "bread", "flour", "pasta", "rye", "wheat"})
    return blocked_terms


def _matches_dietary_notes(row: pd.Series, dietary_notes: str) -> bool:
    recipe_text = " ".join(
        _safe_text(row.get(column))
        for column in ("name", "description", "tags", "ingredients", "steps")
    ).lower()
    blocked_terms = _blocked_dietary_terms(dietary_notes)
    return not any(term in recipe_text for term in blocked_terms)


def _requested_cuisines(description: str) -> list[str]:
    cuisine_terms = [
        "african", "american", "arabian", "caribbean", "chinese", "filipino",
        "french", "greek", "indian", "indonesian", "italian", "japanese",
        "korean", "malaysian", "mediterranean", "mexican", "middle eastern",
        "spanish", "thai", "vietnamese",
    ]
    lowered = description.lower()
    return [term for term in cuisine_terms if term in lowered]


def _parse_recipe_json(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            recipe = json.loads(text[start : end + 1])
            return recipe if isinstance(recipe, dict) else None
        except (json.JSONDecodeError, TypeError):
            return None
    return None


def _needs_recipe_correction(
    recipe: dict[str, Any] | None,
    cuisines: list[str],
    dietary_notes: str,
) -> bool:
    if not recipe:
        return True
    if not isinstance(recipe.get("ingredients"), list) or not isinstance(recipe.get("steps"), list):
        return True
    name = _safe_text(recipe.get("name"))
    identity = f"{name} {_safe_text(recipe.get('description'))}".lower()
    if cuisines and (
        any(cuisine not in identity for cuisine in cuisines) or len(name.split()) < 2
    ):
        return True
    complete_recipe = json.dumps(recipe, ensure_ascii=False).lower()
    return any(term in complete_recipe for term in _blocked_dietary_terms(dietary_notes))


def _deduplicate(items: Any) -> list[str]:
    if not isinstance(items, list):
        return []
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        cleaned = _safe_text(item)
        key = cleaned.lower()
        if cleaned and key not in seen:
            result.append(cleaned)
            seen.add(key)
    return result


def _normalize_generated_lists(recipe: dict[str, Any]) -> tuple[list[str], list[str]]:
    ingredients: list[str] = []
    ingredient_names: set[str] = set()
    for item in recipe.get("ingredients", []):
        if isinstance(item, dict):
            name = _safe_text(item.get("name"))
            quantity = _safe_text(item.get("quantity"))
            unit = _safe_text(item.get("unit"))
            if name:
                ingredient_names.add(name.lower())
            detail = unit if unit and name.lower() in unit.lower() else " ".join(
                part for part in (unit, name) if part
            )
            ingredients.append(" ".join(part for part in (quantity, detail) if part))
        else:
            ingredients.append(_safe_text(item))

    steps: list[str] = []
    declared_in_steps: set[str] = set()
    for item in recipe.get("steps", []):
        if isinstance(item, dict):
            steps.append(_safe_text(item.get("step") or item.get("instruction")))
            declared = item.get("ingredients", [])
            if isinstance(declared, list):
                declared_in_steps.update(_safe_text(value).lower() for value in declared)
        else:
            steps.append(_safe_text(item))

    for missing_name in sorted(declared_in_steps - ingredient_names):
        if missing_name:
            ingredients.append(f"as needed {missing_name}")
    return _deduplicate(ingredients)[:30], _deduplicate(steps)[:25]


def _render_generated_recipe(text: str) -> str:
    recipe = _parse_recipe_json(text)
    if recipe:
        try:
            ingredients, steps = _normalize_generated_lists(recipe)
            lines = [
                f"## {html.escape(_safe_text(recipe.get('name')) or 'Generated recipe')}",
                "",
                html.escape(_safe_text(recipe.get("description"))),
                "",
                f"Preparation time: **{html.escape(_safe_text(recipe.get('minutes')))} minutes**",
                "",
                "**Ingredients**",
            ]
            lines.extend(f"- {html.escape(_safe_text(item))}" for item in ingredients)
            lines.extend(["", "**Steps**"])
            lines.extend(
                f"{index}. {html.escape(_safe_text(step))}"
                for index, step in enumerate(steps, 1)
            )
            return "\n".join(lines)
        except TypeError:
            pass
    return f"## Generated recipe\n\n{html.escape(text)}"


def generate_recipe(description: str, servings: int, dietary_notes: str) -> str:
    description = _safe_text(description)
    if not description:
        return "Describe the recipe you want to generate."

    notes = _safe_text(dietary_notes) or "none"
    requested_cuisines = _requested_cuisines(description)
    candidate_references, _ = _rank_recipes(f"{description} {notes}", 20)
    compatible_references = candidate_references[
        candidate_references.apply(lambda row: _matches_dietary_notes(row, notes), axis=1)
    ]
    references = compatible_references.head(3)
    reference_text = []
    for _, row in references.iterrows():
        ingredients = ", ".join(_parse_list(row.get("ingredients"))[:8])
        steps = "; ".join(_parse_list(row.get("steps"))[:4])
        reference_text.append(
            f"Reference recipe: {_safe_text(row.get('name'))}. "
            f"Ingredients: {ingredients}. Steps: {steps}."
        )

    system_message = (
        "You are Chef's Assistant. Create realistic, internally consistent recipes. "
        "The requested cuisine, dish style, and dietary requirements are mandatory and must never "
        "be replaced by a different cuisine. Use sensible ingredient quantities and provide safe, "
        "ordered cooking instructions. Return only valid JSON with the keys name, description, "
        "minutes, ingredients, and steps. Ingredients and steps must be JSON arrays of strings."
    )
    cuisine_requirement = (
        "The JSON name must be a meaningful dish name containing "
        f"{', '.join(requested_cuisines)}, and the description must explicitly identify that cuisine. "
        if requested_cuisines
        else ""
    )
    user_message = (
        f"Create a recipe for {int(servings)} servings. Request: {description}. "
        f"Mandatory dietary requirements: {notes}. "
        f"{cuisine_requirement}"
        f"The request takes priority. Use these dataset recipes only as technical inspiration: "
        f"{' '.join(reference_text)}"
    )

    try:
        recipe = _generate_with_model(
            system_message,
            user_message,
            max_new_tokens=600,
            temperature=0.25,
        )
        if not recipe:
            return "The model returned an empty recipe. Try a more specific description."
        parsed_recipe = _parse_recipe_json(recipe)
        correction_reason = (
            "The draft failed a mandatory format, cuisine, or dietary check. "
            if _needs_recipe_correction(parsed_recipe, requested_cuisines, notes)
            else "Audit the draft for accuracy and internal consistency. "
        )
        correction_message = (
            f"{correction_reason}Original request: {description}. Servings: {int(servings)}. "
            f"Mandatory dietary requirements: {notes}. "
            f"Mandatory cuisines: {', '.join(requested_cuisines) or 'none specified'}. "
            "Correct the draft while preserving the request. Use a meaningful dish name. Every "
            "ingredient must have a practical quantity, every ingredient mentioned in the steps "
            "must appear in the ingredient list, and every main ingredient must be used in the "
            "steps. Do not switch between rice, noodles, pasta, or another starch. Remove duplicate "
            "and unused ingredients. Return only valid JSON with name, description, minutes, "
            f"ingredients, and steps. Draft: {recipe}"
        )
        recipe = _generate_with_model(
            system_message,
            correction_message,
            max_new_tokens=700,
            temperature=0.1,
            adapter_multiplier=0.0,
        )
        return (
            f"{_render_generated_recipe(recipe)}\n\n"
            "> AI-generated recipe: verify allergens, food safety, and cooking temperatures before use."
        )
    except Exception as exc:
        return f"### Generation unavailable\n\n{html.escape(str(exc))}"


def _category_mask(frame: pd.DataFrame, category: str) -> pd.Series:
    if category == "All":
        return pd.Series(True, index=frame.index)
    keywords = CHINESE_KEYWORDS if category == "Chinese" else WESTERN_KEYWORDS
    return frame["search_text"].str.lower().apply(
        lambda text: any(keyword in text for keyword in keywords)
    )


def browse_recipes(letter: str, category: str, page: int) -> tuple[str, str]:
    try:
        frame = load_recipes()
        mask = _category_mask(frame, category)
        if letter != "All":
            cleaned_names = frame["name"].str.lower().str.replace(
                r"^(?:\d+\s*|the\s+|in\s+)", "", regex=True
            )
            mask &= cleaned_names.str.startswith(letter.lower())
        filtered = frame[mask]
        if filtered.empty:
            return "No recipes match these filters.", "Page 0 of 0"

        page_size = 5
        total_pages = max(1, (len(filtered) + page_size - 1) // page_size)
        current_page = min(max(int(page or 1), 1), total_pages)
        start = (current_page - 1) * page_size
        page_frame = filtered.iloc[start : start + page_size]
        content = "\n\n---\n\n".join(
            _recipe_markdown(row) for _, row in page_frame.iterrows()
        )
        return content, f"Page {current_page:,} of {total_pages:,} · {len(filtered):,} recipes"
    except Exception as exc:
        return f"### Browse unavailable\n\n{html.escape(str(exc))}", ""


def generate_menu_plan(
    days: int,
    meals_per_day: int,
    servings: int,
    budget: str,
    dietary_notes: str,
    preferences: str,
) -> str:
    dietary = _safe_text(dietary_notes) or "none"
    preference_text = _safe_text(preferences) or "varied meals"
    requested_cuisines = _requested_cuisines(preference_text)
    required_meals = int(days) * int(meals_per_day)
    query = " ".join(
        value for value in (preference_text, dietary) if value
    )
    try:
        source_frame = None
        if requested_cuisines:
            frame = load_recipes()
            cuisine_mask = pd.Series(True, index=frame.index)
            for cuisine in requested_cuisines:
                cuisine_mask &= frame["search_text"].str.contains(
                    cuisine, case=False, regex=False
                )
            source_frame = frame[cuisine_mask]
        core_tokens = [
            token
            for token in _query_tokens(preference_text)
            if token not in requested_cuisines
        ]
        if core_tokens:
            core_source = source_frame if source_frame is not None else load_recipes()
            core_mask = pd.Series(True, index=core_source.index)
            for token in core_tokens:
                core_mask &= core_source["ingredients"].apply(
                    lambda value, required=token: _ingredient_contains_core(value, required)
                )
            core_matches = core_source[core_mask]
            if len(core_matches) >= required_meals:
                source_frame = core_matches
            else:
                return (
                    "Not enough dataset recipes contain all mandatory core ingredients "
                    f"({', '.join(core_tokens)}) while also satisfying the cuisine and dietary filters."
                )

        candidate_count = max(30, required_meals * 4)
        candidates, _ = _rank_recipes(
            query, candidate_count, source_frame=source_frame
        )
        candidates = candidates[
            candidates.apply(lambda row: _matches_dietary_notes(row, dietary), axis=1)
        ].reset_index(drop=True)
        if len(candidates) < required_meals:
            return (
                "Not enough dataset recipes satisfy every mandatory cuisine and dietary filter. "
                "Broaden the preferences or increase RECIPE_MAX_ROWS."
            )

        candidate_lines = []
        for candidate_id, row in candidates.iterrows():
            candidate_lines.append(
                f"ID {candidate_id}: {_safe_text(row['name'])}; "
                f"{_safe_text(row['minutes'])} minutes; ingredients: "
                f"{', '.join(_parse_list(row['ingredients'])[:10])}"
            )
        selection = _generate_with_model(
            "You select recipes for a practical menu. Balance variety with ingredient reuse and "
            "respect all stated filters. Return only valid JSON like "
            '{"selected_ids":[2,5,1]}. Do not return commentary or duplicate IDs.',
            f"Select exactly {required_meals} IDs for {int(servings)} people. "
            f"Budget preference: {_safe_text(budget) or 'not specified'}. "
            f"Requirements: {dietary}; {preference_text}. Candidates:\n"
            + "\n".join(candidate_lines),
            max_new_tokens=200,
            temperature=0.0,
        )
        selected_ids: list[int] = []
        try:
            parsed = json.loads(selection[selection.find("{") : selection.rfind("}") + 1])
            for value in parsed.get("selected_ids", []):
                candidate_id = int(value)
                if 0 <= candidate_id < len(candidates) and candidate_id not in selected_ids:
                    selected_ids.append(candidate_id)
        except (ValueError, TypeError, json.JSONDecodeError):
            selected_ids = []
        for candidate_id in range(len(candidates)):
            if len(selected_ids) >= required_meals:
                break
            if candidate_id not in selected_ids:
                selected_ids.append(candidate_id)

        selected = candidates.iloc[selected_ids[:required_meals]]
        lines = [
            "# Dynamic Menu Plan",
            "",
            f"For **{int(servings)} people** · Budget target: **{html.escape(_safe_text(budget) or 'not specified')}**",
            "",
        ]
        shopping_items: dict[str, str] = {}
        selected_rows = list(selected.iterrows())
        row_position = 0
        for day_number in range(1, int(days) + 1):
            lines.extend([f"## Day {day_number}", ""])
            for meal_number in range(1, int(meals_per_day) + 1):
                _, row = selected_rows[row_position]
                row_position += 1
                name = html.escape(_safe_text(row["name"]).title())
                minutes = html.escape(_safe_text(row.get("minutes")))
                ingredients = _parse_list(row.get("ingredients"))
                steps = _parse_list(row.get("steps"))
                lines.extend(
                    [
                        f"### Meal {meal_number} - {name}",
                        f"Preparation time: **{minutes} minutes**",
                        "",
                        "**Ingredients**",
                    ]
                )
                lines.extend(f"- {html.escape(item)}" for item in ingredients[:20])
                lines.extend(["", "**Steps**"])
                lines.extend(
                    f"{index}. {html.escape(step)}"
                    for index, step in enumerate(steps[:20], 1)
                )
                lines.append("")
                for item in ingredients:
                    key = item.lower().strip()
                    if key:
                        shopping_items.setdefault(key, item)

        lines.extend(["## Consolidated Shopping List", ""])
        lines.extend(
            f"- {html.escape(item)}"
            for item in sorted(shopping_items.values(), key=str.lower)
        )
        lines.extend(
            [
                "",
                "> Budget is a planning target only because the dataset contains no ingredient prices. ",
                "> Verify allergens, halal certification, food safety, and local prices before use.",
            ]
        )
        return "\n".join(lines)
    except Exception as exc:
        return f"### Menu planning unavailable\n\n{html.escape(str(exc))}"


def build_demo() -> gr.Blocks:
    model_status = (
        GENERATION_MODEL
        or f"{BASE_MODEL} + {LORA_ADAPTER} (LoRA scale {LORA_SCALE:g})"
    )
    dataset_status = (
        f"Dataset: `{DATASET_PATH}` ({MAX_ROWS:,} rows loaded at most)  \n"
        f"Single AI model: `{model_status}`  \n"
        "The same model reranks searches, generates recipes, and plans menus."
    )

    with gr.Blocks(title="Chef's Assistant") as demo:
        gr.Markdown("# Chef's Assistant\nTemporary local Gradio interface for the recipe project.")
        gr.Markdown(dataset_status)

        with gr.Tab("AI recipe search"):
            gr.Markdown(
                "The CSV selects candidates, then the same Llama model scores their relevance."
            )
            search_query = gr.Textbox(
                label="What recipe are you looking for?",
                placeholder="e.g. quick spicy chicken with rice",
            )
            result_count = gr.Slider(1, 10, value=5, step=1, label="Results")
            search_button = gr.Button("Search", variant="primary")
            search_output = gr.Markdown()
            search_button.click(
                ai_search,
                inputs=[search_query, result_count],
                outputs=search_output,
            )
            search_query.submit(
                ai_search,
                inputs=[search_query, result_count],
                outputs=search_output,
            )

        with gr.Tab("Random recipe"):
            category = gr.Dropdown(
                ["All", "Chinese", "Western"], value="All", label="Category"
            )
            random_button = gr.Button("Pick a recipe", variant="primary")
            random_output = gr.Markdown()
            random_button.click(random_recipe, inputs=category, outputs=random_output)

        with gr.Tab("Browse all recipes"):
            with gr.Row():
                browse_letter = gr.Dropdown(
                    ["All"] + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
                    value="All",
                    label="Starts with",
                )
                browse_category = gr.Dropdown(
                    ["All", "Chinese", "Western"], value="All", label="Category"
                )
                browse_page = gr.Number(value=1, precision=0, label="Page")
            browse_button = gr.Button("Load recipes", variant="primary")
            browse_status = gr.Markdown()
            browse_output = gr.Markdown()
            browse_button.click(
                browse_recipes,
                inputs=[browse_letter, browse_category, browse_page],
                outputs=[browse_output, browse_status],
            )

        with gr.Tab("Generate with AI"):
            generation_request = gr.Textbox(
                label="Describe your recipe",
                lines=4,
                placeholder="e.g. a simple Malaysian-inspired chicken dinner",
            )
            servings = gr.Slider(1, 12, value=4, step=1, label="Servings")
            dietary_notes = gr.Textbox(
                label="Dietary requirements",
                placeholder="e.g. no peanuts, low sodium",
            )
            generate_button = gr.Button("Generate recipe", variant="primary")
            generation_output = gr.Markdown()
            generate_button.click(
                generate_recipe,
                inputs=[generation_request, servings, dietary_notes],
                outputs=generation_output,
            )

        with gr.Tab("Dynamic menu planner"):
            with gr.Row():
                menu_days = gr.Slider(1, 7, value=3, step=1, label="Days")
                menu_meals = gr.Slider(1, 3, value=2, step=1, label="Meals per day")
                menu_servings = gr.Slider(1, 12, value=4, step=1, label="People")
            menu_budget = gr.Textbox(label="Budget", placeholder="e.g. RM150 total")
            menu_dietary = gr.Textbox(
                label="Dietary requirements", placeholder="e.g. halal, no peanuts"
            )
            menu_preferences = gr.Textbox(
                label="Preferences and available ingredients",
                placeholder="e.g. Malaysian and Chinese food; chicken, rice, vegetables",
            )
            menu_button = gr.Button("Create menu plan", variant="primary")
            menu_output = gr.Markdown()
            menu_button.click(
                generate_menu_plan,
                inputs=[
                    menu_days,
                    menu_meals,
                    menu_servings,
                    menu_budget,
                    menu_dietary,
                    menu_preferences,
                ],
                outputs=menu_output,
            )

    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--share",
        action="store_true",
        default=os.getenv("GRADIO_SHARE", "false").lower() == "true",
        help="Create a temporary public Gradio URL.",
    )
    parser.add_argument(
        "--server-name",
        default=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_demo().queue(default_concurrency_limit=1).launch(
        server_name=args.server_name,
        server_port=args.port,
        share=args.share,
        show_error=True,
    )
