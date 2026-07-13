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


CHEF_THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.orange,
    secondary_hue=gr.themes.colors.amber,
    neutral_hue=gr.themes.colors.stone,
    radius_size=gr.themes.sizes.radius_lg,
    text_size=gr.themes.sizes.text_md,
    font=("Segoe UI", "Inter", "ui-sans-serif", "system-ui", "sans-serif"),
    font_mono=("Cascadia Code", "Consolas", "ui-monospace", "monospace"),
)

APP_CSS = r"""
:root {
    --chef-ink: #20322a;
    --chef-muted: #68756e;
    --chef-forest: #173f32;
    --chef-forest-soft: #245541;
    --chef-orange: #e86f32;
    --chef-orange-dark: #c95422;
    --chef-cream: #f8f3e9;
    --chef-paper: rgba(255, 253, 248, 0.94);
    --chef-line: rgba(39, 61, 51, 0.13);
    --chef-shadow: 0 18px 50px rgba(32, 50, 42, 0.09);
}

body,
.gradio-container {
    background:
        radial-gradient(circle at 8% 4%, rgba(232, 111, 50, 0.10), transparent 29rem),
        radial-gradient(circle at 92% 22%, rgba(54, 119, 89, 0.10), transparent 32rem),
        var(--chef-cream) !important;
    color: var(--chef-ink) !important;
}

.gradio-container {
    --body-background-fill: #f8f3e9;
    --body-background-fill-dark: #f8f3e9;
    --body-text-color: #20322a;
    --body-text-color-dark: #20322a;
    --body-text-color-subdued: #68756e;
    --body-text-color-subdued-dark: #68756e;
    --background-fill-primary: #fffdf8;
    --background-fill-primary-dark: #fffdf8;
    --background-fill-secondary: #f8f3e9;
    --background-fill-secondary-dark: #f8f3e9;
    --block-background-fill: #fffdf8;
    --block-background-fill-dark: #fffdf8;
    --block-border-color: rgba(39, 61, 51, 0.13);
    --block-border-color-dark: rgba(39, 61, 51, 0.13);
    --block-info-text-color: #68756e;
    --block-info-text-color-dark: #68756e;
    --block-label-background-fill: transparent;
    --block-label-background-fill-dark: transparent;
    --block-label-text-color: #31473d;
    --block-label-text-color-dark: #31473d;
    --block-title-text-color: #20322a;
    --block-title-text-color-dark: #20322a;
    --border-color-primary: rgba(39, 61, 51, 0.16);
    --border-color-primary-dark: rgba(39, 61, 51, 0.16);
    --input-background-fill: #fffdf8;
    --input-background-fill-dark: #fffdf8;
    --input-border-color: rgba(39, 61, 51, 0.16);
    --input-border-color-dark: rgba(39, 61, 51, 0.16);
    --input-placeholder-color: #8b958f;
    --input-placeholder-color-dark: #8b958f;
    --shadow-drop: 0 1px 2px rgba(32, 50, 42, 0.05);
    --shadow-drop-lg: 0 18px 50px rgba(32, 50, 42, 0.09);
    max-width: 1280px !important;
    padding: 24px 24px 52px !important;
}

#chef-hero {
    position: relative;
    overflow: hidden;
    padding: 44px 46px 40px;
    border: 1px solid rgba(255, 255, 255, 0.09);
    border-radius: 30px;
    background: linear-gradient(135deg, #173f32 0%, #1c4a39 58%, #6b4b2e 140%);
    box-shadow: 0 24px 70px rgba(23, 63, 50, 0.22);
    color: #fffaf0;
}

#chef-hero::after {
    content: "";
    position: absolute;
    width: 330px;
    height: 330px;
    right: -120px;
    top: -160px;
    border: 68px solid rgba(245, 166, 97, 0.10);
    border-radius: 50%;
}

.hero-content {
    position: relative;
    z-index: 1;
    max-width: 780px;
}

.hero-eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 9px;
    margin-bottom: 16px;
    color: #ffd4ad;
    font-size: 0.76rem;
    font-weight: 750;
    letter-spacing: 0.14em;
    text-transform: uppercase;
}

.hero-eyebrow::before {
    content: "";
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #f7a65f;
    box-shadow: 0 0 0 5px rgba(247, 166, 95, 0.14);
}

.hero-title {
    margin: 0;
    max-width: 720px;
    color: #fffdf8;
    font-family: Georgia, "Times New Roman", serif;
    font-size: clamp(2.6rem, 6vw, 5rem);
    font-weight: 600;
    letter-spacing: -0.055em;
    line-height: 0.98;
}

.hero-copy {
    max-width: 660px;
    margin: 20px 0 0;
    color: rgba(255, 253, 248, 0.76);
    font-size: 1.05rem;
    line-height: 1.65;
}

.hero-stats {
    position: relative;
    z-index: 1;
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 30px;
}

.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 9px 13px;
    border: 1px solid rgba(255, 255, 255, 0.13);
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.08);
    color: rgba(255, 253, 248, 0.90);
    font-size: 0.82rem;
    backdrop-filter: blur(8px);
}

.status-pill strong {
    color: #fff;
    font-weight: 700;
}

#system-note {
    margin: 14px 2px 20px;
    padding: 11px 16px;
    border: 1px solid var(--chef-line);
    border-radius: 14px;
    background: rgba(255, 253, 248, 0.62);
    color: var(--chef-muted);
    font-size: 0.82rem;
}

#chef-workspace {
    margin-top: 6px;
}

#chef-workspace > .tab-nav {
    gap: 6px;
    margin-bottom: 18px;
    padding: 6px;
    border: 1px solid var(--chef-line);
    border-radius: 17px;
    background: rgba(255, 253, 248, 0.80);
    box-shadow: 0 8px 28px rgba(32, 50, 42, 0.06);
}

#chef-workspace > .tab-nav button {
    min-height: 44px;
    border: 0 !important;
    border-radius: 12px !important;
    color: #5d6c64;
    font-size: 0.88rem;
    font-weight: 650;
}

#chef-workspace > .tab-nav button.selected {
    background: var(--chef-forest) !important;
    color: #fffdf8 !important;
    box-shadow: 0 7px 18px rgba(23, 63, 50, 0.20);
}

.tool-heading {
    margin: 5px 0 18px;
    padding: 0 3px;
}

.tool-kicker {
    margin-bottom: 6px;
    color: var(--chef-orange-dark);
    font-size: 0.74rem;
    font-weight: 750;
    letter-spacing: 0.13em;
    text-transform: uppercase;
}

.tool-title {
    margin: 0;
    color: var(--chef-ink);
    font-family: Georgia, "Times New Roman", serif;
    font-size: clamp(1.75rem, 3vw, 2.45rem);
    font-weight: 600;
    letter-spacing: -0.035em;
}

.tool-description {
    max-width: 710px;
    margin: 7px 0 0;
    color: var(--chef-muted);
    line-height: 1.6;
}

.workspace-grid {
    align-items: stretch;
    gap: 17px;
}

.control-panel,
.output-panel {
    border: 1px solid var(--chef-line) !important;
    border-radius: 22px !important;
    background: var(--chef-paper) !important;
    box-shadow: var(--chef-shadow);
}

.control-panel {
    padding: 22px !important;
}

.output-panel {
    min-height: 410px;
    padding: 24px 26px !important;
}

.panel-label {
    margin: 0 0 15px;
    color: var(--chef-muted);
    font-size: 0.73rem;
    font-weight: 750;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

.field-note {
    margin: -4px 2px 12px;
    color: #7a857f;
    font-size: 0.78rem;
    line-height: 1.45;
}

.control-panel .form,
.control-panel .wrap,
.control-panel input,
.control-panel textarea {
    background: #fffdf8 !important;
    border-color: rgba(39, 61, 51, 0.15) !important;
    color: var(--chef-ink) !important;
}

.primary-action {
    min-height: 48px !important;
    margin-top: 8px !important;
    border: 0 !important;
    border-radius: 13px !important;
    background: var(--chef-orange) !important;
    color: #fff !important;
    font-weight: 750 !important;
    box-shadow: 0 10px 22px rgba(232, 111, 50, 0.24) !important;
    transition: transform 140ms ease, background 140ms ease, box-shadow 140ms ease;
}

.primary-action:hover {
    transform: translateY(-1px);
    background: var(--chef-orange-dark) !important;
    box-shadow: 0 13px 26px rgba(201, 84, 34, 0.27) !important;
}

.recipe-output {
    color: var(--chef-ink);
    line-height: 1.65;
}

.recipe-output h2,
.recipe-output h3 {
    color: var(--chef-forest);
    font-family: Georgia, "Times New Roman", serif;
    letter-spacing: -0.02em;
}

.recipe-output blockquote {
    border-left-color: var(--chef-orange) !important;
    background: #fff7ec;
    color: #654c3d;
}

.empty-state {
    display: flex;
    min-height: 330px;
    align-items: center;
    justify-content: center;
    text-align: center;
    color: #7b8780;
}

.browse-status {
    min-height: 0 !important;
    margin-bottom: 8px;
    color: var(--chef-muted);
    font-size: 0.83rem;
}

#chef-footer {
    margin-top: 22px;
    padding: 18px 4px 0;
    border-top: 1px solid var(--chef-line);
    color: var(--chef-muted);
    font-size: 0.78rem;
    line-height: 1.6;
    text-align: center;
}

@media (max-width: 760px) {
    .gradio-container {
        padding: 12px 12px 32px !important;
    }

    #chef-hero {
        padding: 30px 24px 28px;
        border-radius: 22px;
    }

    .hero-title {
        font-size: 2.65rem;
    }

    .hero-copy {
        font-size: 0.96rem;
    }

    #chef-workspace > .tab-nav {
        overflow-x: auto;
        flex-wrap: nowrap;
        justify-content: flex-start;
    }

    #chef-workspace > .tab-nav button {
        flex: 0 0 auto;
        white-space: nowrap;
    }

    .control-panel,
    .output-panel {
        border-radius: 18px !important;
    }

    .output-panel {
        min-height: 300px;
        padding: 20px !important;
    }
}
"""


def build_demo() -> gr.Blocks:
    model_name = Path(GENERATION_MODEL or BASE_MODEL).name
    adapter_label = "fine-tuned LoRA active" if not GENERATION_MODEL else "local checkpoint"
    hero = f"""
    <header id="chef-hero">
      <div class="hero-content">
        <div class="hero-eyebrow">Private local AI kitchen</div>
        <h1 class="hero-title">Cook with confidence,<br>plan with intelligence.</h1>
        <p class="hero-copy">
          Search a large recipe collection, create dishes around your requirements,
          and turn everyday ingredients into practical menus—all from one workspace.
        </p>
      </div>
      <div class="hero-stats" aria-label="Application status">
        <span class="status-pill"><strong>{MAX_ROWS:,}</strong> recipe workspace</span>
        <span class="status-pill"><strong>{html.escape(model_name)}</strong> model</span>
        <span class="status-pill"><strong>Local</strong> &amp; private</span>
      </div>
    </header>
    """
    system_note = (
        f"Dataset: {html.escape(DATASET_PATH.name)} · {html.escape(adapter_label)} · "
        f"adapter scale {LORA_SCALE:g} · AI loads on first use"
    )

    with gr.Blocks(
        title="Chef's Assistant",
        theme=CHEF_THEME,
        css=APP_CSS,
        fill_width=True,
    ) as demo:
        gr.HTML(hero, padding=False)
        gr.HTML(f'<div id="system-note">{system_note}</div>', padding=False)

        with gr.Tabs(elem_id="chef-workspace"):
            with gr.Tab("Search recipes"):
                gr.HTML(
                    """
                    <section class="tool-heading">
                      <div class="tool-kicker">Discover</div>
                      <h2 class="tool-title">Find the right recipe, faster.</h2>
                      <p class="tool-description">Describe what you want in natural language. The dataset supplies grounded candidates and the local Llama model ranks the closest matches.</p>
                    </section>
                    """,
                    padding=False,
                )
                with gr.Row(elem_classes="workspace-grid"):
                    with gr.Column(scale=4, min_width=310):
                        with gr.Group(elem_classes="control-panel"):
                            gr.HTML('<div class="panel-label">Search details</div>', padding=False)
                            search_query = gr.Textbox(
                                label="What would you like to cook?",
                                placeholder="Quick spicy chicken with rice",
                                lines=3,
                            )
                            gr.HTML(
                                '<p class="field-note">Try a cuisine, main ingredient, cooking style, or time limit.</p>',
                                padding=False,
                            )
                            result_count = gr.Slider(
                                1, 10, value=5, step=1, label="Number of matches"
                            )
                            search_button = gr.Button(
                                "Find matching recipes  →",
                                variant="primary",
                                elem_classes="primary-action",
                            )
                    with gr.Column(scale=7, min_width=380):
                        with gr.Group(elem_classes="output-panel"):
                            gr.HTML('<div class="panel-label">Recommended matches</div>', padding=False)
                            search_output = gr.Markdown(
                                "<div class='empty-state'>Your best recipe matches will appear here.</div>",
                                elem_classes="recipe-output",
                            )
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

            with gr.Tab("Surprise me"):
                gr.HTML(
                    """
                    <section class="tool-heading">
                      <div class="tool-kicker">Inspiration</div>
                      <h2 class="tool-title">Let the kitchen choose.</h2>
                      <p class="tool-description">Pick a collection and discover a complete recipe at random—useful when you want inspiration without another decision.</p>
                    </section>
                    """,
                    padding=False,
                )
                with gr.Row(elem_classes="workspace-grid"):
                    with gr.Column(scale=4, min_width=310):
                        with gr.Group(elem_classes="control-panel"):
                            gr.HTML('<div class="panel-label">Choose a collection</div>', padding=False)
                            category = gr.Radio(
                                ["All", "Chinese", "Western"],
                                value="All",
                                label="Recipe category",
                            )
                            random_button = gr.Button(
                                "Pick a recipe  →",
                                variant="primary",
                                elem_classes="primary-action",
                            )
                    with gr.Column(scale=7, min_width=380):
                        with gr.Group(elem_classes="output-panel"):
                            gr.HTML('<div class="panel-label">Today\'s discovery</div>', padding=False)
                            random_output = gr.Markdown(
                                "<div class='empty-state'>Choose a category, then let chance set the menu.</div>",
                                elem_classes="recipe-output",
                            )
                random_button.click(random_recipe, inputs=category, outputs=random_output)

            with gr.Tab("Browse collection"):
                gr.HTML(
                    """
                    <section class="tool-heading">
                      <div class="tool-kicker">Recipe library</div>
                      <h2 class="tool-title">Explore the collection your way.</h2>
                      <p class="tool-description">Browse alphabetically, narrow by category, and move through the recipe library one page at a time.</p>
                    </section>
                    """,
                    padding=False,
                )
                with gr.Row(elem_classes="workspace-grid"):
                    with gr.Column(scale=4, min_width=310):
                        with gr.Group(elem_classes="control-panel"):
                            gr.HTML('<div class="panel-label">Library filters</div>', padding=False)
                            browse_letter = gr.Dropdown(
                                ["All"] + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
                                value="All",
                                label="Recipe starts with",
                            )
                            browse_category = gr.Dropdown(
                                ["All", "Chinese", "Western"],
                                value="All",
                                label="Category",
                            )
                            browse_page = gr.Number(value=1, precision=0, label="Page number")
                            browse_button = gr.Button(
                                "Browse recipes  →",
                                variant="primary",
                                elem_classes="primary-action",
                            )
                    with gr.Column(scale=7, min_width=380):
                        with gr.Group(elem_classes="output-panel"):
                            gr.HTML('<div class="panel-label">Recipe collection</div>', padding=False)
                            browse_status = gr.Markdown(elem_classes="browse-status")
                            browse_output = gr.Markdown(
                                "<div class='empty-state'>Set your filters to open the recipe library.</div>",
                                elem_classes="recipe-output",
                            )
                browse_button.click(
                    browse_recipes,
                    inputs=[browse_letter, browse_category, browse_page],
                    outputs=[browse_output, browse_status],
                )

            with gr.Tab("Create a recipe"):
                gr.HTML(
                    """
                    <section class="tool-heading">
                      <div class="tool-kicker">AI recipe studio</div>
                      <h2 class="tool-title">Turn an idea into a complete dish.</h2>
                      <p class="tool-description">Describe the meal you have in mind. The local model builds a structured recipe and audits it against your cuisine and dietary requirements.</p>
                    </section>
                    """,
                    padding=False,
                )
                with gr.Row(elem_classes="workspace-grid"):
                    with gr.Column(scale=4, min_width=310):
                        with gr.Group(elem_classes="control-panel"):
                            gr.HTML('<div class="panel-label">Recipe brief</div>', padding=False)
                            generation_request = gr.Textbox(
                                label="Describe your recipe",
                                lines=5,
                                placeholder="A simple Malaysian-inspired spicy chicken dinner with rice",
                            )
                            servings = gr.Slider(1, 12, value=4, step=1, label="Servings")
                            dietary_notes = gr.Textbox(
                                label="Dietary requirements",
                                placeholder="Halal, no peanuts, low sodium",
                            )
                            generate_button = gr.Button(
                                "Create my recipe  →",
                                variant="primary",
                                elem_classes="primary-action",
                            )
                    with gr.Column(scale=7, min_width=380):
                        with gr.Group(elem_classes="output-panel"):
                            gr.HTML('<div class="panel-label">Your generated recipe</div>', padding=False)
                            generation_output = gr.Markdown(
                                "<div class='empty-state'>Your custom recipe will appear here.</div>",
                                elem_classes="recipe-output",
                            )
                generate_button.click(
                    generate_recipe,
                    inputs=[generation_request, servings, dietary_notes],
                    outputs=generation_output,
                )

            with gr.Tab("Plan a menu"):
                gr.HTML(
                    """
                    <section class="tool-heading">
                      <div class="tool-kicker">Menu planning</div>
                      <h2 class="tool-title">Build a practical plan for the days ahead.</h2>
                      <p class="tool-description">Set the schedule, people, budget target, dietary needs, and preferences. The planner selects grounded recipes and prepares one shopping list.</p>
                    </section>
                    """,
                    padding=False,
                )
                with gr.Row(elem_classes="workspace-grid"):
                    with gr.Column(scale=5, min_width=330):
                        with gr.Group(elem_classes="control-panel"):
                            gr.HTML('<div class="panel-label">Planning brief</div>', padding=False)
                            with gr.Row():
                                menu_days = gr.Slider(1, 7, value=3, step=1, label="Days")
                                menu_meals = gr.Slider(
                                    1, 3, value=2, step=1, label="Meals per day"
                                )
                            menu_servings = gr.Slider(
                                1, 12, value=4, step=1, label="People"
                            )
                            menu_budget = gr.Textbox(
                                label="Budget target", placeholder="RM150 total"
                            )
                            menu_dietary = gr.Textbox(
                                label="Dietary requirements",
                                placeholder="Halal, no peanuts",
                            )
                            menu_preferences = gr.Textbox(
                                label="Preferences and available ingredients",
                                lines=4,
                                placeholder="Malaysian and Chinese food; chicken, rice, vegetables",
                            )
                            menu_button = gr.Button(
                                "Create menu plan  →",
                                variant="primary",
                                elem_classes="primary-action",
                            )
                    with gr.Column(scale=7, min_width=380):
                        with gr.Group(elem_classes="output-panel"):
                            gr.HTML('<div class="panel-label">Menu and shopping list</div>', padding=False)
                            menu_output = gr.Markdown(
                                "<div class='empty-state'>Your menu plan and consolidated shopping list will appear here.</div>",
                                elem_classes="recipe-output",
                            )
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

        gr.HTML(
            """
            <footer id="chef-footer">
              Chef's Assistant runs locally on your machine. Always verify allergens,
              food safety, halal certification, nutrition, and ingredient prices before use.
            </footer>
            """,
            padding=False,
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
