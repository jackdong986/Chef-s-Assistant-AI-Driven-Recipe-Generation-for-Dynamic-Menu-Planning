"""Temporary Gradio server for recipe search, discovery, and generation."""

from __future__ import annotations

import argparse
import ast
import csv
import html
import json
import os
import re
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_PATH = Path.home() / "Downloads" / "RAW_recipes_with_amount.csv"
DEFAULT_BASE_MODEL_PATH = Path(
    r"C:\Users\Jack\Downloads\aistackphison\aistackphison\Llama-3.2-3B-Instruct"
)
DEFAULT_ADAPTER_PATH = PROJECT_ROOT / "models" / "chef-llama-3.2-3b-lora"
DEFAULT_INDEX_PATH = PROJECT_ROOT / ".cache" / "recipes.sqlite3"
EXPORT_DIR = PROJECT_ROOT / ".cache" / "exports"

# Keep downloaded Hugging Face models with this checkout. The directory is ignored by Git.
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface"))

import gradio as gr
import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from recipe_quality import (
    QualityIssue,
    find_blocked_terms,
    normalize_recipe,
    quality_score,
    validate_recipe,
)
from recipe_store import RecipeStore

DATASET_PATH = Path(os.getenv("RECIPE_DATASET_PATH", str(DEFAULT_DATASET_PATH))).expanduser()
INDEX_PATH = Path(os.getenv("RECIPE_INDEX_PATH", str(DEFAULT_INDEX_PATH))).expanduser()
GENERATION_MODEL = os.getenv("RECIPE_GENERATION_MODEL", "").strip()
BASE_MODEL = os.getenv("RECIPE_BASE_MODEL", str(DEFAULT_BASE_MODEL_PATH))
LORA_ADAPTER = os.getenv("RECIPE_LORA_ADAPTER", str(DEFAULT_ADAPTER_PATH))
LORA_SCALE = float(os.getenv("RECIPE_LORA_SCALE", "0.5"))


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
def get_recipe_store() -> RecipeStore:
    return RecipeStore(DATASET_PATH, INDEX_PATH)


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
    max_minutes: int | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    frame = (
        source_frame.copy()
        if source_frame is not None
        else get_recipe_store().search(
            query,
            limit=max(100, count * 5),
            max_minutes=max_minutes,
        )
    )
    normalized_query = query.lower().strip()
    tokens = _query_tokens(normalized_query)
    if not tokens or frame.empty:
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


def ai_search(
    query: str,
    result_count: int,
    dietary_notes: str = "",
    max_minutes: int = 0,
) -> str:
    query = _safe_text(query)
    if not query:
        return "Enter a dish, ingredient, cuisine, or description to search."

    try:
        count = min(max(int(result_count), 1), 10)
        notes = _safe_text(dietary_notes)
        requested_cuisines = _requested_cuisines(query)
        source_frame = get_recipe_store().search(
            query,
            limit=max(100, count * 15),
            max_minutes=int(max_minutes or 0),
        )
        if requested_cuisines:
            cuisine_mask = pd.Series(True, index=source_frame.index)
            for cuisine in requested_cuisines:
                cuisine_mask &= source_frame["search_text"].str.contains(
                    cuisine, case=False, regex=False
                )
            source_frame = source_frame[cuisine_mask]
        if notes:
            source_frame = source_frame[
                source_frame.apply(
                    lambda row: _matches_dietary_notes(row, notes), axis=1
                )
            ]
        candidates, lexical_scores = _rank_recipes(
            query,
            max(20, count),
            source_frame=source_frame,
            max_minutes=int(max_minutes or 0),
        )
        if candidates.empty:
            return "No recipe in the complete catalog matches every selected filter."

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
        recipe = get_recipe_store().random_recipe(category)
        if recipe is None:
            return f"No {category.lower()} recipe was found in the catalog."
        return _recipe_markdown(recipe)
    except Exception as exc:
        return f"### Random recipe unavailable\n\n{html.escape(str(exc))}"


def _matches_dietary_notes(row: pd.Series, dietary_notes: str) -> bool:
    recipe_text = " ".join(
        _safe_text(row.get(column))
        for column in ("name", "description", "tags", "ingredients", "steps")
    ).lower()
    return not find_blocked_terms(recipe_text, dietary_notes)


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


def _render_generated_recipe(
    recipe: dict[str, Any],
    issues: list[QualityIssue],
    *,
    audited: bool,
) -> str:
    ingredients = _deduplicate(recipe.get("ingredients", []))[:30]
    steps = _deduplicate(recipe.get("steps", []))[:25]
    score = quality_score(issues)
    status = "AI audit applied" if audited else "Draft passed deterministic checks"
    lines = [
        f"## {html.escape(_safe_text(recipe.get('name')) or 'Generated recipe')}",
        "",
        f"**Quality score: {score}/100** · {status}",
        "",
        html.escape(_safe_text(recipe.get("description"))),
        "",
        f"**Servings:** {html.escape(_safe_text(recipe.get('servings')))} · "
        f"**Total time:** {html.escape(_safe_text(recipe.get('minutes')))} minutes",
        "",
        "### Ingredients",
    ]
    lines.extend(f"- {html.escape(_safe_text(item))}" for item in ingredients)
    lines.extend(["", "### Method"])
    lines.extend(
        f"{index}. {html.escape(_safe_text(step))}"
        for index, step in enumerate(steps, 1)
    )
    if issues:
        lines.extend(["", "### Chef review notes"])
        lines.extend(f"- {html.escape(issue.message)}" for issue in issues)
    return "\n".join(lines)


def generate_recipe_record(
    description: str,
    servings: int,
    dietary_notes: str,
) -> tuple[dict[str, Any], list[QualityIssue], bool]:
    description = _safe_text(description)
    if not description:
        raise ValueError("Describe the recipe you want to generate.")

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
        "servings, minutes, ingredients, and steps. Ingredients and steps must be JSON arrays of "
        "strings, and every ingredient string must begin with a practical quantity."
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

    draft_text = _generate_with_model(
        system_message,
        user_message,
        max_new_tokens=650,
        temperature=0.25,
    )
    if not draft_text:
        raise ValueError("The model returned an empty recipe. Try a more specific description.")
    draft = normalize_recipe(_parse_recipe_json(draft_text))
    issues = validate_recipe(
        draft,
        cuisines=requested_cuisines,
        dietary_notes=notes,
        servings=int(servings),
    )
    audited = False
    final_recipe = draft
    if issues:
        audited = True
        issue_summary = " ".join(issue.message for issue in issues)
        correction_message = (
            f"The deterministic checks found these problems: {issue_summary} "
            f"Original request: {description}. Servings: {int(servings)}. "
            f"Mandatory dietary requirements: {notes}. "
            f"Mandatory cuisines: {', '.join(requested_cuisines) or 'none specified'}. "
            "Correct the draft while preserving the request. Use a meaningful dish name. Every "
            "ingredient must have a practical quantity, every ingredient mentioned in the steps "
            "must appear in the ingredient list, and every main ingredient must be used in the "
            "steps. Do not switch between rice, noodles, pasta, or another starch. Remove duplicate "
            "and unused ingredients. Return only valid JSON with name, description, servings, "
            f"minutes, ingredients, and steps. Draft: {draft_text}"
        )
        corrected_text = _generate_with_model(
            system_message,
            correction_message,
            max_new_tokens=700,
            temperature=0.1,
            adapter_multiplier=0.0,
        )
        corrected = normalize_recipe(_parse_recipe_json(corrected_text))
        if corrected:
            final_recipe = corrected
    final_issues = validate_recipe(
        final_recipe,
        cuisines=requested_cuisines,
        dietary_notes=notes,
        servings=int(servings),
    )
    return final_recipe, final_issues, audited


def generate_recipe(description: str, servings: int, dietary_notes: str) -> str:
    try:
        recipe, issues, audited = generate_recipe_record(
            description, servings, dietary_notes
        )
        return (
            f"{_render_generated_recipe(recipe, issues, audited=audited)}\n\n"
            "> AI-generated recipe: verify allergens, food safety, and cooking temperatures before use."
        )
    except Exception as exc:
        return f"### Generation unavailable\n\n{html.escape(str(exc))}"


def _write_export(prefix: str, suffix: str, content: str) -> str:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = EXPORT_DIR / f"{prefix}-{uuid.uuid4().hex[:8]}{suffix}"
    path.write_text(content, encoding="utf-8")
    return str(path)


def generate_recipe_ui(
    description: str,
    servings: int,
    dietary_notes: str,
) -> tuple[str, str | None]:
    result = generate_recipe(description, servings, dietary_notes)
    if result.startswith("### Generation unavailable") or not _safe_text(description):
        return result, None
    return result, _write_export("recipe", ".md", result)


def browse_recipes(letter: str, category: str, page: int) -> tuple[str, str]:
    try:
        page_frame, total, current_page = get_recipe_store().browse(
            letter=letter,
            category=category,
            page=int(page or 1),
            page_size=6,
        )
        if page_frame.empty:
            return "No recipes match these filters.", "Page 0 of 0"

        total_pages = max(1, (total + 5) // 6)
        content = "\n\n---\n\n".join(
            _recipe_markdown(row) for _, row in page_frame.iterrows()
        )
        return content, f"Page {current_page:,} of {total_pages:,} · {total:,} recipes"
    except Exception as exc:
        return f"### Browse unavailable\n\n{html.escape(str(exc))}", ""


def _comma_terms(value: str) -> list[str]:
    return [
        item.strip().lower()
        for item in re.split(r"[,;\n]", _safe_text(value))
        if item.strip()
    ]


def create_menu_plan(
    days: int,
    meals_per_day: int,
    servings: int,
    budget: str,
    dietary_notes: str,
    preferences: str,
    meal_types: list[str] | None = None,
    available_ingredients: str = "",
    pantry_ingredients: str = "",
    avoid_repeats: bool = True,
    reuse_leftovers: bool = False,
) -> tuple[str, list[dict[str, str]], list[str]]:
    dietary = _safe_text(dietary_notes) or "none"
    preference_text = _safe_text(preferences) or "easy practical meals"
    available_terms = _comma_terms(available_ingredients)
    pantry_terms = _comma_terms(pantry_ingredients)
    requested_cuisines = _requested_cuisines(preference_text)
    required_slots = max(1, int(days)) * max(1, int(meals_per_day))
    unique_required = (required_slots + 1) // 2 if reuse_leftovers else required_slots
    query = " ".join(
        part
        for part in (
            preference_text,
            " ".join(available_terms),
            dietary if dietary != "none" else "",
        )
        if part
    )
    candidates = get_recipe_store().search(
        query or "easy main dish",
        limit=max(300, unique_required * 30),
    )
    if requested_cuisines:
        cuisine_mask = pd.Series(True, index=candidates.index)
        for cuisine in requested_cuisines:
            cuisine_mask &= candidates["search_text"].str.contains(
                cuisine, case=False, regex=False
            )
        candidates = candidates[cuisine_mask]
    candidates = candidates[
        candidates.apply(lambda row: _matches_dietary_notes(row, dietary), axis=1)
    ].copy()
    if available_terms and not candidates.empty:
        candidates["available_match"] = candidates["ingredients"].apply(
            lambda value: sum(
                1
                for term in available_terms
                if _ingredient_contains_core(value, term)
            )
        )
        candidates = candidates.sort_values(
            ["available_match", "fts_rank"], ascending=[False, True]
        )
    candidates = candidates.head(max(60, unique_required * 8)).reset_index(drop=True)
    if candidates.empty:
        raise ValueError("No catalog recipes satisfy the selected menu requirements.")
    if avoid_repeats and len(candidates) < unique_required:
        raise ValueError(
            "Not enough catalog recipes satisfy every mandatory filter without repeats. "
            "Broaden the requirements or allow repeats."
        )

    candidate_lines = [
        f"ID {candidate_id}: {_safe_text(row['name'])}; "
        f"{_safe_text(row['minutes'])} minutes; ingredients: "
        f"{', '.join(_parse_list(row['ingredients'])[:10])}"
        for candidate_id, row in candidates.iterrows()
    ]
    selection = _generate_with_model(
        "You select recipes for a professional kitchen menu. Balance variety, prep time, "
        "ingredient reuse, and all stated cuisine and dietary constraints. Return only valid "
        'JSON like {"selected_ids":[2,5,1]}. Do not include commentary.',
        f"Select exactly {unique_required} IDs for {int(servings)} people. "
        f"Budget target: {_safe_text(budget) or 'not specified'}. Requirements: {dietary}; "
        f"preferences: {preference_text}; available: "
        f"{', '.join(available_terms) or 'not specified'}. Candidates:\n"
        + "\n".join(candidate_lines),
        max_new_tokens=220,
        temperature=0.0,
    )
    selected_ids: list[int] = []
    try:
        parsed = json.loads(selection[selection.find("{") : selection.rfind("}") + 1])
        for value in parsed.get("selected_ids", []):
            selected_id = int(value)
            if 0 <= selected_id < len(candidates):
                if not avoid_repeats or selected_id not in selected_ids:
                    selected_ids.append(selected_id)
    except (ValueError, TypeError, json.JSONDecodeError):
        selected_ids = []
    candidate_position = 0
    while len(selected_ids) < unique_required:
        selected_id = candidate_position % len(candidates)
        candidate_position += 1
        if not avoid_repeats or selected_id not in selected_ids:
            selected_ids.append(selected_id)

    chosen_rows = [candidates.iloc[index] for index in selected_ids[:unique_required]]
    scheduled_rows: list[tuple[pd.Series, bool]] = []
    chosen_position = 0
    for slot in range(required_slots):
        is_leftover = reuse_leftovers and slot % 2 == 1
        if is_leftover:
            scheduled_rows.append((scheduled_rows[-1][0], True))
        else:
            scheduled_rows.append((chosen_rows[chosen_position], False))
            chosen_position += 1

    labels = [_safe_text(label) for label in (meal_types or []) if _safe_text(label)]
    if not labels:
        labels = [f"Meal {index}" for index in range(1, int(meals_per_day) + 1)]
    lines = [
        "# Kitchen Menu Plan",
        "",
        f"For **{int(servings)} people** · **{int(days)} days** · "
        f"Budget target: **{html.escape(_safe_text(budget) or 'not specified')}**",
        "",
    ]
    shopping_items: dict[str, str] = {}
    export_rows: list[dict[str, str]] = []
    slot_position = 0
    for day_number in range(1, int(days) + 1):
        lines.extend([f"## Day {day_number}", ""])
        for meal_number in range(1, int(meals_per_day) + 1):
            row, is_leftover = scheduled_rows[slot_position]
            slot_position += 1
            meal_label = labels[(meal_number - 1) % len(labels)]
            name_text = _safe_text(row["name"]).title()
            ingredients = _parse_list(row.get("ingredients"))
            steps = _parse_list(row.get("steps"))
            heading = f"{meal_label} — {'Leftovers: ' if is_leftover else ''}{name_text}"
            lines.append(f"### {html.escape(heading)}")
            if is_leftover:
                lines.extend(
                    [
                        "Use the previous prepared portion. Cool promptly, refrigerate safely, "
                        "and reheat until piping hot.",
                        "",
                    ]
                )
            else:
                lines.extend(
                    [
                        f"Preparation time: **{html.escape(_safe_text(row.get('minutes')))} minutes**",
                        "",
                        "**Ingredients**",
                    ]
                )
                lines.extend(f"- {html.escape(item)}" for item in ingredients[:20])
                lines.extend(["", "**Method**"])
                lines.extend(
                    f"{index}. {html.escape(step)}"
                    for index, step in enumerate(steps[:20], 1)
                )
                lines.append("")
                for item in ingredients:
                    lowered = item.lower().strip()
                    if lowered and not any(term in lowered for term in pantry_terms):
                        shopping_items.setdefault(lowered, item)
            export_rows.append(
                {
                    "day": str(day_number),
                    "meal": meal_label,
                    "recipe": name_text,
                    "minutes": _safe_text(row.get("minutes")),
                    "leftovers": "yes" if is_leftover else "no",
                    "ingredients": "; ".join(ingredients),
                }
            )

    shopping_list = sorted(shopping_items.values(), key=str.lower)
    lines.extend(["## Consolidated Shopping List", ""])
    lines.extend(f"- {html.escape(item)}" for item in shopping_list)
    if pantry_terms:
        lines.extend(
            ["", f"Pantry items excluded: **{html.escape(', '.join(pantry_terms))}**"]
        )
    lines.extend(
        [
            "",
            "> The budget is a planning target because the dataset has no verified ingredient prices.",
            "> Verify allergens, halal certification, storage, food safety, and local prices before service.",
        ]
    )
    return "\n".join(lines), export_rows, shopping_list


def generate_menu_plan(
    days: int,
    meals_per_day: int,
    servings: int,
    budget: str,
    dietary_notes: str,
    preferences: str,
    meal_types: list[str] | None = None,
    available_ingredients: str = "",
    pantry_ingredients: str = "",
    avoid_repeats: bool = True,
    reuse_leftovers: bool = False,
) -> str:
    try:
        markdown, _, _ = create_menu_plan(
            days, meals_per_day, servings, budget, dietary_notes, preferences,
            meal_types, available_ingredients, pantry_ingredients,
            avoid_repeats, reuse_leftovers,
        )
        return markdown
    except Exception as exc:
        return f"### Menu planning unavailable\n\n{html.escape(str(exc))}"


def generate_menu_plan_ui(
    days: int,
    meals_per_day: int,
    servings: int,
    meal_types: list[str],
    budget: str,
    dietary_notes: str,
    preferences: str,
    available_ingredients: str,
    pantry_ingredients: str,
    avoid_repeats: bool,
    reuse_leftovers: bool,
) -> tuple[str, str | None, str | None]:
    try:
        markdown, rows, shopping = create_menu_plan(
            days, meals_per_day, servings, budget, dietary_notes, preferences,
            meal_types, available_ingredients, pantry_ingredients,
            avoid_repeats, reuse_leftovers,
        )
        markdown_path = _write_export("menu-plan", ".md", markdown)
        EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        csv_path = EXPORT_DIR / f"menu-plan-{uuid.uuid4().hex[:8]}.csv"
        with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["day", "meal", "recipe", "minutes", "leftovers", "ingredients"],
            )
            writer.writeheader()
            writer.writerows(rows)
            handle.write("\nSHOPPING LIST\n")
            for item in shopping:
                handle.write(f"{item}\n")
        return markdown, str(csv_path), markdown_path
    except Exception as exc:
        return f"### Menu planning unavailable\n\n{html.escape(str(exc))}", None, None


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
        <span class="status-pill"><strong>Complete</strong> indexed recipe catalog</span>
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
                            search_dietary = gr.Textbox(
                                label="Dietary and allergen filters",
                                placeholder="Halal, vegetarian, no peanuts",
                            )
                            search_minutes = gr.Slider(
                                0,
                                240,
                                value=0,
                                step=15,
                                label="Maximum time (0 = any)",
                            )
                            gr.Examples(
                                examples=[
                                    ["Quick Malaysian chicken and rice"],
                                    ["Chinese vegetable dinner"],
                                    ["One-pot Western comfort food"],
                                ],
                                inputs=search_query,
                                label="Chef shortcuts",
                            )
                            with gr.Row():
                                search_button = gr.Button(
                                    "Find matching recipes  →",
                                    variant="primary",
                                    elem_classes="primary-action",
                                )
                                search_cancel = gr.Button("Cancel", variant="secondary")
                    with gr.Column(scale=7, min_width=380):
                        with gr.Group(elem_classes="output-panel"):
                            gr.HTML('<div class="panel-label">Recommended matches</div>', padding=False)
                            search_output = gr.Markdown(
                                "<div class='empty-state'>Your best recipe matches will appear here.</div>",
                                elem_classes="recipe-output",
                            )
                search_event = search_button.click(
                    ai_search,
                    inputs=[search_query, result_count, search_dietary, search_minutes],
                    outputs=search_output,
                )
                search_submit_event = search_query.submit(
                    ai_search,
                    inputs=[search_query, result_count, search_dietary, search_minutes],
                    outputs=search_output,
                )
                search_cancel.click(
                    fn=None,
                    cancels=[search_event, search_submit_event],
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
                            gr.Examples(
                                examples=[
                                    ["Malaysian spicy chicken with rice", 4, "Halal, no peanuts"],
                                    ["Chinese tofu and vegetable stir-fry", 2, "Vegan"],
                                    ["Western baked fish dinner", 6, "Gluten-free"],
                                ],
                                inputs=[generation_request, servings, dietary_notes],
                                label="Service-ready examples",
                            )
                            with gr.Row():
                                generate_button = gr.Button(
                                    "Create my recipe  →",
                                    variant="primary",
                                    elem_classes="primary-action",
                                )
                                generation_cancel = gr.Button("Cancel", variant="secondary")
                    with gr.Column(scale=7, min_width=380):
                        with gr.Group(elem_classes="output-panel"):
                            gr.HTML('<div class="panel-label">Your generated recipe</div>', padding=False)
                            generation_output = gr.Markdown(
                                "<div class='empty-state'>Your custom recipe will appear here.</div>",
                                elem_classes="recipe-output",
                            )
                            recipe_download = gr.File(
                                label="Download recipe",
                                interactive=False,
                            )
                generation_event = generate_button.click(
                    generate_recipe_ui,
                    inputs=[generation_request, servings, dietary_notes],
                    outputs=[generation_output, recipe_download],
                )
                generation_cancel.click(fn=None, cancels=[generation_event])

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
                            menu_meal_types = gr.CheckboxGroup(
                                ["Breakfast", "Lunch", "Dinner"],
                                value=["Lunch", "Dinner"],
                                label="Service periods",
                            )
                            menu_budget = gr.Textbox(
                                label="Budget target", placeholder="RM150 total"
                            )
                            menu_dietary = gr.Textbox(
                                label="Dietary requirements",
                                placeholder="Halal, no peanuts",
                            )
                            menu_preferences = gr.Textbox(
                                label="Cuisine and menu style",
                                lines=3,
                                placeholder="Malaysian and Chinese dishes; quick family-style meals",
                            )
                            menu_available = gr.Textbox(
                                label="Ingredients already available",
                                placeholder="Chicken, rice, carrots, cabbage",
                            )
                            menu_pantry = gr.Textbox(
                                label="Pantry items to exclude from shopping",
                                placeholder="Salt, pepper, cooking oil, soy sauce",
                            )
                            with gr.Row():
                                menu_no_repeats = gr.Checkbox(
                                    value=True,
                                    label="Avoid repeated dishes",
                                )
                                menu_leftovers = gr.Checkbox(
                                    value=False,
                                    label="Plan safe leftovers",
                                )
                            gr.Examples(
                                examples=[
                                    ["Malaysian weeknight menu"],
                                    ["Chinese family-style menu"],
                                    ["Balanced Western lunch service"],
                                ],
                                inputs=menu_preferences,
                                label="Menu brief examples",
                            )
                            with gr.Row():
                                menu_button = gr.Button(
                                    "Create menu plan  →",
                                    variant="primary",
                                    elem_classes="primary-action",
                                )
                                menu_cancel = gr.Button("Cancel", variant="secondary")
                    with gr.Column(scale=7, min_width=380):
                        with gr.Group(elem_classes="output-panel"):
                            gr.HTML('<div class="panel-label">Menu and shopping list</div>', padding=False)
                            menu_output = gr.Markdown(
                                "<div class='empty-state'>Your menu plan and consolidated shopping list will appear here.</div>",
                                elem_classes="recipe-output",
                            )
                            with gr.Row():
                                menu_csv_download = gr.File(
                                    label="Download kitchen CSV", interactive=False
                                )
                                menu_markdown_download = gr.File(
                                    label="Download printable plan", interactive=False
                                )
                menu_event = menu_button.click(
                    generate_menu_plan_ui,
                    inputs=[
                        menu_days,
                        menu_meals,
                        menu_servings,
                        menu_meal_types,
                        menu_budget,
                        menu_dietary,
                        menu_preferences,
                        menu_available,
                        menu_pantry,
                        menu_no_repeats,
                        menu_leftovers,
                    ],
                    outputs=[menu_output, menu_csv_download, menu_markdown_download],
                )
                menu_cancel.click(fn=None, cancels=[menu_event])

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
