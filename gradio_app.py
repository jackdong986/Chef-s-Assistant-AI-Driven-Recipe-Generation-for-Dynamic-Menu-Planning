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
    --chef-ink: #17241e;
    --chef-muted: #66736c;
    --chef-forest: #153e30;
    --chef-forest-2: #245a45;
    --chef-sage: #e8efe9;
    --chef-sage-2: #d8e5dc;
    --chef-orange: #e7652b;
    --chef-orange-dark: #c84e1d;
    --chef-cream: #f4f1e8;
    --chef-paper: #fffefa;
    --chef-line: rgba(27, 57, 44, 0.13);
    --chef-line-strong: rgba(27, 57, 44, 0.22);
    --chef-shadow-sm: 0 8px 24px rgba(24, 51, 40, 0.06);
    --chef-shadow: 0 20px 55px rgba(24, 51, 40, 0.10);
}

* {
    box-sizing: border-box;
}

body,
.gradio-container {
    background:
        radial-gradient(circle at 5% 2%, rgba(231, 101, 43, 0.09), transparent 28rem),
        radial-gradient(circle at 96% 18%, rgba(36, 90, 69, 0.10), transparent 34rem),
        var(--chef-cream) !important;
    color: var(--chef-ink) !important;
}

html,
body {
    max-width: 100%;
    overflow-x: hidden;
}

.gradio-container {
    --body-background-fill: #f4f1e8;
    --body-background-fill-dark: #f4f1e8;
    --body-text-color: #17241e;
    --body-text-color-dark: #17241e;
    --body-text-color-subdued: #66736c;
    --body-text-color-subdued-dark: #66736c;
    --background-fill-primary: #fffefa;
    --background-fill-primary-dark: #fffefa;
    --background-fill-secondary: #f4f1e8;
    --background-fill-secondary-dark: #f4f1e8;
    --block-background-fill: #fffefa;
    --block-background-fill-dark: #fffefa;
    --block-border-color: rgba(27, 57, 44, 0.13);
    --block-border-color-dark: rgba(27, 57, 44, 0.13);
    --block-info-text-color: #66736c;
    --block-info-text-color-dark: #66736c;
    --block-label-background-fill: transparent;
    --block-label-background-fill-dark: transparent;
    --block-label-text-color: #35483f;
    --block-label-text-color-dark: #35483f;
    --block-title-text-color: #17241e;
    --block-title-text-color-dark: #17241e;
    --border-color-primary: rgba(27, 57, 44, 0.17);
    --border-color-primary-dark: rgba(27, 57, 44, 0.17);
    --input-background-fill: #fffefa;
    --input-background-fill-dark: #fffefa;
    --input-border-color: rgba(27, 57, 44, 0.17);
    --input-border-color-dark: rgba(27, 57, 44, 0.17);
    --input-placeholder-color: #8b958f;
    --input-placeholder-color-dark: #8b958f;
    --shadow-drop: 0 1px 2px rgba(24, 51, 40, 0.05);
    --shadow-drop-lg: 0 20px 55px rgba(24, 51, 40, 0.10);
    width: 100% !important;
    min-width: 0 !important;
    margin: 0 auto !important;
    max-width: 1420px !important;
    padding: 18px 28px 54px !important;
}

#chef-app-bar,
#chef-hero,
#system-note,
#chef-workspace {
    width: 100%;
    max-width: 100%;
}

#chef-app-bar {
    display: flex;
    min-height: 62px;
    align-items: center;
    justify-content: space-between;
    gap: 22px;
    margin-bottom: 14px;
    padding: 10px 4px;
}

.brand-lockup {
    display: flex;
    align-items: center;
    gap: 12px;
}

.brand-mark {
    display: grid;
    width: 42px;
    height: 42px;
    place-items: center;
    border-radius: 13px;
    background: var(--chef-forest);
    box-shadow: 0 9px 20px rgba(21, 62, 48, 0.20);
    color: #fff;
    font-family: Georgia, "Times New Roman", serif;
    font-size: 1.15rem;
    font-weight: 700;
}

.brand-name {
    color: var(--chef-ink);
    font-size: 0.98rem;
    font-weight: 800;
    letter-spacing: -0.015em;
}

.brand-subtitle {
    margin-top: 2px;
    color: var(--chef-muted);
    font-size: 0.72rem;
}

.top-status {
    display: flex;
    align-items: center;
    gap: 9px;
}

.privacy-badge,
.model-badge {
    display: inline-flex;
    min-height: 34px;
    align-items: center;
    gap: 7px;
    padding: 7px 11px;
    border: 1px solid var(--chef-line);
    border-radius: 999px;
    background: rgba(255, 254, 250, 0.72);
    color: #506158;
    font-size: 0.74rem;
    font-weight: 650;
}

.privacy-badge::before {
    content: "";
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #39a66e;
    box-shadow: 0 0 0 4px rgba(57, 166, 110, 0.12);
}

#chef-hero {
    position: relative;
    display: grid;
    grid-template-columns: minmax(0, 1.45fr) minmax(280px, 0.55fr);
    gap: 46px;
    overflow: hidden;
    padding: 46px 48px;
    border: 1px solid rgba(255, 255, 255, 0.09);
    border-radius: 28px;
    background: linear-gradient(135deg, #12382b 0%, #1b4c39 60%, #725038 145%);
    box-shadow: 0 26px 70px rgba(21, 62, 48, 0.22);
    color: #fffaf0;
}

#chef-hero::before,
#chef-hero::after {
    content: "";
    position: absolute;
    border-radius: 50%;
    pointer-events: none;
}

#chef-hero::before {
    width: 240px;
    height: 240px;
    right: 18%;
    bottom: -180px;
    background: rgba(231, 101, 43, 0.15);
    filter: blur(3px);
}

#chef-hero::after {
    width: 340px;
    height: 340px;
    right: -155px;
    top: -185px;
    border: 72px solid rgba(245, 166, 97, 0.09);
}

.hero-content {
    position: relative;
    z-index: 1;
    align-self: center;
    max-width: 800px;
}

.hero-eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 9px;
    margin-bottom: 18px;
    color: #ffd4ad;
    font-size: 0.73rem;
    font-weight: 800;
    letter-spacing: 0.16em;
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
    max-width: 760px;
    color: #fffdf8;
    font-family: Georgia, "Times New Roman", serif;
    font-size: clamp(2.5rem, 5vw, 4.65rem);
    font-weight: 600;
    letter-spacing: -0.055em;
    line-height: 1.01;
}

.hero-copy {
    max-width: 680px;
    margin: 20px 0 0;
    color: rgba(255, 253, 248, 0.76);
    font-size: 1rem;
    line-height: 1.62;
}

.hero-command {
    position: relative;
    z-index: 1;
    align-self: stretch;
    padding: 20px;
    border: 1px solid rgba(255, 255, 255, 0.11);
    border-radius: 20px;
    background: rgba(255, 255, 255, 0.075);
    backdrop-filter: blur(12px);
}

.command-kicker {
    margin: 1px 0 15px;
    color: rgba(255, 244, 230, 0.60);
    font-size: 0.68rem;
    font-weight: 800;
    letter-spacing: 0.14em;
    text-transform: uppercase;
}

.command-item {
    display: grid;
    grid-template-columns: 32px 1fr;
    align-items: center;
    gap: 11px;
    padding: 13px 2px;
    border-top: 1px solid rgba(255, 255, 255, 0.09);
}

.command-item:first-of-type {
    border-top: 0;
}

.command-index {
    display: grid;
    width: 30px;
    height: 30px;
    place-items: center;
    border: 1px solid rgba(255, 255, 255, 0.14);
    border-radius: 9px;
    background: rgba(255, 255, 255, 0.08);
    color: #ffbd82;
    font-size: 0.7rem;
    font-weight: 800;
}

.command-title {
    color: #fffdf8;
    font-size: 0.84rem;
    font-weight: 750;
}

.command-copy {
    margin-top: 2px;
    color: rgba(255, 253, 248, 0.58);
    font-size: 0.72rem;
    line-height: 1.4;
}

#system-note {
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 13px 2px 19px;
    padding: 10px 16px;
    border: 1px solid var(--chef-line);
    border-radius: 12px;
    background: rgba(255, 254, 250, 0.66);
    color: var(--chef-muted);
    font-size: 0.75rem;
    text-align: center;
}

#chef-workspace {
    margin-top: 4px;
}

.gradio-container .tab-nav,
.gradio-container [role="tablist"] {
    position: sticky;
    z-index: 20;
    top: 10px;
    display: grid;
    grid-template-columns: repeat(5, minmax(0, 1fr));
    gap: 5px;
    margin-bottom: 22px;
    padding: 6px;
    border: 1px solid var(--chef-line);
    border-radius: 16px;
    background: rgba(255, 254, 250, 0.92);
    box-shadow: 0 10px 28px rgba(24, 51, 40, 0.08);
    backdrop-filter: blur(14px);
    max-width: 100%;
}

.gradio-container .tab-nav button,
.gradio-container [role="tablist"] [role="tab"] {
    min-width: 0;
    min-height: 46px;
    border: 0 !important;
    border-radius: 11px !important;
    color: #59685f;
    font-size: 0.82rem;
    font-weight: 700;
    white-space: nowrap;
    transition: background 140ms ease, color 140ms ease, transform 140ms ease;
}

.gradio-container .tab-nav button.selected,
.gradio-container [role="tablist"] [role="tab"][aria-selected="true"] {
    background: var(--chef-forest) !important;
    color: #fffdf8 !important;
    box-shadow: 0 8px 20px rgba(21, 62, 48, 0.22);
}

.gradio-container .tab-nav button:not(.selected):hover,
.gradio-container [role="tablist"] [role="tab"]:not([aria-selected="true"]):hover {
    background: var(--chef-sage) !important;
    color: var(--chef-forest) !important;
}

.tool-heading {
    display: grid;
    grid-template-columns: 62px minmax(0, 1fr);
    gap: 17px;
    align-items: start;
    margin: 3px 0 22px;
    padding: 0 4px;
}

.tool-number {
    display: grid;
    width: 54px;
    height: 54px;
    place-items: center;
    border: 1px solid var(--chef-line-strong);
    border-radius: 16px;
    background: rgba(255, 254, 250, 0.74);
    box-shadow: var(--chef-shadow-sm);
    color: var(--chef-orange-dark);
    font-family: Georgia, "Times New Roman", serif;
    font-size: 1.18rem;
    font-weight: 700;
}

.tool-heading-copy {
    min-width: 0;
}

.tool-kicker {
    margin: 1px 0 5px;
    color: var(--chef-orange-dark);
    font-size: 0.7rem;
    font-weight: 800;
    letter-spacing: 0.14em;
    text-transform: uppercase;
}

.tool-title {
    margin: 0;
    color: var(--chef-ink);
    font-family: Georgia, "Times New Roman", serif;
    font-size: clamp(1.7rem, 3vw, 2.35rem);
    font-weight: 600;
    letter-spacing: -0.035em;
}

.tool-description {
    max-width: 760px;
    margin: 6px 0 0;
    color: var(--chef-muted);
    font-size: 0.91rem;
    line-height: 1.55;
}

.workspace-grid {
    align-items: flex-start;
    gap: 22px;
    max-width: 100%;
}

.workspace-grid > div {
    min-width: 0 !important;
}

.control-panel,
.output-panel {
    border: 1px solid var(--chef-line) !important;
    border-radius: 20px !important;
    background: var(--chef-paper) !important;
    box-shadow: var(--chef-shadow);
}

.control-panel {
    gap: 14px !important;
    padding: 22px !important;
}

.output-panel {
    min-height: 440px;
    gap: 12px !important;
    padding: 22px 26px 26px !important;
}

.panel-label {
    display: flex;
    align-items: center;
    gap: 9px;
    margin: 0 0 2px;
    color: var(--chef-muted);
    font-size: 0.69rem;
    font-weight: 800;
    letter-spacing: 0.14em;
    text-transform: uppercase;
}

.panel-label::before {
    content: "";
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--chef-orange);
    box-shadow: 0 0 0 4px rgba(231, 101, 43, 0.10);
}

.output-panel .panel-label::before {
    background: #3a9b6d;
    box-shadow: 0 0 0 4px rgba(58, 155, 109, 0.11);
}

.section-divider {
    margin: 4px 0 0;
    padding-top: 15px;
    border-top: 1px solid var(--chef-line);
}

.section-title {
    color: #30463b;
    font-size: 0.76rem;
    font-weight: 780;
}

.field-grid,
.triple-grid {
    gap: 12px !important;
    align-items: end;
}

.field-grid > div,
.triple-grid > div {
    min-width: 0 !important;
}

.field-note {
    margin: -6px 2px 0;
    color: #7a857f;
    font-size: 0.74rem;
    line-height: 1.45;
}

.control-panel label > span,
.control-panel .label-wrap span {
    padding: 0 !important;
    border: 0 !important;
    background: transparent !important;
    color: #3b4e45 !important;
    font-size: 0.78rem !important;
    font-weight: 700 !important;
}

.control-panel .label-wrap,
.control-panel .block-label {
    padding: 0 0 7px !important;
    border: 0 !important;
    background: transparent !important;
    color: #3b4e45 !important;
}

.control-panel .html-container,
.output-panel .html-container {
    min-height: 0 !important;
    padding: 0 !important;
    border: 0 !important;
    background: transparent !important;
}

.control-panel .block:has(.panel-label),
.output-panel .block:has(.panel-label),
.control-panel .block:has(.field-note) {
    min-height: 0 !important;
    padding: 0 !important;
    border: 0 !important;
    background: transparent !important;
}

.output-panel .block,
.output-panel .prose,
.output-panel .markdown {
    border-color: transparent !important;
    background: transparent !important;
}

.control-panel table,
.control-panel table tbody,
.control-panel table tr,
.control-panel table td {
    border: 0 !important;
    background: transparent !important;
}

.control-panel table button {
    border: 1px solid var(--chef-line) !important;
    border-radius: 9px !important;
    background: var(--chef-sage) !important;
    color: #3c5146 !important;
    font-size: 0.74rem !important;
}

.control-panel input,
.control-panel textarea,
.control-panel [role="listbox"] {
    background: #fffefa !important;
    border-color: rgba(27, 57, 44, 0.18) !important;
    color: var(--chef-ink) !important;
}

.control-panel input:focus,
.control-panel textarea:focus {
    border-color: var(--chef-forest-2) !important;
    box-shadow: 0 0 0 3px rgba(36, 90, 69, 0.11) !important;
}

.action-row {
    align-items: stretch !important;
    gap: 10px !important;
    margin-top: 3px;
    padding-top: 16px;
    border-top: 1px solid var(--chef-line);
}

.primary-action {
    flex: 1 1 auto !important;
    min-height: 50px !important;
    border: 0 !important;
    border-radius: 12px !important;
    background: var(--chef-orange) !important;
    color: #fff !important;
    font-weight: 800 !important;
    letter-spacing: 0.005em;
    box-shadow: 0 10px 22px rgba(231, 101, 43, 0.23) !important;
    transition: transform 140ms ease, background 140ms ease, box-shadow 140ms ease;
}

.primary-action:hover {
    transform: translateY(-1px);
    background: var(--chef-orange-dark) !important;
    box-shadow: 0 13px 26px rgba(200, 78, 29, 0.27) !important;
}

.secondary-action {
    flex: 0 0 108px !important;
    min-width: 108px !important;
    min-height: 50px !important;
    border: 1px solid var(--chef-line-strong) !important;
    border-radius: 12px !important;
    background: #f8f6ef !important;
    color: #4e5d55 !important;
    font-weight: 700 !important;
}

.secondary-action:hover {
    border-color: rgba(27, 57, 44, 0.32) !important;
    background: var(--chef-sage) !important;
    color: var(--chef-forest) !important;
}

.solo-action {
    width: 100% !important;
    margin-top: 4px;
}

.chef-examples {
    margin-top: 0;
    border: 1px solid var(--chef-line) !important;
    border-radius: 13px !important;
    background: #faf8f2 !important;
}

.chef-examples button {
    border-radius: 9px !important;
    font-size: 0.76rem !important;
}

.chef-accordion {
    border: 1px solid var(--chef-line) !important;
    border-radius: 14px !important;
    background: #faf8f2 !important;
}

.chef-accordion > button {
    color: #354b40 !important;
    font-size: 0.8rem !important;
    font-weight: 750 !important;
}

.recipe-output {
    min-height: 330px;
    padding: 0 !important;
    border: 0 !important;
    background: transparent !important;
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
    display: grid;
    min-height: 320px;
    place-content: center;
    align-items: center;
    justify-content: center;
    padding: 32px;
    border: 1px dashed var(--chef-line-strong);
    border-radius: 16px;
    background: linear-gradient(145deg, rgba(232, 239, 233, 0.62), rgba(255, 254, 250, 0.72));
    text-align: center;
    color: #7b8780;
}

.empty-icon {
    display: grid;
    width: 52px;
    height: 52px;
    place-items: center;
    margin: 0 auto 13px;
    border: 1px solid var(--chef-line);
    border-radius: 16px;
    background: var(--chef-sage);
    color: var(--chef-forest);
    font-family: Georgia, "Times New Roman", serif;
    font-size: 1.1rem;
    font-weight: 700;
}

.empty-state strong {
    display: block;
    margin-bottom: 5px;
    color: #3a4d43;
    font-size: 0.94rem;
}

.empty-state span {
    display: block;
    max-width: 360px;
    font-size: 0.8rem;
    line-height: 1.5;
}

.browse-status {
    min-height: 0 !important;
    margin: 0 0 5px;
    padding: 8px 11px;
    border-radius: 10px;
    background: var(--chef-sage);
    color: var(--chef-muted);
    font-size: 0.78rem;
}

.download-row {
    gap: 12px !important;
    margin-top: 8px;
    padding-top: 16px;
    border-top: 1px solid var(--chef-line);
}

.download-row > div,
.recipe-download {
    min-width: 0 !important;
}

.download-row .file-preview,
.recipe-download .file-preview {
    border: 1px dashed var(--chef-line-strong) !important;
    border-radius: 12px !important;
    background: #faf8f2 !important;
}

#chef-footer {
    margin-top: 26px;
    padding: 20px 4px 0;
    border-top: 1px solid var(--chef-line);
    color: var(--chef-muted);
    font-size: 0.75rem;
    line-height: 1.6;
    text-align: center;
}

@media (max-width: 1040px) {
    #chef-hero {
        grid-template-columns: minmax(0, 1fr) 290px;
        gap: 28px;
        padding: 40px 38px;
    }

    .hero-title {
        font-size: clamp(2.45rem, 6vw, 3.8rem);
    }

    .gradio-container .tab-nav,
    .gradio-container [role="tablist"] {
        display: flex;
        overflow-x: auto;
        justify-content: flex-start;
        scrollbar-width: thin;
    }

    .gradio-container .tab-nav button,
    .gradio-container [role="tablist"] [role="tab"] {
        flex: 0 0 auto;
        min-width: 150px;
    }
}

@media (max-width: 780px) {
    .gradio-container {
        padding: 12px 12px 32px !important;
    }

    #chef-app-bar {
        min-height: auto;
        align-items: flex-start;
        padding: 4px 2px 8px;
    }

    .model-badge {
        display: none;
    }

    #chef-hero {
        display: block;
        padding: 32px 24px 24px;
        border-radius: 21px;
    }

    .hero-title {
        font-size: clamp(2.35rem, 12vw, 3.15rem);
        line-height: 1.02;
    }

    .hero-copy {
        margin-top: 16px;
        font-size: 0.92rem;
    }

    .hero-command {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 7px;
        margin-top: 24px;
        padding: 9px;
        border-radius: 15px;
    }

    .command-kicker,
    .command-copy {
        display: none;
    }

    .command-item,
    .command-item:first-of-type {
        display: block;
        padding: 10px 7px;
        border: 0;
        border-radius: 10px;
        background: rgba(255, 255, 255, 0.06);
        text-align: center;
    }

    .command-index {
        margin: 0 auto 7px;
    }

    .command-title {
        font-size: 0.7rem;
    }

    .gradio-container .tab-nav,
    .gradio-container [role="tablist"] {
        top: 5px;
        margin-bottom: 18px;
    }

    .gradio-container .tab-nav button,
    .gradio-container [role="tablist"] [role="tab"] {
        min-width: 130px;
        min-height: 43px;
        font-size: 0.76rem;
    }

    .tool-heading {
        grid-template-columns: 45px minmax(0, 1fr);
        gap: 12px;
        margin-bottom: 17px;
    }

    .tool-number {
        width: 43px;
        height: 43px;
        border-radius: 13px;
        font-size: 1rem;
    }

    .tool-title {
        font-size: 1.65rem;
    }

    .tool-description {
        font-size: 0.83rem;
    }

    .control-panel,
    .output-panel {
        border-radius: 17px !important;
        padding: 18px !important;
    }

    .workspace-grid {
        flex-direction: column !important;
        flex-wrap: nowrap !important;
    }

    .workspace-grid > div {
        width: 100% !important;
        max-width: 100% !important;
        min-width: 0 !important;
    }

    .output-panel {
        min-height: 300px;
    }

    .recipe-output,
    .empty-state {
        min-height: 250px;
    }

    .action-row {
        flex-wrap: nowrap !important;
    }

    .secondary-action {
        flex-basis: 90px !important;
        min-width: 90px !important;
    }

    .download-row {
        flex-direction: column !important;
    }
}

@media (max-width: 540px) {
    .brand-subtitle,
    #system-note {
        display: none;
    }

    .privacy-badge {
        padding-inline: 10px;
        font-size: 0.69rem;
    }

    #chef-hero {
        padding: 28px 20px 20px;
    }

    .hero-eyebrow {
        margin-bottom: 13px;
        font-size: 0.66rem;
    }

    .hero-command {
        grid-template-columns: repeat(3, minmax(0, 1fr));
        margin-top: 20px;
    }

    .field-grid,
    .triple-grid {
        flex-direction: column !important;
    }

    .action-row {
        flex-direction: column !important;
    }

    .secondary-action {
        flex: 1 1 auto !important;
        width: 100% !important;
    }
}
"""


def build_demo() -> gr.Blocks:
    model_name = Path(GENERATION_MODEL or BASE_MODEL).name
    adapter_label = "fine-tuned LoRA active" if not GENERATION_MODEL else "local checkpoint"
    hero = f"""
    <div id="chef-app-bar">
      <div class="brand-lockup">
        <div class="brand-mark" aria-hidden="true">CA</div>
        <div>
          <div class="brand-name">Chef's Assistant</div>
          <div class="brand-subtitle">Recipe intelligence and menu operations</div>
        </div>
      </div>
      <div class="top-status" aria-label="Application status">
        <span class="model-badge">{html.escape(model_name)}</span>
        <span class="privacy-badge">Local workspace</span>
      </div>
    </div>
    <header id="chef-hero">
      <div class="hero-content">
        <div class="hero-eyebrow">Your digital kitchen desk</div>
        <h1 class="hero-title">From first idea<br>to final service.</h1>
        <p class="hero-copy">
          Search the full recipe library, develop dishes around real constraints,
          and assemble practical menus in one focused chef workspace.
        </p>
      </div>
      <aside class="hero-command" aria-label="Chef workflow">
        <div class="command-kicker">One connected workflow</div>
        <div class="command-item">
          <div class="command-index">01</div>
          <div><div class="command-title">Discover</div><div class="command-copy">Search and browse grounded recipes.</div></div>
        </div>
        <div class="command-item">
          <div class="command-index">02</div>
          <div><div class="command-title">Create</div><div class="command-copy">Develop a recipe with local AI.</div></div>
        </div>
        <div class="command-item">
          <div class="command-index">03</div>
          <div><div class="command-title">Plan</div><div class="command-copy">Build service-ready menus and lists.</div></div>
        </div>
      </aside>
    </header>
    """
    system_note = (
        f"Dataset: {html.escape(DATASET_PATH.name)} &middot; {html.escape(adapter_label)} &middot; "
        f"adapter scale {LORA_SCALE:g} &middot; AI loads only when generation is requested"
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
            with gr.Tab("01 · Find recipes"):
                gr.HTML(
                    """
                    <section class="tool-heading">
                      <div class="tool-number">01</div>
                      <div class="tool-heading-copy">
                        <div class="tool-kicker">Recipe search</div>
                        <h2 class="tool-title">Find the right recipe, faster.</h2>
                        <p class="tool-description">Describe the dish, ingredient, cuisine, or service constraint. The complete indexed collection returns grounded matches in seconds.</p>
                      </div>
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
                            with gr.Row(elem_classes="field-grid"):
                                result_count = gr.Slider(
                                    1, 10, value=5, step=1, label="Matches"
                                )
                                search_minutes = gr.Slider(
                                    0,
                                    240,
                                    value=0,
                                    step=15,
                                    label="Max time",
                                )
                            gr.HTML(
                                '<p class="field-note">Set max time to 0 when there is no service-time limit.</p>',
                                padding=False,
                            )
                            search_dietary = gr.Textbox(
                                label="Dietary and allergen filters",
                                placeholder="Halal, vegetarian, no peanuts",
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
                            with gr.Row(elem_classes="action-row"):
                                search_button = gr.Button(
                                    "Find matching recipes →",
                                    variant="primary",
                                    elem_classes="primary-action",
                                )
                                search_cancel = gr.Button(
                                    "Cancel",
                                    variant="secondary",
                                    elem_classes="secondary-action",
                                )
                    with gr.Column(scale=7, min_width=380):
                        with gr.Group(elem_classes="output-panel"):
                            gr.HTML('<div class="panel-label">Recommended matches</div>', padding=False)
                            search_output = gr.Markdown(
                                "<div class='empty-state'><div class='empty-icon'>01</div><strong>Ready to search</strong><span>Your best recipe matches will appear here, ranked around the brief and filters.</span></div>",
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

            with gr.Tab("02 · Surprise me"):
                gr.HTML(
                    """
                    <section class="tool-heading">
                      <div class="tool-number">02</div>
                      <div class="tool-heading-copy">
                        <div class="tool-kicker">Kitchen inspiration</div>
                        <h2 class="tool-title">Let the kitchen choose.</h2>
                        <p class="tool-description">Choose a collection and draw one complete recipe at random—ideal when the team needs a fresh direction without another meeting.</p>
                      </div>
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
                                "Pick a recipe →",
                                variant="primary",
                                elem_classes=["primary-action", "solo-action"],
                            )
                    with gr.Column(scale=7, min_width=380):
                        with gr.Group(elem_classes="output-panel"):
                            gr.HTML('<div class="panel-label">Today\'s discovery</div>', padding=False)
                            random_output = gr.Markdown(
                                "<div class='empty-state'><div class='empty-icon'>02</div><strong>Waiting for a category</strong><span>Set the collection, then let chance give the kitchen its next idea.</span></div>",
                                elem_classes="recipe-output",
                            )
                random_button.click(random_recipe, inputs=category, outputs=random_output)

            with gr.Tab("03 · Browse library"):
                gr.HTML(
                    """
                    <section class="tool-heading">
                      <div class="tool-number">03</div>
                      <div class="tool-heading-copy">
                        <div class="tool-kicker">Recipe library</div>
                        <h2 class="tool-title">Explore the collection your way.</h2>
                        <p class="tool-description">Browse alphabetically, narrow the category, and move through the indexed catalog one focused page at a time.</p>
                      </div>
                    </section>
                    """,
                    padding=False,
                )
                with gr.Row(elem_classes="workspace-grid"):
                    with gr.Column(scale=4, min_width=310):
                        with gr.Group(elem_classes="control-panel"):
                            gr.HTML('<div class="panel-label">Library filters</div>', padding=False)
                            with gr.Row(elem_classes="field-grid"):
                                browse_letter = gr.Dropdown(
                                    ["All"] + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
                                    value="All",
                                    label="Starts with",
                                )
                                browse_category = gr.Dropdown(
                                    ["All", "Chinese", "Western"],
                                    value="All",
                                    label="Category",
                                )
                            browse_page = gr.Number(value=1, precision=0, label="Page number")
                            browse_button = gr.Button(
                                "Browse recipes →",
                                variant="primary",
                                elem_classes=["primary-action", "solo-action"],
                            )
                    with gr.Column(scale=7, min_width=380):
                        with gr.Group(elem_classes="output-panel"):
                            gr.HTML('<div class="panel-label">Recipe collection</div>', padding=False)
                            browse_status = gr.Markdown(elem_classes="browse-status")
                            browse_output = gr.Markdown(
                                "<div class='empty-state'><div class='empty-icon'>03</div><strong>The library is ready</strong><span>Choose a letter, category, and page to begin browsing.</span></div>",
                                elem_classes="recipe-output",
                            )
                browse_button.click(
                    browse_recipes,
                    inputs=[browse_letter, browse_category, browse_page],
                    outputs=[browse_output, browse_status],
                )

            with gr.Tab("04 · Create recipe"):
                gr.HTML(
                    """
                    <section class="tool-heading">
                      <div class="tool-number">04</div>
                      <div class="tool-heading-copy">
                        <div class="tool-kicker">AI recipe studio</div>
                        <h2 class="tool-title">Turn an idea into a complete dish.</h2>
                        <p class="tool-description">Write the culinary brief, set the covers and restrictions, then let the local model build and audit a structured recipe.</p>
                      </div>
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
                            with gr.Row(elem_classes="action-row"):
                                generate_button = gr.Button(
                                    "Create my recipe →",
                                    variant="primary",
                                    elem_classes="primary-action",
                                )
                                generation_cancel = gr.Button(
                                    "Cancel",
                                    variant="secondary",
                                    elem_classes="secondary-action",
                                )
                    with gr.Column(scale=7, min_width=380):
                        with gr.Group(elem_classes="output-panel"):
                            gr.HTML('<div class="panel-label">Your generated recipe</div>', padding=False)
                            generation_output = gr.Markdown(
                                "<div class='empty-state'><div class='empty-icon'>04</div><strong>Your recipe studio is ready</strong><span>Submit a clear brief to generate ingredients, method, timings, and quality checks.</span></div>",
                                elem_classes="recipe-output",
                            )
                            recipe_download = gr.File(
                                label="Download recipe",
                                interactive=False,
                                elem_classes="recipe-download",
                            )
                generation_event = generate_button.click(
                    generate_recipe_ui,
                    inputs=[generation_request, servings, dietary_notes],
                    outputs=[generation_output, recipe_download],
                )
                generation_cancel.click(fn=None, cancels=[generation_event])

            with gr.Tab("05 · Plan menu"):
                gr.HTML(
                    """
                    <section class="tool-heading">
                      <div class="tool-number">05</div>
                      <div class="tool-heading-copy">
                        <div class="tool-kicker">Menu operations</div>
                        <h2 class="tool-title">Build a practical plan for every service.</h2>
                        <p class="tool-description">Define the schedule, covers, budget, restrictions, and pantry. The planner selects grounded recipes and consolidates the shopping list.</p>
                      </div>
                    </section>
                    """,
                    padding=False,
                )
                with gr.Row(elem_classes="workspace-grid"):
                    with gr.Column(scale=5, min_width=330):
                        with gr.Group(elem_classes="control-panel"):
                            gr.HTML('<div class="panel-label">Planning brief</div>', padding=False)
                            with gr.Row(elem_classes="triple-grid"):
                                menu_days = gr.Slider(1, 7, value=3, step=1, label="Days")
                                menu_meals = gr.Slider(
                                    1, 3, value=2, step=1, label="Meals per day"
                                )
                                menu_servings = gr.Slider(
                                    1, 12, value=4, step=1, label="Covers"
                                )
                            with gr.Row(elem_classes="field-grid"):
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
                            with gr.Accordion(
                                "Pantry, stock and planning rules",
                                open=False,
                                elem_classes="chef-accordion",
                            ):
                                with gr.Row(elem_classes="field-grid"):
                                    menu_available = gr.Textbox(
                                        label="Ingredients already available",
                                        placeholder="Chicken, rice, carrots, cabbage",
                                    )
                                    menu_pantry = gr.Textbox(
                                        label="Exclude from shopping list",
                                        placeholder="Salt, pepper, cooking oil, soy sauce",
                                    )
                                with gr.Row(elem_classes="field-grid"):
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
                            with gr.Row(elem_classes="action-row"):
                                menu_button = gr.Button(
                                    "Create menu plan →",
                                    variant="primary",
                                    elem_classes="primary-action",
                                )
                                menu_cancel = gr.Button(
                                    "Cancel",
                                    variant="secondary",
                                    elem_classes="secondary-action",
                                )
                    with gr.Column(scale=7, min_width=380):
                        with gr.Group(elem_classes="output-panel"):
                            gr.HTML('<div class="panel-label">Menu and shopping list</div>', padding=False)
                            menu_output = gr.Markdown(
                                "<div class='empty-state'><div class='empty-icon'>05</div><strong>Ready for the planning brief</strong><span>Your service schedule, selected recipes, leftovers, and consolidated shopping list will appear here.</span></div>",
                                elem_classes="recipe-output",
                            )
                            with gr.Row(elem_classes="download-row"):
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
