"""Deterministic quality and dietary checks for generated recipes."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class QualityIssue:
    code: str
    message: str
    severity: str = "error"


DIETARY_BLOCKS: dict[str, set[str]] = {
    "halal": {
        "bacon",
        "beer",
        "brandy",
        "ham",
        "lard",
        "pork",
        "rum",
        "wine",
    },
    "vegetarian": {
        "anchovy",
        "beef",
        "chicken",
        "duck",
        "fish",
        "gelatin",
        "lamb",
        "pork",
        "prawn",
        "shrimp",
        "turkey",
    },
    "vegan": {
        "anchovy",
        "beef",
        "butter",
        "cheese",
        "chicken",
        "cream",
        "duck",
        "egg",
        "fish",
        "gelatin",
        "honey",
        "lamb",
        "milk",
        "pork",
        "prawn",
        "shrimp",
        "turkey",
        "yogurt",
    },
    "gluten-free": {"barley", "rye", "seitan", "wheat"},
}

ALLERGEN_ALIASES: dict[str, set[str]] = {
    "peanut": {"groundnut", "peanut"},
    "tree nut": {"almond", "cashew", "hazelnut", "macadamia", "pecan", "pistachio", "walnut"},
    "shellfish": {"crab", "lobster", "prawn", "shrimp"},
    "dairy": {"butter", "cheese", "cream", "milk", "whey", "yogurt"},
    "egg": {"egg", "mayonnaise"},
    "soy": {"miso", "soy", "tempeh", "tofu"},
}

QUANTITY_PATTERN = re.compile(
    r"(?:\b\d+(?:[./]\d+)?\s*(?:g|kg|ml|l|tsp|tbsp|teaspoons?|tablespoons?|cups?|ounces?|oz|pounds?|lb|cloves?|cans?|pieces?|slices?|stalks?|sprigs?)\b|"
    r"\b\d+(?:[./]\d+)?\b|\b(?:a\s+)?pinch\b|\bto taste\b|\bas needed\b)",
    re.IGNORECASE,
)

QUANTITY_ONLY_PATTERN = re.compile(
    r"^(?:\d+(?:[./]\d+)?|a|one)\s*(?:g|kg|ml|l|tsp|tbsp|teaspoons?|"
    r"tablespoons?|cups?|ounces?|oz|pounds?|lb|cloves?|cans?|pieces?|slices?|"
    r"stalks?|sprigs?|blocks?|inches?|small|medium|large)?$",
    re.IGNORECASE,
)

FLEXIBLE_AMOUNT_TERMS = {
    "chili paste",
    "fish sauce",
    "hot sauce",
    "ketchup",
    "oil",
    "oyster sauce",
    "pepper",
    "salt",
    "soy sauce",
    "sugar",
}

INCOMPLETE_INGREDIENT_ENDINGS = {
    "boneless",
    "chopped",
    "diced",
    "fresh",
    "grated",
    "ground",
    "minced",
    "skinless",
    "sliced",
}

QUANTITY_FRAGMENT_WORDS = {
    "block",
    "blocks",
    "can",
    "cans",
    "clove",
    "cloves",
    "cup",
    "cups",
    "g",
    "inch",
    "inches",
    "kg",
    "l",
    "large",
    "lb",
    "medium",
    "ml",
    "ounce",
    "ounces",
    "oz",
    "piece",
    "pieces",
    "pound",
    "pounds",
    "slice",
    "slices",
    "small",
    "sprig",
    "sprigs",
    "stalk",
    "stalks",
    "tablespoon",
    "tablespoons",
    "tbsp",
    "teaspoon",
    "teaspoons",
    "tsp",
}

CUISINE_ALIASES: dict[str, set[str]] = {
    "chinese": {"cantonese", "chinese", "hunan", "sichuan", "szechuan"},
    "indian": {"bengali", "goan", "indian", "punjabi"},
    "italian": {"italian", "sicilian", "tuscan"},
    "japanese": {"japanese", "teriyaki"},
    "korean": {"korean"},
    "malaysian": {"malay", "malaysian", "nasi"},
    "mexican": {"mexican", "tex-mex"},
    "thai": {"thai"},
}

COMMON_STEP_INGREDIENTS = {
    "butter",
    "coriander",
    "carrots",
    "chicken",
    "garlic",
    "ginger",
    "lemon",
    "lime",
    "oil",
    "onion",
    "pepper",
    "salt",
    "scallions",
    "shallots",
    "stock",
    "soy sauce",
    "sesame oil",
    "tofu",
    "water",
}

INGREDIENT_STOPWORDS = {
    "about",
    "chopped",
    "diced",
    "fresh",
    "finely",
    "ground",
    "large",
    "medium",
    "minced",
    "optional",
    "small",
    "sliced",
    "taste",
}


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _quantity_fragment(value: str) -> bool:
    words = re.findall(r"[a-z]+", value.lower())
    return bool(re.search(r"\d", value)) and bool(words) and all(
        word in QUANTITY_FRAGMENT_WORDS for word in words
    )


def dietary_blocked_terms(notes: str) -> set[str]:
    lowered = _text(notes).lower()
    blocked: set[str] = set()
    if "halal" in lowered:
        blocked.update(DIETARY_BLOCKS["halal"])
    if "vegetarian" in lowered:
        blocked.update(DIETARY_BLOCKS["vegetarian"])
    if "vegan" in lowered:
        blocked.update(DIETARY_BLOCKS["vegan"])
    if "gluten-free" in lowered or "gluten free" in lowered:
        blocked.update(DIETARY_BLOCKS["gluten-free"])
    for allergen, aliases in ALLERGEN_ALIASES.items():
        if (
            f"no {allergen}" in lowered
            or f"{allergen} allergy" in lowered
            or f"{allergen}-free" in lowered
            or f"{allergen} free" in lowered
        ):
            blocked.update(aliases)
    return blocked


def find_blocked_terms(text: str, notes: str) -> set[str]:
    lowered = _text(text).lower()
    return {
        term
        for term in dietary_blocked_terms(notes)
        if re.search(rf"\b{re.escape(term)}s?\b", lowered)
    }


def normalize_recipe(recipe: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(recipe, dict):
        return {}
    normalized = dict(recipe)
    ingredient_fragments: list[str] = []
    for item in recipe.get("ingredients", []):
        if isinstance(item, dict):
            quantity = _text(item.get("quantity"))
            unit = _text(item.get("unit"))
            name = _text(item.get("name"))
            item_text = " ".join(value for value in (quantity, unit, name) if value)
        else:
            item_text = _text(item)
        if item_text:
            ingredient_fragments.append(item_text)

    steps: list[str] = []
    for item in recipe.get("steps", []):
        if isinstance(item, dict):
            item_text = _text(item.get("step") or item.get("instruction"))
        else:
            item_text = _text(item)
        if item_text:
            steps.append(item_text)

    ingredients: list[str] = []
    pending_quantity = ""
    for fragment in ingredient_fragments:
        if QUANTITY_ONLY_PATTERN.fullmatch(fragment) or _quantity_fragment(fragment):
            pending_quantity = fragment
            continue
        if pending_quantity:
            if not QUANTITY_PATTERN.search(fragment):
                fragment = f"{pending_quantity} {fragment}"
            pending_quantity = ""
        if (
            not QUANTITY_PATTERN.search(fragment)
            and ingredients
            and any(
                ingredients[-1].lower().endswith(ending)
                for ending in INCOMPLETE_INGREDIENT_ENDINGS
            )
        ):
            ingredients[-1] = f"{ingredients[-1]} {fragment}"
            continue
        if not QUANTITY_PATTERN.search(fragment) and any(
            term in fragment.lower() for term in FLEXIBLE_AMOUNT_TERMS
        ):
            fragment = f"as needed {fragment}"
        ingredients.append(fragment)
    # A quantity without an ingredient is unsafe and not useful. Step-based recovery
    # below can restore named ingredients, while this orphan fragment is discarded.

    ingredient_text = " ".join(ingredients).lower()
    step_text = " ".join(steps).lower()
    for ingredient in sorted(COMMON_STEP_INGREDIENTS):
        if (
            re.search(rf"\b{re.escape(ingredient)}s?\b", step_text)
            and not re.search(rf"\b{re.escape(ingredient)}s?\b", ingredient_text)
        ):
            ingredients.append(f"as needed {ingredient}")

    normalized["name"] = _text(recipe.get("name"))
    normalized["description"] = _text(recipe.get("description"))
    normalized["ingredients"] = _deduplicate(ingredients)
    normalized["steps"] = _deduplicate(steps)
    try:
        normalized["minutes"] = max(1, int(float(recipe.get("minutes", 0))))
    except (TypeError, ValueError):
        normalized["minutes"] = 0
    try:
        normalized["servings"] = max(1, int(float(recipe.get("servings", 0))))
    except (TypeError, ValueError):
        normalized["servings"] = 0
    return normalized


def _deduplicate(items: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        cleaned = _text(item)
        key = re.sub(r"\s+", " ", cleaned.lower())
        if cleaned and key not in seen:
            result.append(cleaned)
            seen.add(key)
    return result


def _ingredient_tokens(ingredient: str) -> set[str]:
    cleaned = QUANTITY_PATTERN.sub(" ", ingredient.lower())
    return {
        token
        for token in re.findall(r"[a-z]{3,}", cleaned)
        if token not in INGREDIENT_STOPWORDS
    }


def validate_recipe(
    recipe: dict[str, Any] | None,
    *,
    cuisines: Iterable[str] = (),
    dietary_notes: str = "",
    servings: int | None = None,
) -> list[QualityIssue]:
    normalized = normalize_recipe(recipe)
    issues: list[QualityIssue] = []
    if not normalized:
        return [QualityIssue("invalid_json", "The model did not return a recipe object.")]

    if len(normalized.get("name", "").split()) < 2:
        issues.append(QualityIssue("weak_name", "Use a meaningful recipe name."))
    if len(normalized.get("description", "")) < 20:
        issues.append(QualityIssue("missing_description", "Add a useful recipe description."))
    if normalized.get("minutes", 0) <= 0:
        issues.append(QualityIssue("invalid_minutes", "Cooking time must be a positive number."))
    if servings and normalized.get("servings") != int(servings):
        issues.append(QualityIssue("servings_mismatch", f"Set servings to {int(servings)}."))

    ingredients = normalized.get("ingredients", [])
    steps = normalized.get("steps", [])
    if len(ingredients) < 3:
        issues.append(QualityIssue("too_few_ingredients", "Provide at least three ingredients."))
    if len(steps) < 2:
        issues.append(QualityIssue("too_few_steps", "Provide at least two cooking steps."))

    missing_quantities = [item for item in ingredients if not QUANTITY_PATTERN.search(item)]
    if missing_quantities:
        preview = ", ".join(missing_quantities[:4])
        issues.append(
            QualityIssue(
                "missing_quantities",
                f"Add practical quantities to: {preview}.",
            )
        )

    identity = f"{normalized.get('name', '')} {normalized.get('description', '')}".lower()
    missing_cuisines = []
    for cuisine in cuisines:
        aliases = CUISINE_ALIASES.get(cuisine.lower(), {cuisine.lower()})
        if not any(alias in identity for alias in aliases):
            missing_cuisines.append(cuisine)
    if missing_cuisines:
        issues.append(
            QualityIssue(
                "cuisine_mismatch",
                f"Explicitly preserve cuisine: {', '.join(missing_cuisines)}.",
            )
        )

    complete_text = " ".join(
        [identity, *ingredients, *steps]
    )
    blocked = find_blocked_terms(complete_text, dietary_notes)
    if blocked:
        issues.append(
            QualityIssue(
                "dietary_violation",
                f"Remove blocked ingredients: {', '.join(sorted(blocked))}.",
            )
        )

    step_text = " ".join(steps).lower()
    unused: list[str] = []
    for ingredient in ingredients:
        tokens = _ingredient_tokens(ingredient)
        if tokens and not any(token in step_text for token in tokens):
            unused.append(ingredient)
    if len(unused) > max(2, len(ingredients) // 3):
        issues.append(
            QualityIssue(
                "ingredient_step_mismatch",
                "Use the listed main ingredients in the cooking steps.",
                severity="warning",
            )
        )
    return issues


def quality_score(issues: Iterable[QualityIssue]) -> int:
    score = 100
    for issue in issues:
        score -= 15 if issue.severity == "error" else 7
    return max(0, score)
