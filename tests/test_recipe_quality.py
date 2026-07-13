from recipe_quality import (
    dietary_blocked_terms,
    normalize_recipe,
    quality_score,
    validate_recipe,
)


def valid_recipe() -> dict:
    return {
        "name": "Malaysian Ginger Chicken",
        "description": "A Malaysian chicken dish with ginger, rice, and aromatic spices.",
        "servings": 4,
        "minutes": 45,
        "ingredients": [
            "500 g chicken thighs",
            "2 tbsp cooking oil",
            "20 g fresh ginger",
            "2 cups cooked rice",
        ],
        "steps": [
            "Heat the cooking oil and fry the ginger until fragrant.",
            "Add the chicken thighs and cook thoroughly, then serve with rice.",
        ],
    }


def test_valid_recipe_passes_mandatory_checks() -> None:
    issues = validate_recipe(
        valid_recipe(),
        cuisines=["malaysian"],
        dietary_notes="halal, no peanuts",
        servings=4,
    )
    assert not [issue for issue in issues if issue.severity == "error"]
    assert quality_score(issues) >= 90


def test_allergen_and_serving_violations_are_reported() -> None:
    recipe = valid_recipe()
    recipe["servings"] = 2
    recipe["ingredients"].append("2 tbsp peanut butter")
    issues = validate_recipe(
        recipe,
        cuisines=["malaysian"],
        dietary_notes="halal, no peanuts",
        servings=4,
    )
    codes = {issue.code for issue in issues}
    assert "servings_mismatch" in codes
    assert "dietary_violation" in codes


def test_missing_quantities_are_reported() -> None:
    recipe = valid_recipe()
    recipe["ingredients"] = ["chicken", "ginger", "rice"]
    assert "missing_quantities" in {
        issue.code for issue in validate_recipe(recipe, servings=4)
    }


def test_dictionary_ingredients_are_normalized() -> None:
    recipe = valid_recipe()
    recipe["ingredients"] = [
        {"quantity": "500", "unit": "g", "name": "chicken"},
        {"quantity": "1", "unit": "tbsp", "name": "oil"},
        {"quantity": "2", "unit": "cloves", "name": "garlic"},
    ]
    normalized = normalize_recipe(recipe)
    assert normalized["ingredients"][0] == "500 g chicken"


def test_separate_quantity_fragments_are_joined() -> None:
    recipe = valid_recipe()
    recipe["ingredients"] = ["2 cups", "uncooked rice", "1 tbsp", "soy sauce"]
    normalized = normalize_recipe(recipe)
    assert normalized["ingredients"][:2] == [
        "2 cups uncooked rice",
        "1 tbsp soy sauce",
    ]


def test_common_step_ingredient_is_added_to_the_list() -> None:
    recipe = valid_recipe()
    recipe["steps"].append("Season with salt and garnish with scallions.")
    normalized = normalize_recipe(recipe)
    assert "as needed salt" in normalized["ingredients"]
    assert "as needed scallions" in normalized["ingredients"]


def test_fragmented_ingredient_description_is_joined() -> None:
    recipe = valid_recipe()
    recipe["ingredients"] = ["2 pounds boneless", "skinless", "chicken breasts"]
    normalized = normalize_recipe(recipe)
    assert normalized["ingredients"][0] == "2 pounds boneless skinless chicken breasts"


def test_szechuan_is_accepted_as_chinese_cuisine() -> None:
    recipe = valid_recipe()
    recipe["name"] = "Vegan Szechuan Tofu"
    recipe["description"] = "A Szechuan-inspired tofu and vegetable stir-fry."
    assert "cuisine_mismatch" not in {
        issue.code for issue in validate_recipe(recipe, cuisines=["chinese"], servings=4)
    }


def test_normalization_is_idempotent() -> None:
    once = normalize_recipe(valid_recipe())
    twice = normalize_recipe(once)
    assert twice == once


def test_dietary_aliases_cover_common_chef_requests() -> None:
    assert {"peanut", "groundnut"} <= dietary_blocked_terms("peanut allergy")
    assert "pork" in dietary_blocked_terms("halal")
    assert "milk" in dietary_blocked_terms("vegan")
