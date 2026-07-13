from pathlib import Path

import pandas as pd

from recipe_store import RecipeStore


def write_dataset(path: Path) -> None:
    frame = pd.DataFrame(
        [
            {
                "id": 1,
                "name": "Chinese Chicken Fried Rice",
                "minutes": 25,
                "tags": "['chinese', 'main-dish']",
                "nutrition": "[1, 2, 3]",
                "n_steps": 2,
                "steps": "['cook chicken', 'fry with rice']",
                "description": "A quick Chinese dinner.",
                "ingredients": "['chicken breast', 'rice', 'soy sauce']",
                "n_ingredients": 3,
            },
            {
                "id": 2,
                "name": "Western Vegetable Pasta",
                "minutes": 40,
                "tags": "['italian', 'vegetarian']",
                "nutrition": "[1, 2, 3]",
                "n_steps": 2,
                "steps": "['boil pasta', 'add vegetables']",
                "description": "A simple Italian-style pasta.",
                "ingredients": "['pasta', 'tomato', 'zucchini']",
                "n_ingredients": 3,
            },
            {
                "id": 3,
                "name": "Thai Coconut Soup",
                "minutes": 55,
                "tags": "['thai', 'soup']",
                "nutrition": "[1, 2, 3]",
                "n_steps": 2,
                "steps": "['simmer broth', 'add coconut milk']",
                "description": "A Thai coconut soup.",
                "ingredients": "['coconut milk', 'mushrooms', 'lime']",
                "n_ingredients": 3,
            },
        ]
    )
    frame.to_csv(path, index=False, encoding="ISO-8859-1")


def test_full_text_search_browse_and_random(tmp_path: Path) -> None:
    dataset = tmp_path / "recipes.csv"
    index = tmp_path / "recipes.sqlite3"
    write_dataset(dataset)
    store = RecipeStore(dataset, index)
    stats = store.ensure_index()
    assert stats.row_count == 3
    assert store.count() == 3

    matches = store.search("Chinese chicken rice", limit=5)
    assert matches.iloc[0]["name"] == "Chinese Chicken Fried Rice"

    page, total, current = store.browse(letter="W", category="Western")
    assert total == 1
    assert current == 1
    assert page.iloc[0]["name"] == "Western Vegetable Pasta"

    random_recipe = store.random_recipe("Chinese")
    assert random_recipe is not None
    assert random_recipe["name"] == "Chinese Chicken Fried Rice"


def test_index_rebuilds_when_dataset_changes(tmp_path: Path) -> None:
    dataset = tmp_path / "recipes.csv"
    index = tmp_path / "recipes.sqlite3"
    write_dataset(dataset)
    store = RecipeStore(dataset, index)
    store.ensure_index()

    frame = pd.read_csv(dataset, encoding="ISO-8859-1")
    frame.loc[len(frame)] = frame.iloc[0].to_dict() | {"id": 4, "name": "New Dish"}
    frame.to_csv(dataset, index=False, encoding="ISO-8859-1")

    assert store.ensure_index().row_count == 4
