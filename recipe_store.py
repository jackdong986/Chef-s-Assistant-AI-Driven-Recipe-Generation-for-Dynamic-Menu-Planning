"""SQLite-backed access to the complete recipe dataset."""

from __future__ import annotations

import os
import re
import sqlite3
import threading
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd


INDEX_VERSION = "1"
CSV_COLUMNS = [
    "id",
    "name",
    "minutes",
    "tags",
    "nutrition",
    "n_steps",
    "steps",
    "description",
    "ingredients",
    "n_ingredients",
]

CHINESE_KEYWORDS = (
    "baozi",
    "char siu",
    "chinese",
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
    "american",
    "british",
    "burger",
    "caesar salad",
    "european",
    "fish and chips",
    "french",
    "italian",
    "lasagna",
    "pasta",
    "pizza",
    "roast",
    "sandwich",
    "steak",
)


@dataclass(frozen=True)
class IndexStats:
    row_count: int
    dataset_size: int
    dataset_mtime_ns: int
    index_path: Path


def _text(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def _integer(value: object, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _category(search_text: str) -> str:
    lowered = search_text.lower()
    if any(keyword in lowered for keyword in CHINESE_KEYWORDS):
        return "Chinese"
    if any(keyword in lowered for keyword in WESTERN_KEYWORDS):
        return "Western"
    return "Other"


def _first_letter(name: str) -> str:
    cleaned = re.sub(r"^(?:\d+\s*|the\s+|in\s+)", "", name.lower()).strip()
    return cleaned[:1].upper() if cleaned[:1].isalpha() else "#"


def _fts_query(query: str) -> str:
    ignored = {"a", "an", "and", "for", "in", "of", "the", "to", "with"}
    tokens = [
        token
        for token in re.findall(r"[a-z0-9]+", query.lower())
        if len(token) > 1 and token not in ignored
    ]
    # OR keeps natural-language searches useful when not every word occurs in a recipe.
    return " OR ".join(f'"{token}"*' for token in tokens[:20])


class RecipeStore:
    """Build and query a local full-text index for the recipe CSV."""

    def __init__(self, dataset_path: Path, index_path: Path):
        self.dataset_path = Path(dataset_path).expanduser().resolve()
        self.index_path = Path(index_path).expanduser().resolve()
        self._build_lock = threading.Lock()

    def _dataset_signature(self) -> tuple[int, int]:
        stat = self.dataset_path.stat()
        return stat.st_size, stat.st_mtime_ns

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.index_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only = ON")
        return connection

    def _metadata(self) -> dict[str, str]:
        if not self.index_path.is_file():
            return {}
        try:
            with closing(self._connect()) as connection:
                return {
                    row["key"]: row["value"]
                    for row in connection.execute("SELECT key, value FROM metadata")
                }
        except (sqlite3.DatabaseError, OSError):
            return {}

    def is_current(self) -> bool:
        if not self.dataset_path.is_file():
            return False
        metadata = self._metadata()
        size, mtime_ns = self._dataset_signature()
        return (
            metadata.get("index_version") == INDEX_VERSION
            and metadata.get("dataset_path") == str(self.dataset_path)
            and metadata.get("dataset_size") == str(size)
            and metadata.get("dataset_mtime_ns") == str(mtime_ns)
        )

    def ensure_index(
        self,
        *,
        force: bool = False,
        progress: Callable[[int], None] | None = None,
    ) -> IndexStats:
        if not self.dataset_path.is_file():
            raise FileNotFoundError(f"Recipe dataset not found: {self.dataset_path}")
        with self._build_lock:
            if force or not self.is_current():
                self._build_index(progress=progress)
        metadata = self._metadata()
        size, mtime_ns = self._dataset_signature()
        return IndexStats(
            row_count=int(metadata.get("row_count", "0")),
            dataset_size=size,
            dataset_mtime_ns=mtime_ns,
            index_path=self.index_path,
        )

    def _build_index(self, progress: Callable[[int], None] | None = None) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.index_path.with_suffix(self.index_path.suffix + ".tmp")
        if temporary_path.exists():
            temporary_path.unlink()

        connection = sqlite3.connect(temporary_path)
        try:
            connection.executescript(
                """
                PRAGMA journal_mode = OFF;
                PRAGMA synchronous = OFF;
                PRAGMA temp_store = MEMORY;
                CREATE TABLE recipes (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    minutes INTEGER NOT NULL DEFAULT 0,
                    tags TEXT NOT NULL DEFAULT '',
                    nutrition TEXT NOT NULL DEFAULT '',
                    n_steps INTEGER NOT NULL DEFAULT 0,
                    steps TEXT NOT NULL DEFAULT '',
                    description TEXT NOT NULL DEFAULT '',
                    ingredients TEXT NOT NULL DEFAULT '',
                    n_ingredients INTEGER NOT NULL DEFAULT 0,
                    search_text TEXT NOT NULL DEFAULT '',
                    category TEXT NOT NULL DEFAULT 'Other',
                    first_letter TEXT NOT NULL DEFAULT '#'
                );
                CREATE INDEX recipes_category_idx ON recipes(category);
                CREATE INDEX recipes_letter_idx ON recipes(first_letter);
                CREATE INDEX recipes_minutes_idx ON recipes(minutes);
                CREATE VIRTUAL TABLE recipes_fts USING fts5(
                    name,
                    description,
                    ingredients,
                    tags,
                    content='recipes',
                    content_rowid='id',
                    tokenize='porter unicode61 remove_diacritics 2'
                );
                CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                """
            )

            inserted = 0
            for chunk in pd.read_csv(
                self.dataset_path,
                usecols=CSV_COLUMNS,
                encoding="ISO-8859-1",
                low_memory=False,
                chunksize=5000,
            ):
                records: list[tuple[object, ...]] = []
                for row in chunk.to_dict(orient="records"):
                    recipe_id = _integer(row.get("id"), -1)
                    name = _text(row.get("name"))
                    if recipe_id < 0 or not name:
                        continue
                    description = _text(row.get("description"))
                    ingredients = _text(row.get("ingredients"))
                    tags = _text(row.get("tags"))
                    search_text = f"{name}. {description}. {ingredients}. {tags}"
                    records.append(
                        (
                            recipe_id,
                            name,
                            _integer(row.get("minutes")),
                            tags,
                            _text(row.get("nutrition")),
                            _integer(row.get("n_steps")),
                            _text(row.get("steps")),
                            description,
                            ingredients,
                            _integer(row.get("n_ingredients")),
                            search_text[:5000],
                            _category(search_text),
                            _first_letter(name),
                        )
                    )
                connection.executemany(
                    """
                    INSERT OR IGNORE INTO recipes (
                        id, name, minutes, tags, nutrition, n_steps, steps,
                        description, ingredients, n_ingredients, search_text,
                        category, first_letter
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    records,
                )
                inserted += len(records)
                if progress:
                    progress(inserted)

            connection.execute("INSERT INTO recipes_fts(recipes_fts) VALUES('rebuild')")
            row_count = int(connection.execute("SELECT COUNT(*) FROM recipes").fetchone()[0])
            size, mtime_ns = self._dataset_signature()
            connection.executemany(
                "INSERT INTO metadata(key, value) VALUES (?, ?)",
                [
                    ("index_version", INDEX_VERSION),
                    ("dataset_path", str(self.dataset_path)),
                    ("dataset_size", str(size)),
                    ("dataset_mtime_ns", str(mtime_ns)),
                    ("row_count", str(row_count)),
                ],
            )
            connection.commit()
        finally:
            connection.close()

        os.replace(temporary_path, self.index_path)

    @staticmethod
    def _rows_to_frame(rows: Iterable[sqlite3.Row]) -> pd.DataFrame:
        records = [dict(row) for row in rows]
        return pd.DataFrame.from_records(records)

    def search(
        self,
        query: str,
        *,
        limit: int = 100,
        category: str = "All",
        max_minutes: int | None = None,
    ) -> pd.DataFrame:
        self.ensure_index()
        fts_query = _fts_query(query)
        if not fts_query:
            return pd.DataFrame()
        clauses = ["recipes_fts MATCH ?"]
        parameters: list[object] = [fts_query]
        if category != "All":
            clauses.append("r.category = ?")
            parameters.append(category)
        if max_minutes and max_minutes > 0:
            clauses.append("r.minutes BETWEEN 1 AND ?")
            parameters.append(int(max_minutes))
        parameters.append(max(1, min(int(limit), 1000)))
        sql = f"""
            SELECT r.*, bm25(recipes_fts, 7.0, 2.0, 3.0, 1.0) AS fts_rank
            FROM recipes_fts
            JOIN recipes AS r ON r.id = recipes_fts.rowid
            WHERE {' AND '.join(clauses)}
            ORDER BY fts_rank ASC
            LIMIT ?
        """
        with closing(self._connect()) as connection:
            return self._rows_to_frame(connection.execute(sql, parameters))

    def random_recipe(self, category: str = "All") -> pd.Series | None:
        self.ensure_index()
        clause = "" if category == "All" else "WHERE category = ?"
        parameters: list[object] = [] if category == "All" else [category]
        with closing(self._connect()) as connection:
            row = connection.execute(
                f"SELECT * FROM recipes {clause} ORDER BY RANDOM() LIMIT 1",
                parameters,
            ).fetchone()
        return pd.Series(dict(row)) if row else None

    def browse(
        self,
        *,
        letter: str = "All",
        category: str = "All",
        page: int = 1,
        page_size: int = 6,
    ) -> tuple[pd.DataFrame, int, int]:
        self.ensure_index()
        clauses: list[str] = []
        parameters: list[object] = []
        if letter != "All":
            clauses.append("first_letter = ?")
            parameters.append(letter.upper())
        if category != "All":
            clauses.append("category = ?")
            parameters.append(category)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with closing(self._connect()) as connection:
            total = int(
                connection.execute(
                    f"SELECT COUNT(*) FROM recipes {where}", parameters
                ).fetchone()[0]
            )
            total_pages = max(1, (total + page_size - 1) // page_size)
            current_page = min(max(int(page), 1), total_pages)
            offset = (current_page - 1) * page_size
            rows = connection.execute(
                f"""
                SELECT * FROM recipes {where}
                ORDER BY name COLLATE NOCASE
                LIMIT ? OFFSET ?
                """,
                [*parameters, page_size, offset],
            )
            frame = self._rows_to_frame(rows)
        return frame, total, current_page

    def count(self) -> int:
        return self.ensure_index().row_count
