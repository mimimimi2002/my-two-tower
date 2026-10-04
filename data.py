"""Turn MovieLens 100K into query and item features for the two-tower model."""

from __future__ import annotations

import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch

ML_100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
MIN_RATING = 4.0
MIN_POSITIVES = 3


@dataclass(frozen=True)
class InteractionExample:
    user_index: int
    gender_index: int
    occupation_index: int
    age: float
    history: tuple[int, ...]
    target: int


@dataclass
class PreparedData:
    train: list[InteractionExample]
    validation: list[InteractionExample]
    test: list[InteractionExample]
    genre_multi_hot: torch.Tensor
    positive_mask: torch.Tensor
    catalog: dict
    num_users: int
    num_items: int
    num_genders: int
    num_occupations: int
    num_genres: int


def ensure_ml100k(data_dir: Path) -> Path:
    root = data_dir / "ml-100k"
    if (root / "u.data").exists():
        return root
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / "ml-100k.zip"
    print(f"downloading {ML_100K_URL}")
    urllib.request.urlretrieve(ML_100K_URL, zip_path)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(data_dir)
    if not (root / "u.data").exists():
        raise FileNotFoundError(f"u.data was not found in {root}")
    return root


def _read_tables(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    ratings = pd.read_csv(
        root / "u.data",
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
        engine="python",
    )
    users = pd.read_csv(
        root / "u.user",
        sep="|",
        names=["user_id", "age", "gender", "occupation", "zip"],
        engine="python",
    )
    genre_table = pd.read_csv(
        root / "u.genre",
        sep="|",
        names=["name", "genre_id"],
        encoding="latin-1",
        engine="python",
    ).dropna(subset=["name"])
    genre_names = genre_table.sort_values("genre_id")["name"].astype(str).tolist()
    item_columns = ["item_id", "title", "release", "video_release", "url", *genre_names]
    items = pd.read_csv(
        root / "u.item",
        sep="|",
        names=item_columns,
        encoding="latin-1",
        engine="python",
    )
    return ratings, users, items, genre_names


def load_movielens(data_dir: Path) -> PreparedData:
    root = ensure_ml100k(data_dir)
    ratings, users, items, genre_names = _read_tables(root)

    positives = ratings.loc[ratings["rating"] >= MIN_RATING].copy()
    positives = positives.sort_values(["user_id", "timestamp", "item_id"])
    positives = positives.drop_duplicates(["user_id", "item_id"], keep="last")

    counts = positives.groupby("user_id").size()
    eligible_user_ids = sorted(int(user_id) for user_id in counts[counts >= MIN_POSITIVES].index)
    eligible_users = users[users["user_id"].isin(eligible_user_ids)].copy()

    item_ids = sorted(int(item_id) for item_id in items["item_id"].tolist())
    item_index = {raw_id: index for index, raw_id in enumerate(item_ids)}
    genders = sorted(eligible_users["gender"].astype(str).unique().tolist())
    occupations = sorted(eligible_users["occupation"].astype(str).unique().tolist())
    gender_index = {name: index for index, name in enumerate(genders)}
    occupation_index = {name: index for index, name in enumerate(occupations)}

    items = items.set_index("item_id")
    genre_multi_hot = torch.zeros(len(item_ids), len(genre_names), dtype=torch.float32)
    item_catalog = []
    for raw_id, index in item_index.items():
        row = items.loc[raw_id]
        flags = [float(row[name]) for name in genre_names]
        genre_multi_hot[index] = torch.tensor(flags, dtype=torch.float32)
        item_catalog.append({"index": index, "raw_id": raw_id, "title": str(row["title"])})

    user_rows = eligible_users.set_index("user_id")
    user_index = {raw_id: index for index, raw_id in enumerate(eligible_user_ids)}
    positive_mask = torch.zeros(len(eligible_user_ids), len(item_ids), dtype=torch.bool)
    catalog_users = []
    train: list[InteractionExample] = []
    validation: list[InteractionExample] = []
    test: list[InteractionExample] = []

    for raw_user_id, index in user_index.items():
        profile = user_rows.loc[raw_user_id]
        sequence = tuple(
            item_index[int(item_id)]
            for item_id in positives.loc[positives["user_id"] == raw_user_id, "item_id"].tolist()
            if int(item_id) in item_index
        )
        if len(sequence) < MIN_POSITIVES:
            continue
        positive_mask[index, list(sequence)] = True
        example_kwargs = {
            "user_index": index,
            "gender_index": gender_index[str(profile["gender"])],
            "occupation_index": occupation_index[str(profile["occupation"])],
            "age": float(profile["age"]) / 100.0,
        }
        for position, target in enumerate(sequence[:-2]):
            train.append(
                InteractionExample(
                    history=sequence[:position],
                    target=target,
                    **example_kwargs,
                )
            )
        validation.append(
            InteractionExample(history=sequence[:-2], target=sequence[-2], **example_kwargs)
        )
        test.append(
            InteractionExample(history=sequence[:-1], target=sequence[-1], **example_kwargs)
        )
        catalog_users.append(
            {
                "raw_id": raw_user_id,
                "index": index,
                "gender_index": example_kwargs["gender_index"],
                "occupation_index": example_kwargs["occupation_index"],
                "age": example_kwargs["age"],
                "sequence": list(sequence),
            }
        )

    catalog = {
        "users": catalog_users,
        "items": item_catalog,
        "genders": genders,
        "occupations": occupations,
        "genres": genre_names,
    }
    return PreparedData(
        train=train,
        validation=validation,
        test=test,
        genre_multi_hot=genre_multi_hot,
        positive_mask=positive_mask,
        catalog=catalog,
        num_users=len(catalog_users),
        num_items=len(item_ids),
        num_genders=len(genders),
        num_occupations=len(occupations),
        num_genres=len(genre_names),
    )


def collate_examples(examples: list[InteractionExample]) -> dict[str, torch.Tensor]:
    max_len = max((len(example.history) for example in examples), default=0)
    max_len = max(max_len, 1)
    history = torch.zeros(len(examples), max_len, dtype=torch.long)
    history_mask = torch.zeros(len(examples), max_len, dtype=torch.bool)
    for row, example in enumerate(examples):
        if not example.history:
            continue
        length = len(example.history)
        history[row, :length] = torch.tensor(example.history, dtype=torch.long)
        history_mask[row, :length] = True
    return {
        "user_index": torch.tensor([example.user_index for example in examples], dtype=torch.long),
        "gender_index": torch.tensor([example.gender_index for example in examples], dtype=torch.long),
        "occupation_index": torch.tensor(
            [example.occupation_index for example in examples], dtype=torch.long
        ),
        "age": torch.tensor([[example.age] for example in examples], dtype=torch.float32),
        "history": history,
        "history_mask": history_mask,
        "target": torch.tensor([example.target for example in examples], dtype=torch.long),
    }
