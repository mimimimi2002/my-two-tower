"""Return top-k items by dot product with the precomputed item embeddings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from data import InteractionExample, collate_examples
from model import TwoTower

ROOT = Path(__file__).resolve().parent


def load_model(artifact_dir: Path, device: torch.device) -> tuple[TwoTower, dict]:
    checkpoint_path = artifact_dir / "model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError("artifacts/model.pt is missing. Run python train.py first")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    genre_multi_hot = torch.zeros(config["num_items"], config["num_genres"])
    model = TwoTower(
        num_users=config["num_users"],
        num_items=config["num_items"],
        num_genders=config["num_genders"],
        num_occupations=config["num_occupations"],
        num_genres=config["num_genres"],
        embedding_dim=config["embedding_dim"],
        genre_multi_hot=genre_multi_hot,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, config


def recommend_for_user(
    model: TwoTower,
    item_embeddings: torch.Tensor,
    user: dict,
    k: int,
    device: torch.device,
) -> tuple[list[tuple[int, float]], int, int]:
    sequence = user["sequence"]
    history = tuple(sequence[:-1])
    held_out = int(sequence[-1])
    example = InteractionExample(
        user_index=int(user["index"]),
        gender_index=int(user["gender_index"]),
        occupation_index=int(user["occupation_index"]),
        age=float(user["age"]),
        history=history,
        target=held_out,
    )
    batch = {key: value.to(device) for key, value in collate_examples([example]).items()}
    with torch.no_grad():
        query = model.encode_queries(
            batch["user_index"],
            batch["gender_index"],
            batch["occupation_index"],
            batch["age"],
            batch["history"],
            batch["history_mask"],
        )
    scores = item_embeddings @ query.squeeze(0).detach().cpu()
    if history:
        scores[list(history)] = float("-inf")
    available = int(torch.isfinite(scores).sum())
    top_k = min(k, available)
    values, indices = torch.topk(scores, k=top_k)
    recommendations = [(int(index), float(score)) for index, score in zip(indices.tolist(), values.tolist())]
    rank = int((scores > scores[held_out]).sum()) + 1
    return recommendations, held_out, rank


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrieve movies from a trained two-tower model")
    parser.add_argument("--user-id", type=int, required=True, help="MovieLens user id")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact_dir = ROOT / "artifacts"
    catalog = json.loads((artifact_dir / "catalog.json").read_text(encoding="utf-8"))
    users = {int(user["raw_id"]): user for user in catalog["users"]}
    if args.user_id not in users:
        raise SystemExit(
            f"user {args.user_id} is not in this dataset because they have fewer than 3 ratings of 4 or higher"
        )
    device = torch.device(args.device)
    model, _config = load_model(artifact_dir, device)
    item_embeddings = torch.from_numpy(np.load(artifact_dir / "item_embeddings.npy"))
    titles = {int(item["index"]): item["title"] for item in catalog["items"]}
    recommendations, held_out, rank = recommend_for_user(
        model, item_embeddings, users[args.user_id], args.k, device
    )
    print(f"user {args.user_id}")
    print(f"holdout: {titles[held_out]}  (rank {rank})")
    for place, (item_index, score) in enumerate(recommendations, start=1):
        print(f"{place:2d}. {titles[item_index]}  dot={score:.3f}")


if __name__ == "__main__":
    main()
