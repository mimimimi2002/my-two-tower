"""Train the two-tower model with a sampled softmax over uniform negatives."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data import InteractionExample, collate_examples, load_movielens
from model import TwoTower

ROOT = Path(__file__).resolve().parent


def sample_negatives(
    user_index: torch.Tensor,
    allowed_items: list[torch.Tensor],
    num_negatives: int,
) -> torch.Tensor:
    rows = []
    for user in user_index.detach().cpu().tolist():
        pool = allowed_items[user]
        if pool.numel() < num_negatives:
            raise RuntimeError(f"user {user} has fewer than {num_negatives} negative candidates")
        choice = pool[torch.randperm(pool.numel())[:num_negatives]]
        rows.append(choice)
    return torch.stack(rows).to(user_index.device)


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def encode_batch(model: TwoTower, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    return model.encode_queries(
        batch["user_index"],
        batch["gender_index"],
        batch["occupation_index"],
        batch["age"],
        batch["history"],
        batch["history_mask"],
    )


def sampled_softmax_loss(
    query: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    tau: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    positive_score = (query * positive).sum(dim=-1)
    negative_score = torch.einsum("bd,bkd->bk", query, negative)
    logits = torch.cat([positive_score.unsqueeze(1), negative_score], dim=1) / tau
    labels = torch.zeros(query.size(0), dtype=torch.long, device=query.device)
    loss = F.cross_entropy(logits, labels)
    return loss, positive_score.mean().detach(), negative_score.mean().detach()


@torch.no_grad()
def evaluate(
    model: TwoTower,
    examples: list[InteractionExample],
    device: torch.device,
    batch_size: int,
    ks: tuple[int, ...] = (10, 50),
) -> dict[str, float]:
    model.eval()
    item_embeddings = model.all_item_embeddings()
    hits = {k: 0 for k in ks}
    seen = 0
    max_k = max(ks)
    for start in range(0, len(examples), batch_size):
        batch = move_batch(collate_examples(examples[start : start + batch_size]), device)
        query = encode_batch(model, batch)
        scores = query @ item_embeddings.T
        for row in range(scores.size(0)):
            watched = batch["history"][row][batch["history_mask"][row]]
            scores[row, watched] = float("-inf")
        top = torch.topk(scores, k=max_k, dim=1).indices
        target = batch["target"].unsqueeze(1)
        for k in ks:
            hits[k] += int((top[:, :k] == target).any(dim=1).sum())
        seen += scores.size(0)
    return {f"recall@{k}": hits[k] / seen for k in ks}


def train_one_epoch(
    model: TwoTower,
    examples: list[InteractionExample],
    allowed_items: list[torch.Tensor],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    batch_size: int,
    num_negatives: int,
    tau: float,
) -> dict[str, float]:
    model.train()
    order = torch.randperm(len(examples)).tolist()
    total_loss = 0.0
    total_positive = 0.0
    total_negative = 0.0
    seen = 0
    for start in range(0, len(order), batch_size):
        batch_examples = [examples[index] for index in order[start : start + batch_size]]
        batch = move_batch(collate_examples(batch_examples), device)
        query = encode_batch(model, batch)
        positive = model.encode_items(batch["target"])
        negative_ids = sample_negatives(batch["user_index"], allowed_items, num_negatives)
        negative = model.encode_items(negative_ids)
        loss, positive_score, negative_score = sampled_softmax_loss(query, positive, negative, tau)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        batch_size_actual = batch["target"].size(0)
        total_loss += float(loss.detach()) * batch_size_actual
        total_positive += float(positive_score) * batch_size_actual
        total_negative += float(negative_score) * batch_size_actual
        seen += batch_size_actual
    return {
        "loss": total_loss / seen,
        "pos_dot": total_positive / seen,
        "neg_dot": total_negative / seen,
    }


def build_model(data, embedding_dim: int) -> TwoTower:
    return TwoTower(
        num_users=data.num_users,
        num_items=data.num_items,
        num_genders=data.num_genders,
        num_occupations=data.num_occupations,
        num_genres=data.num_genres,
        embedding_dim=embedding_dim,
        genre_multi_hot=data.genre_multi_hot,
    )


def save_checkpoint(
    path: Path,
    model: TwoTower,
    config: dict,
    item_embeddings: torch.Tensor,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "config": config}, path)
    np.save(path.parent / "item_embeddings.npy", item_embeddings.detach().cpu().numpy())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a two-tower model on MovieLens 100K")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-negatives", type=int, default=16)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    data = load_movielens(ROOT / "data")
    print(
        f"users={data.num_users} items={data.num_items} "
        f"train={len(data.train)} val={len(data.validation)} test={len(data.test)}"
    )

    allowed_items = [
        torch.nonzero(~data.positive_mask[user], as_tuple=False).flatten()
        for user in range(data.num_users)
    ]
    model = build_model(data, args.embedding_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    artifact_dir = ROOT / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "catalog.json").write_text(
        json.dumps(data.catalog, ensure_ascii=False),
        encoding="utf-8",
    )
    log_path = artifact_dir / "train_log.jsonl"
    log_path.write_text("", encoding="utf-8")

    config = {
        "num_users": data.num_users,
        "num_items": data.num_items,
        "num_genders": data.num_genders,
        "num_occupations": data.num_occupations,
        "num_genres": data.num_genres,
        "embedding_dim": args.embedding_dim,
        "tau": args.tau,
    }
    best_recall = -1.0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        stats = train_one_epoch(
            model,
            data.train,
            allowed_items,
            optimizer,
            device,
            args.batch_size,
            args.num_negatives,
            args.tau,
        )
        validation = evaluate(model, data.validation, device, args.batch_size)
        row = {"epoch": epoch, **stats, **{f"val_{key}": value for key, value in validation.items()}}
        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(row) + "\n")
        print(
            f"epoch {epoch}/{args.epochs}  loss={stats['loss']:.4f}  "
            f"pos_dot={stats['pos_dot']:.4f}  neg_dot={stats['neg_dot']:.4f}  "
            f"val_recall@10={validation['recall@10']:.4f}  "
            f"val_recall@50={validation['recall@50']:.4f}"
        )
        if validation["recall@10"] > best_recall:
            best_recall = validation["recall@10"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("no trained weights were saved")
    model.load_state_dict({key: value.to(device) for key, value in best_state.items()})
    test_metrics = evaluate(model, data.test, device, args.batch_size)
    print(
        f"test  recall@10={test_metrics['recall@10']:.4f}  "
        f"recall@50={test_metrics['recall@50']:.4f}"
    )
    model.eval()
    item_embeddings = model.all_item_embeddings()
    save_checkpoint(artifact_dir / "model.pt", model, config, item_embeddings)
    (artifact_dir / "test_metrics.json").write_text(
        json.dumps(test_metrics, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
