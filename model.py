"""Query tower and item tower. The score is a dot product, not a full-corpus softmax."""

from __future__ import annotations

import torch
from torch import nn


def _init_embedding(embedding: nn.Embedding) -> None:
    nn.init.normal_(embedding.weight, std=0.1)


class ItemTower(nn.Module):
    def __init__(
        self,
        num_items: int,
        num_genres: int,
        embedding_dim: int,
        genre_multi_hot: torch.Tensor,
    ) -> None:
        super().__init__()
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.genre_embedding = nn.Embedding(num_genres, embedding_dim)
        self.register_buffer("genre_multi_hot", genre_multi_hot.float())
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        _init_embedding(self.item_embedding)
        _init_embedding(self.genre_embedding)

    def forward(self, item_ids: torch.Tensor) -> torch.Tensor:
        id_embedding = self.item_embedding(item_ids)
        genre_sum = self.genre_multi_hot[item_ids] @ self.genre_embedding.weight
        return self.mlp(id_embedding + genre_sum)


class QueryTower(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_genders: int,
        num_occupations: int,
        embedding_dim: int,
    ) -> None:
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.gender_embedding = nn.Embedding(num_genders, embedding_dim)
        self.occupation_embedding = nn.Embedding(num_occupations, embedding_dim)
        self.age_projection = nn.Linear(1, embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 5, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        _init_embedding(self.user_embedding)
        _init_embedding(self.gender_embedding)
        _init_embedding(self.occupation_embedding)

    def forward(
        self,
        user_index: torch.Tensor,
        gender_index: torch.Tensor,
        occupation_index: torch.Tensor,
        age: torch.Tensor,
        history_mean: torch.Tensor,
    ) -> torch.Tensor:
        features = torch.cat(
            [
                self.user_embedding(user_index),
                self.gender_embedding(gender_index),
                self.occupation_embedding(occupation_index),
                self.age_projection(age),
                history_mean,
            ],
            dim=-1,
        )
        return self.mlp(features)


class TwoTower(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        num_genders: int,
        num_occupations: int,
        num_genres: int,
        embedding_dim: int,
        genre_multi_hot: torch.Tensor,
    ) -> None:
        super().__init__()
        self.item_tower = ItemTower(num_items, num_genres, embedding_dim, genre_multi_hot)
        self.query_tower = QueryTower(num_users, num_genders, num_occupations, embedding_dim)

    def encode_items(self, item_ids: torch.Tensor) -> torch.Tensor:
        return self.item_tower(item_ids)

    def encode_queries(
        self,
        user_index: torch.Tensor,
        gender_index: torch.Tensor,
        occupation_index: torch.Tensor,
        age: torch.Tensor,
        history: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> torch.Tensor:
        history_embedding = self.item_tower(history)
        mask = history_mask.unsqueeze(-1).to(history_embedding.dtype)
        summed = (history_embedding * mask).sum(dim=1)
        denom = history_mask.sum(dim=1, keepdim=True).clamp(min=1).to(history_embedding.dtype)
        history_mean = summed / denom
        return self.query_tower(user_index, gender_index, occupation_index, age, history_mean)

    def all_item_embeddings(self) -> torch.Tensor:
        device = self.item_tower.item_embedding.weight.device
        item_ids = torch.arange(self.item_tower.item_embedding.num_embeddings, device=device)
        return self.encode_items(item_ids)


def dot_product(query: torch.Tensor, item: torch.Tensor) -> torch.Tensor:
    return (query * item).sum(dim=-1)
