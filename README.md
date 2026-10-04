# my-two-tower

A MovieLens 100K implementation of the two-tower model from Google's [Recommendation Systems](https://developers.google.com/machine-learning/recommendation) course.

The query tower maps user features to a vector, and the item tower maps movie features to a vector. The score is the dot product of those two vectors. The model does not compute a softmax over the whole corpus.

## Course mapping

- `data.py`: Treat ratings of 4 and above as implicit positives. A query is the user id, gender, occupation, age, and movies watched before the target. An item is the movie id plus genres. Age is divided by 100 and passed in as a single numeric feature.
- `model.py`: Query tower \(\psi(x_{\text{query}})\) and item tower \(\phi(x_{\text{item}})\). Each tower passes its embeddings through an MLP. On the item side, the id embedding is added to the genre embeddings. The score is the dot product, with no L2 normalization.
- `train.py`: For each positive, uniformly sample 16 movies the user has not interacted with and apply a sampled softmax. The temperature \(\tau\) is 0.1. These negatives prevent the folding that happens when training on positives only. Each epoch's mean positive and negative dot products, plus Recall@10 and Recall@50, are written to `artifacts/train_log.jsonl`.
- `retrieve.py`: Precompute an embedding for every movie, then rank the whole catalog by dot product with the query. Movies already in the history are removed. When the catalog is larger than MovieLens 100K and scoring every item is too expensive, run approximate nearest neighbor search over these precomputed embeddings.

Within each user, interactions are ordered by time. The last one is test, the one before it is validation, and the rest are train. The history used for a prediction contains only movies watched before that movie.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python train.py
python retrieve.py --user-id 1
```

`train.py` defaults to 5 epochs, an embedding size of 64, 16 negatives, and CPU. After training it writes `artifacts/model.pt` and `artifacts/item_embeddings.npy`.

`retrieve.py` builds the query from the history with the last movie removed. The printed holdout is that removed movie, and its rank is the dot-product rank after the history has been excluded.
