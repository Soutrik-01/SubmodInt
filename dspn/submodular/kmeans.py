import torch


def nn_kmeans(X, k, iters=25, generator=None):
    """Lloyd's k-means (random-point init, as faiss.Kmeans) on the rows of X; returns, for each centroid,
    the index of its nearest row (the NN-k-means summary used by DSPN's pair sampling)."""
    centroids = X[torch.randperm(len(X), generator=generator, device=X.device)[:k]].clone()
    sq = (X ** 2).sum(1, keepdim=True)
    for _ in range(iters):
        assign = (sq - 2 * X @ centroids.T + (centroids ** 2).sum(1)).argmin(1)
        sums = torch.zeros_like(centroids).index_add_(0, assign, X)
        counts = torch.bincount(assign, minlength=k).unsqueeze(1)
        centroids = torch.where(counts > 0, sums / counts.clamp(min=1), centroids)
    return (sq - 2 * X @ centroids.T + (centroids ** 2).sum(1)).argmin(0)
