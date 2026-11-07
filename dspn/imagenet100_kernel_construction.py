import torch
import numpy as np 
from tqdm.notebook import tqdm


X = np.load("IN100_X_train.npy")[:, :, 0]
X = torch.from_numpy(X).cuda()
dist_mat = torch.cdist(X,X)


N=130000
num_shards=65
shard_size = N//num_shards
for jj in tqdm(range(num_shards)):
    start = jj*shard_size; end = (jj+1)*shard_size
    dist_mat[start:end, :] = dist_mat[start:end, :]**2



mean = dist_mat.mean().item()
kw=0.3


N=130000
num_shards=65
shard_size = N//num_shards
for jj in tqdm(range(num_shards)):
    start = jj*shard_size; end = (jj+1)*shard_size
    dist_mat[start:end, :] = torch.exp(-dist_mat[start:end, :]/(mean*kw))


dist_mat = dist_mat.cpu().numpy()



N=130000
num_shards=65
shard_size = N//num_shards
for jj in tqdm(range(num_shards)):
    start = jj*shard_size; end = (jj+1)*shard_size
    np.save(f"IN100_sim_train_sharded_{jj}.npy", dist_mat[start:end, :])


