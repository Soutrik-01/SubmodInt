import torch 	
from tqdm import tqdm
import numpy as np
import sys
import pickle as pkl
import torch
from sklearn.cluster import KMeans
from scipy import spatial
import faiss 
import torchvision
import os 


def get_cdist(V):
	def get_square(mat):
		return mat**2
	dist_mat = torch.cdist(V, V)
	return get_square(dist_mat)

def get_rbf_kernel(dist_mat, kw):
	bsz=2048
	sim = torch.exp(-dist_mat/(kw*dist_mat.mean()))
	return sim

def pad_probs(probs, num_padding):
	probs = np.concatenate([np.asarray(probs), np.zeros(num_padding)])
	return probs


class DataGenerator():
	"""docstring for DataGenerator"""
	def __init__(self, X, Y, sampling_type=[1,0,0,0], train=True):
		self.args = {'K': 100, 'prob_list':[0.6, 0.4], 'set_size':100, 'n_iteration':1000, 'bsz':40}
		self.sampling_type = np.array(sampling_type)
		self.sampling_type = sampling_type/np.sum(sampling_type)
		self.X = X
		self.Y = np.asarray(Y)
		self.V_partition_to_gs={}
		for i in range(self.args['K']):
			self.V_partition_to_gs[i] = np.nonzero(self.Y==i)[0]

		self.prob_list = pad_probs(self.args['prob_list'],self.args['K']-len(self.args['prob_list']))
		if train:
			self.H = np.load('H_train.npy')
			self.sim = get_rbf_kernel(get_cdist(torch.from_numpy(np.load('H_train.npy')).cuda()), 0.01).cpu()
		else:
			self.H = np.load('H_test.npy')
			self.sim = get_rbf_kernel(get_cdist(torch.from_numpy(np.load('H_test.npy')).cuda()), 0.01).cpu()
		torch.cuda.empty_cache()

	def get_example(self, fname):	
		dataset={}

		for _ in tqdm(range(int(self.sampling_type[0] * (self.args['n_iteration']*self.args['bsz'])))):
	
			num_clust=1; ones = np.ones(self.args['K'])
			class_idx = np.random.choice(self.args['K'], size=num_clust, replace=False)[0]
			ones[class_idx]=0
			D_A_idx = np.random.choice(self.V_partition_to_gs[class_idx], self.args['set_size'], replace=False)
			E_set_class_idx = np.random.choice(np.nonzero(ones)[0], 1)[0]; E_set_GS = np.concatenate([self.V_partition_to_gs[class_idx], self.V_partition_to_gs[E_set_class_idx]])
			D_B_idx = np.random.choice(E_set_GS, self.args['set_size'], replace=False)
	
			y_A = np.asarray([class_idx])
			y_B = np.asarray([class_idx, E_set_class_idx])

			if not ((len(y_A), len(y_B)) in dataset.keys()):
				dataset[(len(y_A), len(y_B))]=[(y_A, y_B, D_A_idx, D_B_idx)]
			else:
				dataset[(len(y_A), len(y_B))].append((y_A, y_B, D_A_idx, D_B_idx))
			assert len(D_A_idx) == self.args['set_size']
			assert len(D_B_idx) == self.args['set_size']

			if not num_clust==1:
				class_idx=class_idx[0]
				D_A_idx = (self.V_partition_to_gs[class_idx])[np.random.choice(len(self.V_partition_to_gs[class_idx]), self.args['set_size'], replace=False)]
				y_A = np.unique(self.Y[D_A_idx])

				assert y_A[0] == class_idx

				D_B_idx = ground_set[np.random.choice(len(ground_set), self.args['set_size'], replace=False)]
				y_B = np.unique(self.Y[D_B_idx])

				if not ((len(y_A), len(y_B)) in dataset.keys()):
					dataset[(len(y_A), len(y_B))]=[(y_A, y_B, D_A_idx, D_B_idx)]
				else:
					dataset[(len(y_A), len(y_B))].append((y_A, y_B, D_A_idx, D_B_idx))
				assert len(D_A_idx) == self.args['set_size']
				assert len(D_B_idx) == self.args['set_size']


		for _ in tqdm(range(int(self.sampling_type[1] * (self.args['n_iteration']*self.args['bsz'])))):

			D_A_idx = []
			D_B_idx = []


			cls_list = [i for i in range(self.args['K'])]
			np.random.shuffle(cls_list)

			for k in cls_list:
				idx = np.random.choice(self.V_partition_to_gs[k], 1)[0]
				assert self.Y[idx] == k

				tmp = torch.topk(self.sim[idx,self.V_partition_to_gs[k]],k=5)[1][-1]
				idx2 = self.V_partition_to_gs[k][tmp]
				assert self.Y[idx2] == k

				D_A_idx.append(idx)
				D_A_idx.append(idx2)


			for k in cls_list:

				idx = np.random.choice(self.V_partition_to_gs[k], 1)[0]
				assert self.Y[idx] == k

				idx2 = self.V_partition_to_gs[k][np.argmin(self.sim[idx,self.V_partition_to_gs[k]])]
				assert self.Y[idx2] == k

				D_B_idx.append(idx)
				D_B_idx.append(idx2)


			y_A = np.unique(self.Y[D_A_idx])
			y_B = np.unique(self.Y[D_B_idx])



			if not ((len(y_A), len(y_B)) in dataset.keys()):
				dataset[(len(y_A), len(y_B))]=[(y_A, y_B, D_A_idx, D_B_idx)]
			else:
				dataset[(len(y_A), len(y_B))].append((y_A, y_B, D_A_idx, D_B_idx))

		for _ in tqdm(range(int(self.sampling_type[2] * (self.args['n_iteration']*self.args['bsz'])))):

			num_clust = 2
			class_idx = np.random.choice(self.args['K'], size=num_clust, replace=False)

			D_A_idx = []

			lists = np.array_split([i for i in range(self.args['set_size'])], num_clust)
			num_per_clust = [len(l) for l in lists]
			assert np.sum(num_per_clust) == self.args['set_size']

			for k,num in zip(class_idx, num_per_clust):

				if num == 0:
					continue

				idx = np.random.choice(self.V_partition_to_gs[k], 1)[0]
				assert self.Y[idx] == k
				D_A_idx.append(idx)

				if num == 1:
					continue

				tmp = torch.topk(self.sim[idx,self.V_partition_to_gs[k]],k=num)[1][1:]
				idx2 = self.V_partition_to_gs[k][tmp]


				if num > 2:
					D_A_idx.extend(idx2)
				else:
					D_A_idx.append(idx2)

			assert len(D_A_idx) == self.args['set_size']


			D_B_idx = []

			ground_set = []
			for j in range(self.args['K']):
				if j in class_idx:
					ground_set.append(self.V_partition_to_gs[j])

			ground_set = np.concatenate(ground_set)


			X = self.H[ground_set]
			kmeans_n_init = 1
			summary_size = self.args['set_size']
			d = X.shape[-1]
			kmeans_model = faiss.Kmeans(d, summary_size); kmeans_model.train(X) 
			index = faiss.IndexFlatL2(d)
			index.add(X)

			_, I = index.search(kmeans_model.centroids, 1)
			D_B_idx = ground_set[I].squeeze()

			assert len(D_B_idx) == self.args['set_size']


			sorted_idx = np.argsort(self.Y[D_B_idx])
			D_B_idx = (np.array(D_B_idx)[sorted_idx]).tolist()

			y_A = np.unique(self.Y[D_A_idx])
			y_B = np.unique(self.Y[D_B_idx])

			if not ((len(y_A), len(y_B)) in dataset.keys()):
				dataset[(len(y_A), len(y_B))]=[(y_A, y_B, D_A_idx, D_B_idx)]
			else:
				dataset[(len(y_A), len(y_B))].append((y_A, y_B, D_A_idx, D_B_idx))


		for _ in tqdm(range(int(self.sampling_type[3] * (self.args['n_iteration']*self.args['bsz'])))):

			num_clust = np.random.choice(self.args['K']-1)+1
			class_idx = np.random.choice(self.args['K'], size=num_clust, replace=False)

			D_A_idx = []

			lists = np.array_split([i for i in range(self.args['set_size'])], num_clust)
			num_per_clust = [len(l) for l in lists]
			assert np.sum(num_per_clust) == self.args['set_size']

			for k,num in zip(class_idx, num_per_clust):

				if num == 0:
					continue

				idx = np.random.choice(self.V_partition_to_gs[k], 1)[0]
				assert self.Y[idx] == k
				D_A_idx.append(idx)

				tmp = torch.topk(self.sim[idx,self.V_partition_to_gs[k]],k=num)[1][1:]
				idx2 = self.V_partition_to_gs[k][tmp]


				if num > 2:
					D_A_idx.extend(idx2)
				else:
					D_A_idx.append(idx2)

			assert len(D_A_idx) == self.args['set_size']

			D_B_idx = []


			X = self.H
			kmeans_n_init = 1
			summary_size = self.args['set_size']
			kmeans_model = KMeans(n_clusters=summary_size,init='random',n_init=kmeans_n_init).fit(X)
			kd_tree = spatial.KDTree(X)
			
			for ii in range(summary_size):
				item = kd_tree.query(kmeans_model.cluster_centers_[ii,:],p=1)[1]
				D_B_idx.append(item)

			assert len(D_B_idx) == self.args['set_size']



			y_A = np.unique(self.Y[D_A_idx])
			y_B = np.unique(self.Y[D_B_idx])
			if not ((len(y_A), len(y_B)) in dataset.keys()):
				dataset[(len(y_A), len(y_B))]=[(y_A, y_B, D_A_idx, D_B_idx)]
			else:
				dataset[(len(y_A), len(y_B))].append((y_A, y_B, D_A_idx, D_B_idx))



		for k in dataset.keys():
			y_A, y_B, D_A, D_B = zip(*dataset[k])
			dataset[k]=(y_A, y_B, torch.from_numpy(np.stack(D_A, axis=0)), torch.from_numpy(np.stack(D_B, axis=0)) )

		with open(fname, 'wb') as f:
			pkl.dump(dataset, f)

path = "enter your path here"
X, Y= np.load(f"{path}/dataset/X_train.npy"), np.load(f"{path}/dataset/Y_train.npy")
DataGenerator(X, Y, sampling_type=[1,0,1,0]).get_example(fname="passive_samples.pkl")
