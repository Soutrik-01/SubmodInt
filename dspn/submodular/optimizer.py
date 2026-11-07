import numpy as np
import torch 
import math
from tqdm import tqdm as tqdm


class DSF_opt():
	'''
	To perform submodular max on DSF
	'''
	def __init__(self, model, H):
		self.model = model
		self.out_cache=0 
		self.gs_feat = H

	def add(self, e):
		out_cache_old = self.out_cache
		self.in_cache +=  self.gs_feat[e]
		self.out_cache  = self.model.dsf(self.in_cache.unsqueeze(dim=0))
		return self.out_cache - out_cache_old

	def greedy_max(self, B, device, out_dim, cc=False):
		self.in_cache= torch.zeros(out_dim).to(device)
		with torch.no_grad():
			sset=[]; gain=[]
			R = set([i for i in range(len(self.gs_feat))])
			while(len(sset)<B):
				R_list = list(R)
				g = self.model.dsf(self.in_cache.unsqueeze(dim=0) + self.gs_feat[R_list]) - self.out_cache
				g = g.squeeze()
				idx = torch.argmax(g); gg = self.add(R_list[idx.item()])
				gain.append(gg.item())
				R.remove(R_list[idx.item()])
				sset.append(R_list[idx.item()])

			if cc:	
				X_sset = self.gs_feat[sset]
				cc = 1 - (torch.FloatTensor(gain).squeeze()).cuda()/(self.model.dsf(X_sset.cuda())).squeeze()
				return sset, cc 
			else:	
				return sset

	def balanced_stochastic_greedy_max(self, B, device, out_dim, Y, K, epsilon=1e-2, cc=False, r_size=None):
		budgets = np.ones(K)*(B//K) 
		current_count = np.zeros(K)
		if B%K != 0 :
			idx = np.random.choice(K, B%K, replace=False)
			budgets[idx] += 1
		assert budgets.sum() == B 

		self.in_cache= torch.zeros(out_dim).to(device)
		n = self.gs_feat.shape[0]
		if  (r_size is None):
			r_size = int((n/B)*math.log(1/epsilon))

		with torch.no_grad():
			sset=[]; gain=[]
			V = set([i for i in range(len(self.gs_feat))])
			while(len(sset)<B):
				idx = np.random.choice(len(V), r_size, replace=False)
				R_list = np.asarray(list(V))[idx]
				g = self.model.dsf(self.in_cache.unsqueeze(dim=0) + self.gs_feat[R_list]) - self.out_cache
				g = g.squeeze()

				order = torch.argsort(-g)
				for loop_idx, jj in enumerate(order):
					true_idx = R_list[jj.item()]
					true_class = Y[true_idx]
					if current_count[true_class] < budgets[true_class]:
						idx = jj.item() 
						current_count[true_class]+=1
						gg = self.add(true_idx)
						gain.append(gg.item())
						V.remove(true_idx)
						sset.append(true_idx)
						break 
						
				if loop_idx == len(order) -1:
					print("Re-sampling in order to meet the balance constraint.")
				
		return sset 

	def stochastic_greedy_prob(self, B, device, out_dim, probs, epsilon=1e-2, cc=False, r_size=None):
		self.in_cache= torch.zeros(out_dim).to(device)
		n = self.gs_feat.shape[0]
		if r_size is None:	
			r_size = int((n/B)*math.log(1/epsilon))
		coin_flips = np.random.choice([0, 1], B, p=[1-probs, probs])

		with torch.no_grad():
			sset=[]; gain=[]
			V = set([i for i in range(len(self.gs_feat))])
			while(len(sset)<B):
				idx = np.random.choice(len(V), r_size, replace=False)
				R_list = np.asarray(list(V))[idx]
				g = self.model.dsf(self.in_cache.unsqueeze(dim=0) + self.gs_feat[R_list]) - self.out_cache
				g = g.squeeze()
				if coin_flips[len(gain)]  == 1:	
					idx = torch.argmax(g); gg = self.add(R_list[idx.item()])
				else:
					idx = torch.argmin(g); gg = self.add(R_list[idx.item()])
					
				gain.append(gg.item())
				V.remove(R_list[idx.item()])
				sset.append(R_list[idx.item()])

			X_sset = self.gs_feat[sset]

			if cc:	
				cc = 1 - (torch.FloatTensor(gain).squeeze()).cuda()/(self.model.dsf(X_sset.cuda())).squeeze()
				return sset, cc 
			else:	
				return sset

	def stochastic_greedy(self, B, device, out_dim, epsilon=1e-2, cc=False, r_size=None):
		self.in_cache= torch.zeros(out_dim).to(device)
		n = self.gs_feat.shape[0]
		if  (r_size is None):
			r_size = int((n/B)*math.log(1/epsilon))
		
		with torch.no_grad():
			sset=[]; gain=[]
			V = set([i for i in range(len(self.gs_feat))])
			while(len(sset)<B):
				idx = np.random.choice(len(V), r_size, replace=False)
				R_list = np.asarray(list(V))[idx]
				g = self.model.dsf(self.in_cache.unsqueeze(dim=0) + self.gs_feat[R_list]) - self.out_cache
				g = g.squeeze()
				idx = torch.argmax(g); gg = self.add(R_list[idx.item()])
				gain.append(gg.item())
				V.remove(R_list[idx.item()])
				sset.append(R_list[idx.item()])

			X_sset = self.gs_feat[sset]

			if cc:	
				cc = 1 - (torch.FloatTensor(gain).squeeze()).cuda()/(self.model.dsf(X_sset.cuda())).squeeze()
				return sset, cc 
			else:	
				return sset


	def streaming_max(self, delta, L, K, C=1, out_dim=512, alpha=1e-5, device='cuda:0'):
		sset=[]; gains=[] 

		cur_val_dsf = 0
		cur_val_mod = torch.zeros(out_dim).to(device)

		for t,v in enumerate(tqdm(self.gs_feat)):
			val_m = cur_val_mod + v
			val_f = self.model.dsf(val_m.unsqueeze(dim=0))
			g =  val_f - cur_val_dsf
			gains.append(g.item())

			if len(sset) < K:
				if g > alpha:
					sset.append(t)
					cur_val_dsf = val_f 
					cur_val_mod = val_m 

			else:
				c = np.mean(gains[-L:]) + delta 

				cur_best = -np.inf
				for j in sset:
					v_del = self.gs_feat[j]
					val_m_ = cur_val_mod + v - v_del
					val_f_ = self.model.dsf(val_m_.unsqueeze(dim=0))
					g_hat_ = val_f_ - cur_val_dsf

					if g_hat_ > cur_best:
						val_m = val_m_
						val_f = val_f_
						s_del = j
						g_hat = g_hat_
						cur_best = g_hat 


				if (g_hat >= C * cur_val_dsf/K):
					sset.remove(s_del)
					sset.append(t)

					cur_val_dsf = val_f
					cur_val_mod = val_m

				elif g >= c:
					sset.remove(s_del)
					sset.append(t)

					cur_val_dsf = val_f
					cur_val_mod = val_m
		return sset


class DSF_opt_set_transformer():
	'''
	To perform submodular max on DSF
	'''
	def __init__(self, model, H):
		self.model = model
		self.out_cache=0 
		self.in_cache = []
		self.gs_feat = H

	def add(self, e):
		out_cache_old = self.out_cache
		self.in_cache +=  [e]
		input = self.gs_feat[self.in_cache].unsqueeze(0)
		features =  self.model.feat(input)[1]
		self.out_cache = self.model.dsf(features).squeeze()
		return self.out_cache - out_cache_old

	def greedy_max(self, B, device, out_dim, cc=False):
		self.in_cache = []
		self.gs_feat = self.gs_feat.to(device)
		with torch.no_grad():
			sset=[]; gain=[]
			R = set([i for i in range(len(self.gs_feat))])
			while(len(sset)<B):
				R_list = list(R); print(len(sset))
				sset_temp = torch.tensor([sset + [r] for r in R_list])
				features = self.model.feat(self.gs_feat[sset_temp])[1]
				g = self.model.dsf(features).squeeze()  - self.out_cache
				g = g.squeeze()
				idx = torch.argmax(g); gg = self.add(R_list[idx.item()])
				gain.append(gg.item())
				R.remove(R_list[idx.item()])
				sset.append(R_list[idx.item()])

			return sset

	def balanced_stochastic_greedy_max(self, B, device, out_dim, Y, K, epsilon=1e-2, cc=False, r_size=None):
		self.in_cache = []
		self.gs_feat = self.gs_feat.to(device)
		budgets = np.ones(K)*(B//K) 
		current_count = np.zeros(K)
		if B%K != 0 :
			idx = np.random.choice(K, B%K, replace=False)
			budgets[idx] += 1
		assert budgets.sum() == B 

		n = self.gs_feat.shape[0]
		if  (r_size is None):
			r_size = int((n/B)*math.log(1/epsilon))

		with torch.no_grad():
			sset=[]; gain=[]
			V = set([i for i in range(len(self.gs_feat))])
			while(len(sset)<B):
				idx = np.random.choice(len(V), r_size, replace=False)
				R_list = np.asarray(list(V))[idx]

				sset_temp = torch.tensor([sset + [r] for r in R_list])
				features = self.model.feat(self.gs_feat[sset_temp])[1]
				g = self.model.dsf(features).squeeze()  - self.out_cache
				g = g.squeeze()

				order = torch.argsort(-g)
				for loop_idx, jj in enumerate(order):
					true_idx = R_list[jj.item()]
					true_class = Y[true_idx]
					if current_count[true_class] < budgets[true_class]:
						idx = jj.item() 
						current_count[true_class]+=1
						gg = self.add(true_idx)
						gain.append(gg.item())
						V.remove(true_idx)
						sset.append(true_idx)
						break 
						
				if loop_idx == len(order) -1:
					print("Re-sampling in order to meet the balance constraint.")
				
		return sset 

	def stochastic_greedy_prob(self, B, device, out_dim, probs, epsilon=1e-2, cc=False, r_size=None):
		self.in_cache = []
		self.gs_feat = self.gs_feat.to(device)
		n = self.gs_feat.shape[0]
		if r_size is None:	
			r_size = int((n/B)*math.log(1/epsilon))
		coin_flips = np.random.choice([0, 1], B, p=[1-probs, probs])

		with torch.no_grad():
			sset=[]; gain=[]
			V = set([i for i in range(len(self.gs_feat))])
			while(len(sset)<B):
				idx = np.random.choice(len(V), r_size, replace=False)
				R_list = np.asarray(list(V))[idx]

				sset_temp = torch.tensor([sset + [r] for r in R_list])
				features = self.model.feat(self.gs_feat[sset_temp])[1]
				g = self.model.dsf(features).squeeze()  - self.out_cache
				g = g.squeeze()

				if coin_flips[len(gain)]  == 1:	
					idx = torch.argmax(g); gg = self.add(R_list[idx.item()])
				else:
					idx = torch.argmin(g); gg = self.add(R_list[idx.item()])
					
				gain.append(gg.item())
				V.remove(R_list[idx.item()])
				sset.append(R_list[idx.item()])

			X_sset = self.gs_feat[sset]

			if cc:	
				cc = 1 - (torch.FloatTensor(gain).squeeze()).cuda()/(self.model.dsf(X_sset.cuda())).squeeze()
				return sset, cc 
			else:	
				return sset

	def stochastic_greedy(self, B, device, out_dim, epsilon=1e-2, cc=False, r_size=None):
		self.in_cache = []
		self.gs_feat = self.gs_feat.to(device)
		n = self.gs_feat.shape[0]
		if  (r_size is None):
			r_size = int((n/B)*math.log(1/epsilon))
		
		with torch.no_grad():
			sset=[]; gain=[]
			V = set([i for i in range(len(self.gs_feat))])
			while(len(sset)<B):
				idx = np.random.choice(len(V), r_size, replace=False)
				R_list = np.asarray(list(V))[idx]

				sset_temp = torch.tensor([sset + [r] for r in R_list])
				features = self.model.feat(self.gs_feat[sset_temp])[1]
				g = self.model.dsf(features).squeeze()  - self.out_cache
				g = g.squeeze()
				idx = torch.argmax(g); gg = self.add(R_list[idx.item()])
				gain.append(gg.item())
				V.remove(R_list[idx.item()])
				sset.append(R_list[idx.item()])

			return sset
