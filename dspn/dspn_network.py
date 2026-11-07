import torch
import torch.nn as nn
import torch.nn.functional as F
from models.dspn.roof.dsf import  DSF, SelfAttentionDecoder
from models.dspn.pillar.MLP import MLP, MLP_small, SelfAttentionEncoder
from utils import peripteral_cost, compute_accuracy, max_margin_loss
import sys 

class DSPN(nn.Module):
	def __init__(self, out_dims=[512], concave=['log1p'], args=None, feat=None, dsf=None, nesting_list=None):
		super(DSPN, self).__init__()
		if args['model_type'] == 'set_transformer':
			self.feat = SelfAttentionEncoder() 
		else:	
			if feat is None:
				self.feat = MLP() if args['dset'] in ['IN100', 'CIFAR100'] else MLP_small()
			else:
				self.feat = feat 
		
		if args['model_type'] == 'set_transformer':
			self.dsf = SelfAttentionDecoder()	
		elif args['model_type'] in ('submodint', 'submodint_f'):
			from models.dspn.roof.per_feature_submodint import PerFeatureSubmodInt
			self.dsf = PerFeatureSubmodInt(out_dims[0], fixed=args['model_type'] == 'submodint_f', max_set_size=args['set_size'], gamma=args.get('submodint_gamma', 2.5))
		elif args['model_type'] == 'hybrid':
			from models.dspn.roof.per_feature_submodint import PerFeatureSubmodInt
			from models.dspn.roof.submodint_dsf_mixture import SubmodIntDSFMixture
			submodint = PerFeatureSubmodInt(out_dims[0], fixed=False, max_set_size=args['set_size'], gamma=args.get('submodint_gamma', 2.5))
			self.dsf = SubmodIntDSFMixture(submodint, DSF(out_dims, concave), args['hybrid_w_init'])
		else:
			self.dsf = DSF(out_dims, concave) if dsf is None else dsf
		self.args = args
		self.nesting_list=nesting_list
		self.nesting = True if nesting_list else False
		if not (args['model_type'] == 'deepset' or args['model_type'] == 'set_transformer'):
			print("Performing absolute projection for weights for instantiation.")	
			self.abs_project() 

	def set_device(self, device):
		self.device=device

	def project(self):
		if hasattr(self.dsf, 'project'):
			return self.dsf.project()
		for p in self.dsf.parameters():
			p.data.clamp_(min=0.0)

	def abs_project(self):
		if hasattr(self.dsf, 'abs_project'):
			return self.dsf.abs_project()
		if hasattr(self.dsf, 'project'):
			return self.dsf.project()
		for p in self.dsf.parameters():
			p.data = torch.abs(p.data)

	def compute_regularizers(self, embeddings):
		loss_activity_norm = (embeddings.norm(p=2, dim=-1)**2).mean() if self.args['activity_reg_coefficient']>0. else 0.
		return loss_activity_norm

	def compute_nested_peripteral_loss(self, h_A_B_full, nested_margins, bsz=None, feedback=False, N=None):
		peripteral_loss=[]
		disagreements=[]
		accuracies=[]
		if feedback:
			accuracies_sf = []

		gate=[]	
		if bsz:
			self.bsz=bsz
		
		dims = h_A_B_full.shape[-1]

		if self.args['baseline'] == 'ranknet':
			nested_margins = (torch.sign(nested_margins)+1)//2
		
		if self.args['baseline'] == 'DSPN_no_GPC':
			old_shape = nested_margins.shape
			nested_margins = torch.sign(nested_margins)
			assert nested_margins.shape == old_shape

		start=0
		sum_f, m_A_B_batch = 0., 0.	
		for i, nesting in enumerate(self.nesting_list):
			end=nesting
			h_A_B = h_A_B_full[:, start:end, :] 
			sum_f+=h_A_B.sum(dim=1) 
			
			f_M_E_batch = self.dsf(sum_f).squeeze()

			SCMI_A_B_batch =  - f_M_E_batch

			SCMI_A_batch = SCMI_A_B_batch[:self.bsz]
			SCMI_B_batch = SCMI_A_B_batch[self.bsz:]

			if self.args['baseline'] == 'DSPN' or self.args['baseline'] == 'DSPN_no_GPC':
				batch_peripteral_cost, mean_gate = peripteral_cost(SCMI_B_batch, SCMI_A_batch, nested_margins[:, i],  beta=self.args['beta'], tau=self.args['tau'], reduction='mean')
			elif self.args['baseline'] == 'max_margin':
				batch_peripteral_cost, mean_gate = max_margin_loss(SCMI_B_batch, SCMI_A_batch, nested_margins[:, i],  beta=self.args['beta'])
			elif self.args['baseline'] == 'ranknet':
				batch_peripteral_cost = F.binary_cross_entropy(torch.sigmoid(SCMI_A_batch - SCMI_B_batch), nested_margins[:, i]); mean_gate = torch.zeros(1)
			elif self.args['baseline'] == 'regression':
				batch_peripteral_cost = F.mse_loss(SCMI_A_batch - SCMI_B_batch, nested_margins[:, i]); mean_gate = torch.zeros(1)
				
			peripteral_loss.append(batch_peripteral_cost)
			if feedback:
				accuracies.append(compute_accuracy(SCMI_A_batch[:N], SCMI_B_batch[:N], nested_margins[:N, i]))
				accuracies_sf.append(compute_accuracy(SCMI_A_batch[N:], SCMI_B_batch[N:], nested_margins[N:, i]))
			else:
				accuracies.append(compute_accuracy(SCMI_A_batch, SCMI_B_batch, nested_margins[:, i]))
			gate.append(mean_gate)

			start=end

		
		if feedback:
			return torch.stack(peripteral_loss), (accuracies, accuracies_sf), torch.stack(gate)
		else:
			return torch.stack(peripteral_loss), accuracies, torch.stack(gate)

	def compute_nested_peripteral_loss_set_transformer(self, h_A_B_full, nested_margins, bsz=None, feedback=False, N=None):
		peripteral_loss=[]
		accuracies=[]
		if feedback:
			accuracies_sf = []

		gate=[]	
		if bsz:
			self.bsz=bsz
		
		if self.args['baseline'] == 'ranknet':
			nested_margins = (torch.sign(nested_margins)+1)//2

		if self.args['baseline'] == 'DSPN_no_GPC':
			nested_margins = torch.sign(nested_margins)


		_, m_A_B_batch = 0., 0.	
		for i, nesting in enumerate(self.nesting_list):

			end=nesting
			h_A_B = h_A_B_full[:, :end, :] 
			f_M_E_batch = self.dsf(h_A_B).squeeze()

			SCMI_A_B_batch =  - f_M_E_batch

			SCMI_A_batch = SCMI_A_B_batch[:self.bsz]
			SCMI_B_batch = SCMI_A_B_batch[self.bsz:]

			if self.args['baseline'] == 'DSPN' or self.args['baseline'] == 'DSPN_no_GPC':
				batch_peripteral_cost, mean_gate = peripteral_cost(SCMI_B_batch, SCMI_A_batch, nested_margins[:, i],  beta=self.args['beta'], tau=self.args['tau'], reduction='mean')
			elif self.args['baseline'] == 'max_margin':
				batch_peripteral_cost, mean_gate = max_margin_loss(SCMI_B_batch, SCMI_A_batch, nested_margins[:, i],  beta=self.args['beta'])
			elif self.args['baseline'] == 'ranknet':
				batch_peripteral_cost = F.binary_cross_entropy(torch.sigmoid(SCMI_A_batch - SCMI_B_batch), nested_margins[:, i]); mean_gate = torch.zeros(1)
			elif self.args['baseline'] == 'regression':
				batch_peripteral_cost = F.mse_loss(SCMI_A_batch - SCMI_B_batch, nested_margins[:, i]); mean_gate = torch.zeros(1)
				
			peripteral_loss.append(batch_peripteral_cost)
			if feedback:
				accuracies.append(compute_accuracy(SCMI_A_batch[:N], SCMI_B_batch[:N], nested_margins[:N, i]))
				accuracies_sf.append(compute_accuracy(SCMI_A_batch[N:], SCMI_B_batch[N:], nested_margins[N:, i]))
			else:
				accuracies.append(compute_accuracy(SCMI_A_batch, SCMI_B_batch, nested_margins[:, i]))
			gate.append(mean_gate)

		if feedback:
			return torch.stack(peripteral_loss), (accuracies, accuracies_sf), torch.stack(gate)
		else:
			return torch.stack(peripteral_loss), accuracies, torch.stack(gate)

	def compute_loss(self, D_A, D_B, margin, compute_reg=True, feedback=False,N=None):
		self.bsz = D_A.shape[0]
		dim = 1280 if self.args['dset'] in ['IN100', 'CIFAR100'] else 512
		D_A = D_A.view(-1, dim); D_B = D_B.view(-1, dim)

		concat = torch.cat([D_A, D_B], dim=0)
		
		_, embeddings = self.feat(concat.float(), last=True)
		embeddings = F.relu(embeddings)
		dims = embeddings.shape[-1]

		if compute_reg:	
			loss_activity_norm = self.compute_regularizers(embeddings) 
		else:
			loss_activity_norm = 0.
		

		peripteral_loss, set_classification_accuracy, gate = self.compute_nested_peripteral_loss(embeddings.view(-1,self.args['set_size'], dims), margin, feedback=feedback, N=N)

		return peripteral_loss, loss_activity_norm, set_classification_accuracy, gate
	
	def compute_loss_set_transformer(self, D_A, D_B, margin, compute_reg=True, feedback=False, N=None):
		self.bsz = D_A.shape[0]
		dim = 1280 if self.args['dset'] in ['IN100', 'CIFAR100'] else 512
		concat = torch.cat([D_A, D_B], dim=0)
		
		_, embeddings = self.feat(concat.float(), last=True)
		embeddings = F.relu(embeddings)
		dims = embeddings.shape[-1]
		if compute_reg:	
			loss_activity_norm = self.compute_regularizers(embeddings.view(-1, dims)) 
		else:
			loss_activity_norm = 0.  
		

		peripteral_loss, set_classification_accuracy, gate = self.compute_nested_peripteral_loss_set_transformer(embeddings, margin, feedback=feedback, N=N)

		return peripteral_loss, loss_activity_norm, set_classification_accuracy, gate
	
	def cross_loss(self, emb_A, emb_B, margin):
		emb_A1, emb_A2 = emb_A[0], emb_A[1]
		emb_B1, emb_B2 = emb_B[0], emb_B[1]
		dims = emb_A1.shape[-1]
		peripteral_loss_11, set_classification_accuracy_1, gate = self.compute_nested_peripteral_loss(torch.cat([emb_A1, emb_B1]).view(-1,self.args['set_size'], dims), margin)
		peripteral_loss_12, set_classification_accuracy_2, _ = self.compute_nested_peripteral_loss(torch.cat([emb_A1, emb_B2]).view(-1,self.args['set_size'], dims), margin)
		peripteral_loss_21, set_classification_accuracy_3, _ = self.compute_nested_peripteral_loss(torch.cat([emb_A2, emb_B1]).view(-1,self.args['set_size'], dims), margin)
		peripteral_loss_22, set_classification_accuracy_4, _ = self.compute_nested_peripteral_loss(torch.cat([emb_A2, emb_B2]).view(-1,self.args['set_size'], dims), margin)
		
		cross_modality_loss = (peripteral_loss_11 + peripteral_loss_12 + peripteral_loss_21 + peripteral_loss_22)/4.
		set_classification_accuracy = (torch.Tensor(set_classification_accuracy_1) + torch.Tensor(set_classification_accuracy_2) + torch.Tensor(set_classification_accuracy_3) + torch.Tensor(set_classification_accuracy_4))/4
		set_classification_accuracy = set_classification_accuracy.tolist()
		return cross_modality_loss, set_classification_accuracy, gate

	def roof_consistency_loss(self, emb_A, emb_B, margin):
		emb_A1, emb_A2 = emb_A[0], emb_A[1]
		emb_B1, emb_B2 = emb_B[0], emb_B[1]
		dims = emb_A1.shape[-1]
		zero_margin = torch.zeros_like(margin) 
		peripteral_loss_A, _, _ = self.compute_nested_peripteral_loss((torch.cat([emb_A1, emb_A2], dim=0)).view(-1,self.args['set_size'], dims), zero_margin)
		peripteral_loss_B, _, _ = self.compute_nested_peripteral_loss((torch.cat([emb_B1, emb_B2], dim=0)).view(-1,self.args['set_size'], dims), zero_margin)

		within_modality_loss = peripteral_loss_A + peripteral_loss_B
		return within_modality_loss

	def gain_loss(self, emb_A, emb_B):  
		dims = emb_A[0].shape[-1]
		emb_A1, emb_A2 = emb_A[0].view(-1,self.args['set_size'], dims), emb_A[1].view(-1,self.args['set_size'], dims)
		emb_B1, emb_B2 = emb_B[0].view(-1,self.args['set_size'], dims), emb_B[1].view(-1,self.args['set_size'], dims) 
		emb_A1B1, emb_A2B2 = torch.cat([emb_A1, emb_B1], 0), torch.cat([emb_A2, emb_B2], 0)
		start = 0; running_sum_1, running_sum_2, running_sum_joint= 0., 0., 0.
		nested_gain_losses = []
		for i, nesting in enumerate(self.nesting_list):
			end = nesting
			running_sum_1 += emb_A1B1[:, start:end, :].sum(1)
			running_sum_2 += emb_A2B2[:, start:end, :].sum(1)
			running_sum_joint = running_sum_1+running_sum_2
			nested_gain_losses.append(F.mse_loss(self.dsf(running_sum_joint), (self.dsf(running_sum_1) + self.dsf(running_sum_2))/2))
			start = end

		return torch.stack(nested_gain_losses)

	def singleton_roof_consistency(self, emb_A, emb_B, compute_gains=True):  
		dims = emb_A[0].shape[-1]
		emb_A1, emb_A2 = emb_A[0].view(-1, dims), emb_A[1].view(-1, dims)
		emb_B1, emb_B2 = emb_B[0].view(-1, dims), emb_B[1].view(-1, dims)
		emb_A1B1, emb_A2B2 = torch.cat([emb_A1, emb_B1], 0), torch.cat([emb_A2, emb_B2], 0)
		DSF_A1B1, DSF_A2B2 = self.dsf(emb_A1B1), self.dsf(emb_A2B2)
		DSF_A1cupA2_B1cupB2 = self.dsf(emb_A1B1 + emb_A2B2)
		
		singleton_roof_consistency_loss = F.mse_loss(self.dsf(emb_A1B1), self.dsf(emb_A2B2))
		singleton_gain_loss = F.mse_loss(DSF_A1cupA2_B1cupB2, (DSF_A1B1+DSF_A2B2)/2)  if compute_gains else 0.

		return singleton_roof_consistency_loss, singleton_gain_loss
	
	def singleton_roof_consistency_set_transformer(self, emb_A, emb_B, compute_gains=True):  
		dims = emb_A[0].shape[-1]
		emb_A1, emb_A2 = emb_A[0].view(-1, 1, dims), emb_A[1].view(-1, 1, dims)
		emb_B1, emb_B2 = emb_B[0].view(-1, 1, dims), emb_B[1].view(-1, 1, dims)


		emb_A1B1, emb_A2B2 = torch.cat([emb_A1, emb_B1], 0), torch.cat([emb_A2, emb_B2], 0)
		DSF_A1B1, DSF_A2B2 = self.dsf(emb_A1B1), self.dsf(emb_A2B2)
		DSF_A1B1, DSF_A2B2 = DSF_A1B1.squeeze(), DSF_A2B2.squeeze()

		DSF_A1cupA2_B1cupB2 = self.dsf(torch.cat([emb_A1B1 , emb_A2B2], dim=1))
		
		singleton_roof_consistency_loss = F.mse_loss(DSF_A1B1, DSF_A2B2)

		singleton_gain_loss = F.mse_loss(DSF_A1cupA2_B1cupB2, (DSF_A1B1+DSF_A2B2)/2)  if compute_gains else 0.

		return singleton_roof_consistency_loss, singleton_gain_loss
		
	def compute_nested_peripteral_loss_with_augmentations(self, emb_A, emb_B1, FL_M, FL_E):  
		dims = emb_A[0].shape[-1]
		emb_A1, emb_A2 = emb_A[0].view(-1,self.args['set_size'], dims), emb_A[1].view(-1,self.args['set_size'], dims)
		emb_B1= emb_B1.view(-1,self.args['set_size'], dims)
		nesting_to_idx  = {e:i for i, e in enumerate(self.nesting_list)}
		start=0; peripteral_loss=[]
		m_E, m_M = 0., 0.	
		for i, nesting in enumerate(self.nesting_list):
			if not (nesting*2 in nesting_to_idx.keys()):
				break
			
			end=nesting; end_2 = nesting*2
			m_E += emb_B1[:, start:end_2, :].sum(1)
			m_M += emb_A1[:, start:end, :].sum(1) + emb_A2[:, start:end, :].sum(1)
			
			SCMI_A_B_batch =  - self.dsf(torch.cat([m_M, m_E])).squeeze()

			SCMI_A_batch = SCMI_A_B_batch[:self.bsz]
			SCMI_B_batch = SCMI_A_B_batch[self.bsz:]

			margin = FL_E[:, nesting_to_idx[2*nesting]] - FL_M[:, i]

			if self.args['baseline'] == 'ranknet':
				margin = (torch.sign(margin) + 1)//2
			
			if self.args['baseline'] == 'DSPN_no_GPC':
				margin = torch.sign(margin)

			if self.args['baseline'] == 'DSPN' or self.args['baseline'] == 'DSPN_no_GPC':
				batch_peripteral_cost, mean_gate = peripteral_cost(SCMI_B_batch, SCMI_A_batch, margin,  beta=self.args['beta'], tau=self.args['tau'], reduction='mean')
			elif self.args['baseline'] == 'max_margin':
				batch_peripteral_cost, mean_gate = max_margin_loss(SCMI_B_batch, SCMI_A_batch, margin,  beta=self.args['beta'])
			elif self.args['baseline'] == 'ranknet':
				batch_peripteral_cost = F.binary_cross_entropy(torch.sigmoid(SCMI_A_batch - SCMI_B_batch), margin); mean_gate = torch.zeros(1)
			elif self.args['baseline'] == 'regression':
				batch_peripteral_cost = F.mse_loss(SCMI_A_batch - SCMI_B_batch, margin); mean_gate = torch.zeros(1)
				
			peripteral_loss.append(batch_peripteral_cost)

		return torch.stack(peripteral_loss).sum()


	def compute_nested_peripteral_loss_with_augmentations_set_transformer(self, emb_A, emb_B1, FL_M, FL_E):  
		dims = emb_A[0].shape[-1]
		emb_A1, emb_A2 = emb_A[0].view(-1,self.args['set_size'], dims), emb_A[1].view(-1,self.args['set_size'], dims)
		emb_B1= emb_B1.view(-1,self.args['set_size'], dims)
		nesting_to_idx  = {e:i for i, e in enumerate(self.nesting_list)}
		start=0; peripteral_loss=[]
		m_E, m_M = 0., 0.	
		for i, nesting in enumerate(self.nesting_list):
			if not (nesting*2 in nesting_to_idx.keys()):
				break
			
			end=nesting; end_2 = nesting*2
			SCMI_B_batch = - self.dsf(emb_B1[:, :end_2, :])
			SCMI_A_batch = - self.dsf(torch.cat([emb_A1[:, :end, :],emb_A2[:, :end, :]], dim=1))
			margin = FL_E[:, nesting_to_idx[2*nesting]] - FL_M[:, i]

			if self.args['baseline'] == 'ranknet':
				margin  = (torch.sign(margin)+1)//2
			
			if self.args['baseline'] == 'DSPN_no_GPC':
				margin  = torch.sign(margin)

			if self.args['baseline'] == 'DSPN' or self.args['baseline'] == 'DSPN_no_GPC':
				batch_peripteral_cost, mean_gate = peripteral_cost(SCMI_B_batch, SCMI_A_batch, margin, beta=self.args['beta'], tau=self.args['tau'], reduction='mean')
			elif self.args['baseline'] == 'max_margin':
				batch_peripteral_cost, mean_gate = max_margin_loss(SCMI_B_batch, SCMI_A_batch, margin, beta=self.args['beta'])
			elif self.args['baseline'] == 'ranknet':
				batch_peripteral_cost = F.binary_cross_entropy(torch.sigmoid(SCMI_A_batch - SCMI_B_batch), margin); mean_gate = torch.zeros(1)
			elif self.args['baseline'] == 'regression':
				batch_peripteral_cost = F.mse_loss(SCMI_A_batch - SCMI_B_batch, margin); mean_gate = torch.zeros(1)
				
			peripteral_loss.append(batch_peripteral_cost)

		return torch.stack(peripteral_loss).sum()

	def compute_loss_augmented(self, D_A, D_B, margin, FL_M, FL_E, compute_reg=True, feedback=False,N=None):
		self.bsz = D_A[0].shape[0] 
		dim = 1280 if self.args['dset'] in ['IN100', 'CIFAR100'] else 512
		D_A1 = D_A[0].view(-1, dim); D_B1 = D_B[0].view(-1, dim)
		D_A2 = D_A[1].view(-1, dim); D_B2 = D_B[1].view(-1, dim)
		size_temp = len(D_A1)

		emb_A1A2B1B2 = F.relu(self.feat(torch.cat([D_A1, D_A2, D_B1, D_B2], 0), last=True)[1])
		
		if compute_reg:	
			loss_activity_norm = self.compute_regularizers(emb_A1A2B1B2) 
		else:
			loss_activity_norm = 0.  

		emb_A1, emb_A2, emb_B1, emb_B2 = emb_A1A2B1B2[:size_temp],emb_A1A2B1B2[size_temp:2*size_temp], emb_A1A2B1B2[2*size_temp:3*size_temp], emb_A1A2B1B2[3*size_temp:]

		cross_modality_loss, set_classification_accuracy, gate = self.cross_loss([emb_A1, emb_A2], [emb_B1, emb_B2], margin)
		roof_consistency_loss = self.roof_consistency_loss([emb_A1, emb_A2], [emb_B1, emb_B2], margin)
		
		gain_loss = self.gain_loss([emb_A1, emb_A2], [emb_B1, emb_B2])
		singleton_roof_consistency_loss, singleton_gain_loss = self.singleton_roof_consistency([emb_A1, emb_A2], [emb_B1, emb_B2])
		nested_peripteral_loss_with_augmentation = self.compute_nested_peripteral_loss_with_augmentations([emb_A1, emb_A2], emb_B1, FL_M, FL_E)

		return cross_modality_loss, roof_consistency_loss, gain_loss, singleton_roof_consistency_loss, singleton_gain_loss,nested_peripteral_loss_with_augmentation, set_classification_accuracy, gate, loss_activity_norm 

	def cross_loss_nview(self, emb_A, emb_B, margin, chunk_size):

		dims = emb_A.shape[-1]
		cross_loss = 0

		peripteral_loss_11, set_classification_accuracy, gate = self.compute_nested_peripteral_loss(torch.cat([emb_A[:chunk_size], emb_B[:chunk_size]]).view(-1,self.args['set_size'], dims), margin)
		set_classification_accuracy = torch.Tensor(set_classification_accuracy)
		cross_loss = peripteral_loss_11

		for i in range(1,self.nviews):

			peripteral_loss_1i, set_acc_1i, _ = self.compute_nested_peripteral_loss(torch.cat([emb_A[:chunk_size], emb_B[i*chunk_size:(i+1)*chunk_size]]).view(-1,self.args['set_size'], dims), margin)
			peripteral_loss_i1, set_acc_i1, _ = self.compute_nested_peripteral_loss(torch.cat([emb_A[i*chunk_size:(i+1)*chunk_size], emb_B[:chunk_size]]).view(-1,self.args['set_size'], dims), margin)
			peripteral_loss_ii, set_acc_ii, _ = self.compute_nested_peripteral_loss(torch.cat([emb_A[i*chunk_size:(i+1)*chunk_size], emb_B[i*chunk_size:(i+1)*chunk_size]]).view(-1,self.args['set_size'], dims), margin)


			set_classification_accuracy = set_classification_accuracy + (torch.Tensor(set_acc_1i) + torch.Tensor(set_acc_i1) + torch.Tensor(set_acc_ii))
		
			cross_loss = cross_loss + (peripteral_loss_1i + peripteral_loss_i1 + peripteral_loss_ii)
	
		set_classification_accuracy = set_classification_accuracy/(1 + 3*(self.nviews-1))
		set_classification_accuracy = set_classification_accuracy.tolist()


		cross_loss = cross_loss/(1 + 3*(self.nviews-1))
		return cross_loss, set_classification_accuracy, gate

	def gain_loss_nview(self, emb_A, emb_B, chunk_size):
		gain_loss = 0. 
		for i in range(self.args['n_views']-1):
			gain_loss += self.gain_loss([emb_A[i*chunk_size:(i+1)*chunk_size], emb_A[(i+1)*chunk_size:(i+2)*chunk_size]], [emb_B[i*chunk_size:(i+1)*chunk_size], emb_B[(i+1)*chunk_size:(i+2)*chunk_size]])
		gain_loss = gain_loss/(self.args['n_views']-1)
		return gain_loss 

	def singleton_roof_consistency_nview(self, emb_A, emb_B, chunk_size, compute_gains=True):
		singleton_roof_consistency_loss, singleton_gain_loss = 0., 0. 
		for i in range(self.args['n_views']-1):
			loss = self.singleton_roof_consistency([emb_A[i*chunk_size:(i+1)*chunk_size], emb_A[(i+1)*chunk_size:(i+2)*chunk_size]], [emb_B[i*chunk_size:(i+1)*chunk_size], emb_B[(i+1)*chunk_size:(i+2)*chunk_size]], compute_gains)
			singleton_roof_consistency_loss+= loss[0]
			singleton_gain_loss+= loss[1]
		singleton_roof_consistency_loss/=(self.args['n_views']-1)
		singleton_gain_loss/=(self.args['n_views']-1)
		return singleton_roof_consistency_loss, singleton_gain_loss


	def singleton_roof_consistency_nview_set_transformer(self, emb_A, emb_B, chunk_size, compute_gains=True):
		singleton_roof_consistency_loss, singleton_gain_loss = 0., 0. 
		for i in range(self.args['n_views']-1):
			loss = self.singleton_roof_consistency_set_transformer([emb_A[i*chunk_size:(i+1)*chunk_size], emb_A[(i+1)*chunk_size:(i+2)*chunk_size]], [emb_B[i*chunk_size:(i+1)*chunk_size], emb_B[(i+1)*chunk_size:(i+2)*chunk_size]], compute_gains)
			singleton_roof_consistency_loss+= loss[0]
			singleton_gain_loss+= loss[1]
		singleton_roof_consistency_loss/=(self.args['n_views']-1)
		singleton_gain_loss/=(self.args['n_views']-1)
		return singleton_roof_consistency_loss, singleton_gain_loss

	def roof_consistency_loss_nview(self, emb, margin, chunk_size):

		dims = emb.shape[-1]

		zero_margin = torch.zeros_like(margin) 
		loss = 0

		for i in range(1,self.nviews):

			peripteral_loss, _, _ = self.compute_nested_peripteral_loss(torch.cat([emb[:chunk_size], emb[i*chunk_size:(i+1)*chunk_size]]).view(-1,self.args['set_size'], dims), zero_margin)

			if i == 1:
				loss = peripteral_loss
			else:
				loss = loss + peripteral_loss


		loss = loss/(self.nviews-1)
		return loss


	def roof_consistency_loss_nview_set_transformer(self, emb, margin, chunk_size):

		dims = emb.shape[-1]

		zero_margin = torch.zeros_like(margin) 
		loss = 0

		for i in range(1,self.nviews):

			peripteral_loss, _, _ = self.compute_nested_peripteral_loss_set_transformer(torch.cat([emb[:chunk_size], emb[i*chunk_size:(i+1)*chunk_size]]).view(-1,self.args['set_size'], dims), zero_margin)

			if i == 1:
				loss = peripteral_loss
			else:
				loss = loss + peripteral_loss


		loss = loss/(self.nviews-1)
		return loss

	def compute_loss_augmented_nviews(self, D_A, D_B, margin, FL_M=0, FL_E=0, compute_reg=True, feedback=False,N=None, compute_gains=False):
		assert len(D_A) == len(D_B)
		dim = 1280 if self.args['dset'] in ['IN100', 'CIFAR100'] else 512
		size_temp = len(D_A[0].view(-1, dim))
		self.bsz = D_A[0].shape[0] 
		self.nviews = len(D_A)
		
		D_A = torch.cat(D_A, dim=0); D_A = D_A.view(-1, dim); D_B = torch.cat(D_B, dim=0); D_B = D_B.view(-1, dim); temp = len(D_A)
		emb_A_B = F.relu(self.feat(torch.cat([D_A, D_B], 0), last=True)[1]) 	
		emb_A, emb_B = emb_A_B[:temp], emb_A_B[temp:]


		if compute_reg:	
			loss_activity_norm = self.compute_regularizers(emb_A_B) 
		else:
			loss_activity_norm, = 0.


		cross_modality_loss, set_classification_accuracy, gate = self.cross_loss_nview(emb_A, emb_B, margin, size_temp)
		roof_consistency_loss = (self.roof_consistency_loss_nview(emb_A, margin, size_temp) + self.roof_consistency_loss_nview(emb_B, margin, size_temp))/2
		
		

		if  compute_gains:
			gain_loss = self.gain_loss_nview(emb_A, emb_B, size_temp)
		else:
			gain_loss = 0.
		singleton_roof_consistency_loss, singleton_gain_loss = self.singleton_roof_consistency_nview(emb_A, emb_B, size_temp, compute_gains)
		
		nested_peripteral_loss_with_augmentation = self.compute_nested_peripteral_loss_with_augmentations([emb_A[:size_temp], emb_A[size_temp:2*size_temp]], emb_B[:size_temp], FL_M, FL_E)

		return cross_modality_loss, roof_consistency_loss, gain_loss, singleton_roof_consistency_loss, singleton_gain_loss,nested_peripteral_loss_with_augmentation, set_classification_accuracy, gate, loss_activity_norm 


	def compute_loss_augmented_nviews_set_transformer(self, D_A, D_B, margin, FL_M=0, FL_E=0, compute_reg=True, feedback=False,N=None, compute_gains=False):
		assert len(D_A) == len(D_B)
		dim = 1280 if self.args['dset'] in ['IN100', 'CIFAR100'] else 512
		size_temp = len(D_A[0].view(-1, dim))
		self.bsz = D_A[0].shape[0] 
		self.nviews = len(D_A)
		dim=2048


		D_A = torch.cat(D_A, dim=0); D_B = torch.cat(D_B, dim=0); temp = len(D_A)
		emb_A_B = F.relu(self.feat(torch.cat([D_A, D_B], 0), last=True)[1]) 	
		emb_A, emb_B = emb_A_B[:temp], emb_A_B[temp:]
		emb_A = emb_A.view(-1, dim)
		emb_B = emb_B.view(-1, dim)

		if compute_reg:	
			loss_activity_norm = self.compute_regularizers(emb_A_B.view(-1, dim)) 
		else:
			loss_activity_norm = 0.  


		print(emb_A.shape, emb_B.shape)
		cross_modality_loss, set_classification_accuracy, gate = self.cross_loss_nview_set_transformer(emb_A, emb_B, margin, size_temp)
		roof_consistency_loss = (self.roof_consistency_loss_nview_set_transformer(emb_A, margin, size_temp) + self.roof_consistency_loss_nview_set_transformer(emb_B, margin, size_temp))/2
		
		

		if  compute_gains:
			gain_loss = self.gain_loss_nview(emb_A, emb_B, size_temp)
		else:
			gain_loss = 0.
		singleton_roof_consistency_loss, singleton_gain_loss = self.singleton_roof_consistency_nview_set_transformer(emb_A, emb_B, size_temp, compute_gains)
		
		nested_peripteral_loss_with_augmentation = self.compute_nested_peripteral_loss_with_augmentations_set_transformer([emb_A[:size_temp], emb_A[size_temp:2*size_temp]], emb_B[:size_temp], FL_M, FL_E)

		return cross_modality_loss, roof_consistency_loss, gain_loss, singleton_roof_consistency_loss, singleton_gain_loss,nested_peripteral_loss_with_augmentation, set_classification_accuracy, gate, loss_activity_norm

	def cross_loss_nview_set_transformer(self, emb_A, emb_B, margin, chunk_size):

		dims = emb_A.shape[-1]
		cross_loss = 0

		peripteral_loss_11, set_classification_accuracy, gate = self.compute_nested_peripteral_loss_set_transformer(torch.cat([emb_A[:chunk_size], emb_B[:chunk_size]]).view(-1,self.args['set_size'], dims), margin)
		set_classification_accuracy = torch.Tensor(set_classification_accuracy)
		cross_loss = peripteral_loss_11

		for i in range(1,self.nviews):

			peripteral_loss_1i, set_acc_1i, _ = self.compute_nested_peripteral_loss_set_transformer(torch.cat([emb_A[:chunk_size], emb_B[i*chunk_size:(i+1)*chunk_size]]).view(-1,self.args['set_size'], dims), margin)
			peripteral_loss_i1, set_acc_i1, _ = self.compute_nested_peripteral_loss_set_transformer(torch.cat([emb_A[i*chunk_size:(i+1)*chunk_size], emb_B[:chunk_size]]).view(-1,self.args['set_size'], dims), margin)
			peripteral_loss_ii, set_acc_ii, _ = self.compute_nested_peripteral_loss_set_transformer(torch.cat([emb_A[i*chunk_size:(i+1)*chunk_size], emb_B[i*chunk_size:(i+1)*chunk_size]]).view(-1,self.args['set_size'], dims), margin)


			set_classification_accuracy = set_classification_accuracy + (torch.Tensor(set_acc_1i) + torch.Tensor(set_acc_i1) + torch.Tensor(set_acc_ii))
		
			cross_loss = cross_loss + (peripteral_loss_1i + peripteral_loss_i1 + peripteral_loss_ii)
	
		set_classification_accuracy = set_classification_accuracy/(1 + 3*(self.nviews-1))
		set_classification_accuracy = set_classification_accuracy.tolist()


		cross_loss = cross_loss/(1 + 3*(self.nviews-1))
		return cross_loss, set_classification_accuracy, gate
