import torch
import numpy as np 
from torch.utils.data import Dataset
import warnings
warnings.filterwarnings("ignore") 


class SetDataset(Dataset):
	def __init__(self, X, Y, D_M_idx_full, D_E_idx_full, margin_full, FL_M=None, FL_E=None, nested=False, device=None, n_views=1, **kwargs):
		self.X = X
		self.Y = Y
		self.D_M_idx_full = D_M_idx_full
		self.D_E_idx_full = D_E_idx_full
		self.margin_full = margin_full
		self.FL_M = FL_M
		self.FL_E = FL_E
		self.nested=nested
		self.num_nesting = len(self.margin_full) if self.nested else 1
		self.device = device
		self.n_views = n_views
		self.num_aug = self.X.shape[-1]

		if not (self.device is None):
			self.X = (self.X).to(device, non_blocking =True)
			self.Y = (self.Y).to(device, non_blocking =True)
			self.margin_full = (self.margin_full).to(device, non_blocking =True)

			if not (FL_M is None):	
				self.FL_M = (self.FL_M).to(device, non_blocking =True)
				self.FL_E = (self.FL_E).to(device, non_blocking =True)

			self.D_M_idx_full =  (self.D_M_idx_full).to(device, non_blocking =True)
			self.D_E_idx_full =  (self.D_E_idx_full).to(device, non_blocking =True)
			
	def __len__(self):
		return len(self.D_M_idx_full)

	def update_data(self, D_M_idx_full, D_E_idx_full, margin_full, FL_M=None, FL_E=None):
		self.margin_full = (margin_full).to(self.device, non_blocking =True)
		if not (FL_M  is None):
			self.FL_M = FL_M.to(self.device, non_blocking =True)
			self.FL_E = FL_E.to(self.device, non_blocking =True)
		self.D_M_idx_full =  (D_M_idx_full).to(self.device, non_blocking =True)
		self.D_E_idx_full =  (D_E_idx_full).to(self.device, non_blocking =True)

	
	def __getitem__(self, idx):
		D_A_idx = self.D_M_idx_full[idx].type(torch.LongTensor)
		D_B_idx = self.D_E_idx_full[idx].type(torch.LongTensor)

		D_A = self.X[D_A_idx]
		D_B = self.X[D_B_idx]
		Y_A = self.Y[D_A_idx]
		Y_B = self.Y[D_B_idx]

		margin = self.margin_full[:, idx]
		
		if self.n_views == 1:
			D_A = [D_A[:, :, 0]]
			D_B = [D_B[:, :, 0]]
		else:
			if self.num_aug == self.n_views:
				D_A = torch.permute(D_A, (2, 0, 1))
				D_B = torch.permute(D_B, (2, 0, 1)) 
			else:
				D_A_ = torch.permute(D_A[:, :, 1:], (2, 0, 1))
				D_B_ = torch.permute(D_B[:, :, 1:], (2, 0, 1))
				ind = np.random.choice(self.num_aug-1, self.n_views-1, replace=False)
				
				if self.n_views==2:
					ind = ind[0]				
					D_A = torch.stack([D_A[:, :, 0], D_A_[ind]], dim=0)
					D_B = torch.stack([D_B[:, :, 0], D_B_[ind]], dim=0)
				else:
					D_A = torch.cat([D_A[:, :, 0].unsqueeze(0), D_A_[ind]], dim=0)
					D_B = torch.cat([D_B[:, :, 0].unsqueeze(0), D_B_[ind]], dim=0)
        

			D_A = [D_A[i] for i in range(len(D_A))]
			D_B = [D_B[i] for i in range(len(D_B))]


		if self.FL_M is None:	
			return D_A_idx, D_B_idx, D_A, D_B, Y_A, Y_B, margin, None, None
		else:
			FL_M, FL_E = self.FL_M[:, idx], self.FL_E[:, idx]
			return D_A_idx, D_B_idx, D_A, D_B, Y_A, Y_B, margin, FL_M, FL_E
