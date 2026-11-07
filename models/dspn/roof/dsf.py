import torch
import torch.nn as nn
import torch.nn.functional as F
import math 

class MAB(nn.Module):
	
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O

class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)

class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)

class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)

class log1p(nn.Module):
	def __init__(self, num_logs=1):
		super(log1p, self).__init__()
		self.num_logs=num_logs

	def forward(self, x):
		for i in range(self.num_logs):
			x = torch.log(1+x)
		return x

class linear(nn.Module):
	def __init__(self):
		super(linear, self).__init__()
	
	def forward(self, x):
		return x

class tanh(nn.Module):
	def __init__(self):
		super(tanh, self).__init__()

	def forward(self, x):
		return F.tanh(x)

class exp(nn.Module):
	def __init__(self):
		super(exp, self).__init__()

	def forward(self, x):
		return 1 - torch.exp(-x)

class sqrt(nn.Module):
	def __init__(self):
		super(sqrt, self).__init__()

	def forward(self, x):
		return torch.sqrt(x)

class pow_(nn.Module):
	def __init__(self, p=0.9):
		super(pow_, self).__init__()
		self.p=p

	def forward(self, x):
		return torch.pow(x, self.p)
		
CONCAVE_FN_INITIALIZER={
	'linear': linear,
	'log1p':log1p,
	'tanh':tanh,
	'exp':exp,
	'sqrt':sqrt,
	'pow_':pow_,
	'relu': nn.ReLU	
}

class concave_fn(nn.Module):
	def __init__(self, concave_fn_list=['log1p'], num_logs=1, p=0.9, dim=None):
		super(concave_fn, self).__init__()
		self.concave_fn_list=[]
		for x in concave_fn_list:
			if x == 'log1p':
				obj = CONCAVE_FN_INITIALIZER[x](num_logs)
			elif x=='pow':
				obj = CONCAVE_FN_INITIALIZER[x](p)
			else:
				obj = CONCAVE_FN_INITIALIZER[x]()
			self.concave_fn_list.append(obj)


	def forward(self, x):
		outputs = []
		for fn in self.concave_fn_list:
			outputs.append(fn(x))

		if len(self.concave_fn_list)==1:
			return outputs[0]
		else:
			return torch.stack(outputs, dim=-1)

class mixing_linear_layer(nn.Module):
	def __init__(self, in_features, out_features, num_concave_prev=1):
		super(mixing_linear_layer, self).__init__()
		self.mixing_layer = nn.Linear(num_concave_prev*in_features, out_features, bias=False)

	def forward(self, x):
		bsz = x.shape[0]
		x = x.view(bsz, -1)
		x = self.mixing_layer(x)
		return x

class DSF(nn.Module):
	def __init__(self, out_dims=[512], concave=['log1p']):
		super(DSF, self).__init__()
		self.concave = concave
		self.out_dims = out_dims+[1]
		module_list=[]

		for i in range(len(self.out_dims)-1):
			module_list.append(nn.ReLU())
			num_concave_prev=None
			if type(concave[0])==list:
				module_list.append(concave_fn(concave[i], dim=self.out_dims[i]))
				num_concave_prev=len(concave[i])
			else:
				module_list.append(concave_fn(concave, dim=self.out_dims[i]))
				num_concave_prev=len(concave)

			module_list.append(mixing_linear_layer(self.out_dims[i], self.out_dims[i+1], num_concave_prev))


		self.dsf = nn.Sequential(*module_list)

	def forward(self, inputs):
		return self.dsf(inputs)


class SelfAttentionDecoder(nn.Module):
	def __init__(self):
		super(SelfAttentionDecoder, self).__init__()
		self.decoder = nn.Sequential(
			PMA(dim=2048, num_heads=4, num_seeds=1),
			nn.Linear(in_features=2048, out_features=1),
		)

	def forward(self, inputs):
		return self.decoder(inputs)
