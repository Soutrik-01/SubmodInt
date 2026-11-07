import torch 
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from dataset.set_dataloader import SetDataset
import wandb
import random 
import numpy as np
import time 
from submodular.fl import facilityLocation
from submodular.advanced_fb_generator import DiverseDataGenerator
import torch
from torch.optim.lr_scheduler import _LRScheduler
import os 
import json 
import pickle as pkl

def set_random_seed(seed: int) -> None:
	"""
	Sets the seeds at a certain value.
	:param seed: the value to be set
	"""
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	random.seed(seed)
	np.random.seed(seed)
 
def seed_worker(worker_id):
	worker_seed = torch.initial_seed() % 2**32
	np.random.seed(worker_seed)
	random.seed(worker_seed)

def dewrap(model, ddp=False):
	if ddp:
		return model.module
	else:
		return model

def is_dist_avail_and_initialized():
	if not dist.is_available():
		return False

	if not dist.is_initialized():
		return False

	return True

def save_on_master(*args, **kwargs):

	if is_main_process():
		torch.save(*args, **kwargs)

def get_rank():

	if not is_dist_avail_and_initialized():
		return 0

	return dist.get_rank()

def is_main_process():
	return get_rank() == 0


def prepare_metadata(args, model, uid):

	def create_directories_and_dump_config(args, uid):
		if not os.path.exists("trainlogs/"):
			os.mkdir("trainlogs/")
		if not os.path.exists(f"trainlogs/{uid}/"):	
			os.mkdir(f"trainlogs/{uid}/")
		with open(f"trainlogs/{uid}/arguments.json", "w") as outfile:
			json.dump(args, outfile)

	create_directories_and_dump_config(args, uid)
	if args['wandb']:
		prepare_wandb_logger(model, args)
		wandb.log({'Ckpt folder': uid})

def prepare_wandb_logger(model, args):
	import wandb
	wandb.init(project=args['project_name'], config=args, settings=wandb.Settings(code_dir="."))
	wandb.watch(model)

def log_DSF_Accuracy(train_loader_sf, margin_full_sf, NESTING_LIST, logger_maps):
	dsf_accuracy={}
	for loaders_idx in range(len(train_loader_sf)):
		loader = train_loader_sf[loaders_idx]
		if loader is None:
			continue 
		for k in range(len(NESTING_LIST)):
			dsf_accuracy[f"{logger_maps[loaders_idx]}_DSF_Accuracy@Nesting{NESTING_LIST[k]}"] =  ((margin_full_sf[loaders_idx][k]>0).sum()/margin_full_sf[loaders_idx].shape[-1]).item() 				

	return dsf_accuracy

def is_last_epoch(epoch, num_epochs):
	return epoch + 1 == num_epochs

def is_saving_epoch(epoch, save_every):
	return (epoch+1)%save_every==0

def save_checkpoint(model, f_optim, dsf_optim, f_scheduler, dsf_scheduler, epoch, args, uid):
	print("Saving Model...")
	if (args['scheduler'] == 'cyclic') or (args['scheduler'] is None):	
		save_dict={
		'epoch': epoch,
		'net': model.state_dict(),
		'f_optim': f_optim.state_dict(),
		'dsf_optim': dsf_optim.state_dict()
		}
	else:
		save_dict={
		'epoch': epoch,
		'net': model.state_dict(),
		'f_optim': f_optim.state_dict(),
		'dsf_optim': dsf_optim.state_dict(),
		'f_scheduler': f_scheduler.state_dict(),
		'dsf_scheduler': dsf_scheduler.state_dict()
		}
	torch.save(save_dict, f'trainlogs/{uid}/dspn_ckpt_{epoch}.pt')	


class TriangularWarmRestartLR(_LRScheduler):
	def __init__(self, optimizer, max_lr, min_lr, period, mult_factor=2, last_epoch=-1, gamma=1., verbose=False):
		self.max_lr = max_lr
		self.min_lr = min_lr
		self.period = period
		self.mult_factor = mult_factor
		self.current_cycle = 0
		self.gamma = gamma

		super(TriangularWarmRestartLR, self).__init__(optimizer, last_epoch, verbose)

	def get_lr(self):
		if self.last_epoch == self.period:
			self.current_cycle += 1
			self.period = self.period * self.mult_factor
			self.last_epoch = 0
			self.max_lr = self.max_lr*self.gamma

		if self.last_epoch > self.period:
			raise ValueError("Number of epochs exceeded the maximum limit.")
		lr_scale = 1.0 - self.last_epoch / self.period
		lrs = [self.min_lr + (self.max_lr - self.min_lr) * lr_scale for _ in self.optimizer.param_groups] 
		return lrs
		
def get_cyclic_lr(epoch, lr, epochs, lr_peak_epoch):
	xs = [0, lr_peak_epoch, epochs]
	ys = [0.05 * lr, lr, 0]
	return np.interp([epoch], xs, ys)[0]

def scheduled_cyclic_lr(epoch, args):
	"""The cyclic schedule; with --anneal_from E it instead decays linearly from its value at E + 1 to 0 over --anneal_epochs."""
	if args.get('anneal_from') is None or epoch <= args['anneal_from'] + 1:
		return get_cyclic_lr(epoch, args['lr'], args['n_epochs'], 0)
	start = get_cyclic_lr(args['anneal_from'] + 1, args['lr'], args['n_epochs'], 0)
	return start * max(0.0, 1.0 - (epoch - args['anneal_from'] - 1) / args['anneal_epochs'])

def last_training_epoch(args):
	if args.get('anneal_from') is None:
		return args['n_epochs']
	return args['anneal_from'] + 1 + args['anneal_epochs']

def get_current_learning_rate(scheduler, args, epoch):
	if args['scheduler'] == 'cyclic':
		current_lr = scheduled_cyclic_lr(epoch, args)
	elif (args['scheduler'] is  None):	
		current_lr = args['lr']
	else:
		current_lr = scheduler.get_lr()[0]
	return current_lr

def instantiate_feedback_loaders_at_epoch(epoch, D_M_idx_full, D_E_idx_full, X_train, Y_train, model, dataset_, args, device, FL, logger_maps, train_loader_sf=None, margin_full_sf=None):
	
	if ((train_loader_sf is None) or (epoch+1)%args['feedback_every'] == 0) and args['submodular_feedback']>=0: 

		train_loader_sf, margin_full_sf =  None, None  
		del train_loader_sf
		del margin_full_sf
		train_loader_sf, margin_full_sf=create_submodular_feedback_loaders(D_M_idx_full, D_E_idx_full, X_train, Y_train, model, dataset_, args, device, FL, dewrap(model, True).nesting_list)
		if is_main_process():	
			dsf_accuracy = log_DSF_Accuracy(train_loader_sf, margin_full_sf, dewrap(model, True).nesting_list, logger_maps)
			print(dsf_accuracy)
			if args['wandb']:
				wandb.log(dsf_accuracy)

	return train_loader_sf, margin_full_sf

def trim_zero_margins(D_M_idx_full, D_E_idx_full, margin_full, F_M=None, F_E=None):
	V = np.arange(len(D_M_idx_full))
	zero_idx = (margin_full == 0.).sum(0); print("# zero entries:", zero_idx.sum().item())
	zero_idx = V[(zero_idx == 0)]
	if F_M is None:
		return D_M_idx_full[zero_idx], D_E_idx_full[zero_idx], margin_full[:, zero_idx]
	else:
		return D_M_idx_full[zero_idx], D_E_idx_full[zero_idx], margin_full[:, zero_idx], F_M[:, zero_idx], F_E[:, zero_idx]


def fetch_dataset_imgs_labels_passive_subsets(args):
	def fetch_dataset_imgs_labels(args):
		path = os.path.join(args['root'], args['dset']) 
		X_train, Y_train= np.load(f"{path}/dataset/X_train.npy"), np.load(f"{path}/dataset/Y_train.npy")
		X_train, Y_train = torch.from_numpy(X_train), torch.from_numpy(Y_train)
		return X_train, Y_train 

	def import_passive_subsets(args):
		path = os.path.join(args['root'], args['dset']) 
		path  = f"{path}/passive_idx/passive_samples.pkl"
		set_size = args['set_size']
		with open(path, 'rb') as g:
			dataset_ = pkl.load(g)

		D_M_idx_full=[]; D_E_idx_full=[]
		for k in dataset_.keys():
			D_M_idx_full.append(dataset_[k][2])
			D_E_idx_full.append(dataset_[k][3])

		D_M_idx_full = torch.cat(D_M_idx_full, dim=0)	
		D_E_idx_full = torch.cat(D_E_idx_full, dim=0)

		if not (set_size is None):
			D_M_idx_full = D_M_idx_full[:, :set_size]
			D_E_idx_full = D_E_idx_full[:, :set_size]

		return D_M_idx_full, D_E_idx_full, dataset_
	D_M_idx_full, D_E_idx_full, dataset_ = import_passive_subsets(args)
	X_train, Y_train  = fetch_dataset_imgs_labels(args)

	return X_train, Y_train, D_M_idx_full, D_E_idx_full, dataset_


def create_train_test_loaders(args, device, NESTING_LIST, local_rank, world_size):

	X_train, Y_train, D_M_idx_full, D_E_idx_full, dataset_ = fetch_dataset_imgs_labels_passive_subsets(args) 

	FL = facilityLocation(X_train[:, :, 0], device, args=args)

	margin_full, F_M, F_E = FL.get_margin_FL_nested(D_M_idx_full, D_E_idx_full, NESTING_LIST, Y=(Y_train), matroid_rank=args['matroid_rank'], rank_tradeoff=args['matroid_rank_tradeoff'], bsz = args['target_computation_bsz'])
	D_M_idx_full, D_E_idx_full, margin_full, F_M, F_E = trim_zero_margins(D_M_idx_full, D_E_idx_full, margin_full, F_M, F_E)
	
	training_set_tuples = SetDataset(X_train, Y_train, D_M_idx_full, D_E_idx_full, margin_full, F_M, F_E, device = torch.device(f'cuda:{local_rank}'),  nested=args['nesting'], n_views=args['n_views'])

	
	train_sampler = torch.utils.data.distributed.DistributedSampler(training_set_tuples,
										num_replicas=world_size,
										rank=local_rank,
										shuffle=True,
										seed=args['seed'])

	train_loader =  DataLoader(training_set_tuples, batch_size=args['bsz'], shuffle=not True, sampler=train_sampler)

	return train_loader, FL, dataset_, X_train, Y_train, D_M_idx_full, D_E_idx_full


def create_submodular_feedback_loaders(D_M_idx_full, D_E_idx_full, X_train, Y_train, model, dataset_, args, device, FL, NESTING_LIST):

	def prepare_feedback_data(D_M_idx_full_sf, D_E_idx_full_sf, FL, NESTING_LIST, args, Y_train):
		D_M_idx_full_sf, D_E_idx_full_sf = torch.from_numpy(D_M_idx_full_sf), torch.from_numpy(D_E_idx_full_sf)

		if args['use_matroid_rank_in_feedback']:	
			margin_full_sf, FL_A_sf, FL_B_sf = FL.get_margin_FL_nested(D_M_idx_full_sf, D_E_idx_full_sf, NESTING_LIST,  Y=(Y_train), matroid_rank=args['matroid_rank'], rank_tradeoff=args['matroid_rank_tradeoff'], target_responsibility=args['target_responsibility'])
		else:
			margin_full_sf, FL_A_sf, FL_B_sf = FL.get_margin_FL_nested(D_M_idx_full_sf, D_E_idx_full_sf, NESTING_LIST, bsz=args['target_computation_bsz'])

		D_M_idx_full_sf, D_E_idx_full_sf, margin_full_sf, FL_A_sf, FL_B_sf = trim_zero_margins(D_M_idx_full_sf, D_E_idx_full_sf, margin_full_sf, FL_A_sf, FL_B_sf)	
		torch.cuda.empty_cache()

		return D_M_idx_full_sf, D_E_idx_full_sf, margin_full_sf, FL_A_sf, FL_B_sf

	def create_dataloader(X_train, Y_train, D_M_idx_full_sf, D_E_idx_full_sf, margin_full_sf, FL_A_sf, FL_B_sf, device, args, nested=True):
		train_loader_sf =  DataLoader(SetDataset(X_train, Y_train, D_M_idx_full_sf, D_E_idx_full_sf, margin_full_sf, FL_A_sf, FL_B_sf, device = device, nested=True), batch_size=args['bsz'])
		torch.cuda.empty_cache()
		return 	train_loader_sf
	if args['use_training_pair_fb']:		
		D_M_idx_full_sf, D_E_idx_full_sf, D_M_idx_full_nnkmeans, D_E_idx_full_nnkmeans, D_M_idx_full_fl, D_E_idx_full_fl, D_M_idx_full_balanced, D_E_idx_full_balanced, D_M_idx_full_matroid_v_nmatroid, D_E_idx_full_matroid_v_nmatroid, D_M_idx_full_all_remaining, D_E_idx_full_all_remaining= DiverseDataGenerator(X_train[:, :, 0], Y_train, model, args, dataset_, device, budget=args['sg_budget'], set_size=args['set_size'], K=args['K']).fetch_diverse_sets(dataset_, V=len(X_train), num_rand=len(D_M_idx_full), D_M_idx_full_train=D_M_idx_full, D_E_idx_full_train=D_E_idx_full)
	else:
		D_M_idx_full_sf, D_E_idx_full_sf, D_M_idx_full_nnkmeans, D_E_idx_full_nnkmeans, D_M_idx_full_fl, D_E_idx_full_fl, D_M_idx_full_balanced, D_E_idx_full_balanced, D_M_idx_full_matroid_v_nmatroid, D_E_idx_full_matroid_v_nmatroid, D_M_idx_full_all_remaining, D_E_idx_full_all_remaining= DiverseDataGenerator(X_train[:, :, 0], Y_train, model, args, dataset_, device, budget=args['sg_budget'], set_size=args['set_size'], K=args['K']).fetch_diverse_sets(dataset_, V=len(X_train), num_rand=len(D_M_idx_full))

	D_M_idx_full_sf, D_E_idx_full_sf, margin_full_sf, FL_A_sf, FL_B_sf = prepare_feedback_data(
		D_M_idx_full_sf, D_E_idx_full_sf, FL, NESTING_LIST, args, Y_train
	)

	if not D_M_idx_full_nnkmeans is None:
		D_M_idx_full_nnkmeans, D_E_idx_full_nnkmeans, margin_full_nnkmeans, FL_A_nnkmeans, FL_B_nnkmeans = prepare_feedback_data(
			D_M_idx_full_nnkmeans, D_E_idx_full_nnkmeans, FL, NESTING_LIST, args, Y_train

		)

	if not D_M_idx_full_fl is None:

		D_M_idx_full_fl, D_E_idx_full_fl, margin_full_sf_fl, FL_A_sf_fl, FL_B_sf_fl = prepare_feedback_data(
			D_M_idx_full_fl, D_E_idx_full_fl, FL, NESTING_LIST, args, Y_train

		)

	if not D_M_idx_full_balanced is None:

		D_M_idx_full_balanced, D_E_idx_full_balanced, margin_full_sf_balanced, FL_A_sf_balanced, FL_B_sf_balanced = prepare_feedback_data(
			D_M_idx_full_balanced, D_E_idx_full_balanced, FL, NESTING_LIST, args, Y_train

		)

	if not D_M_idx_full_matroid_v_nmatroid is None:

		D_M_idx_full_matroid_v_nmatroid, D_E_idx_full_matroid_v_nmatroid, margin_full_sf_matroid_v_nmatroid, FL_A_sf_matroid_v_nmatroid, FL_B_sf_matroid_v_nmatroid = prepare_feedback_data(
			D_M_idx_full_matroid_v_nmatroid, D_E_idx_full_matroid_v_nmatroid, FL, NESTING_LIST, args, Y_train

		)


	if not D_M_idx_full_all_remaining is None:
		D_M_idx_full_all_remaining, D_E_idx_full_all_remaining, margin_full_sf_rem, FL_A_sf_rem, FL_B_sf_rem = prepare_feedback_data(
			D_M_idx_full_all_remaining, D_E_idx_full_all_remaining, FL, NESTING_LIST, args, Y_train

		)

	train_loader_sf = create_dataloader(X_train, Y_train, D_M_idx_full_sf, D_E_idx_full_sf, margin_full_sf, FL_A_sf, FL_B_sf, device, args, nested=True)

	if not D_M_idx_full_nnkmeans is None:
		train_loader_sf_nnkmeans = create_dataloader(X_train, Y_train, D_M_idx_full_nnkmeans, D_E_idx_full_nnkmeans, margin_full_nnkmeans, FL_A_nnkmeans, FL_B_nnkmeans, device, args, nested=True)
	else:
		train_loader_sf_nnkmeans, margin_full_nnkmeans = None,None

	if not D_M_idx_full_fl is None:
		train_loader_sf_fl = create_dataloader(X_train, Y_train, D_M_idx_full_fl, D_E_idx_full_fl, margin_full_sf_fl, FL_A_sf_fl, FL_B_sf_fl, device, args, nested=True)
	else:
		train_loader_sf_fl, margin_full_sf_fl = None,None

	if not D_M_idx_full_balanced is None:
		train_loader_sf_balanced = create_dataloader(X_train, Y_train, D_M_idx_full_balanced, D_E_idx_full_balanced, margin_full_sf_balanced, FL_A_sf_balanced, FL_B_sf_balanced, device, args, nested=True)
	else:
		train_loader_sf_balanced, margin_full_sf_balanced = None,None

	if not D_M_idx_full_matroid_v_nmatroid is None:
		train_loader_sf_matroid_v_nmatroid = create_dataloader(X_train, Y_train, D_M_idx_full_matroid_v_nmatroid, D_E_idx_full_matroid_v_nmatroid, margin_full_sf_matroid_v_nmatroid, FL_A_sf_matroid_v_nmatroid, FL_B_sf_matroid_v_nmatroid, device, args, nested=True)
	else:
		train_loader_sf_matroid_v_nmatroid, margin_full_sf_matroid_v_nmatroid = None,None

	if not D_M_idx_full_all_remaining is None:
		train_loader_sf_rem = create_dataloader(X_train, Y_train, D_M_idx_full_all_remaining, D_E_idx_full_all_remaining, margin_full_sf_rem, FL_A_sf_rem, FL_B_sf_rem, device, args, nested=True)
	else:
		train_loader_sf_rem, margin_full_sf_rem = None,None

	train_loader_sf_fine, margin_full_sf_fine = None,None

	return [train_loader_sf, train_loader_sf_fine, train_loader_sf_nnkmeans, train_loader_sf_fl, train_loader_sf_balanced, train_loader_sf_matroid_v_nmatroid, train_loader_sf_rem], [margin_full_sf, margin_full_sf_fine, margin_full_nnkmeans, margin_full_sf_fl, margin_full_sf_balanced, margin_full_sf_matroid_v_nmatroid, margin_full_sf_rem]


def peripteral_cost(SCMI_B_batch, SCMI_A_batch, margin, gate_alpha=1e-4, thresh=1e-15, beta=1., tau=1., reduction='mean'):
	
	if (abs(margin)).sum() < 1e-12:
		peripteral_loss = torch.abs(SCMI_B_batch-SCMI_A_batch)

		if reduction == 'mean':
			peripteral_loss = peripteral_loss.mean()

		return peripteral_loss, torch.tensor(1).to(peripteral_loss.device)


	gate = F.tanh(gate_alpha/(torch.abs(margin))) 
	l1 = F.softplus(tau- (SCMI_A_batch-SCMI_B_batch)/(margin+torch.sign(margin)*thresh), beta=beta)*torch.abs(margin+torch.sign(margin)*thresh)
	l2 = torch.abs(SCMI_B_batch-SCMI_A_batch)

	peripteral_loss = (1-gate)*l1 + gate*l2
	if reduction == 'mean':	
		peripteral_loss = peripteral_loss.mean()
		
	return peripteral_loss, gate.mean()

def max_margin_loss(SCMI_B_batch, SCMI_A_batch, margin, gate_alpha=1e-4, beta=1.):
	bsz = len(SCMI_A_batch)
	idx_pos = margin> gate_alpha
	idx_neg = margin< -gate_alpha
	idx_small =  (margin < gate_alpha)*(margin>-gate_alpha)
	loss_pos = (F.softplus(SCMI_A_batch[idx_pos] + margin[idx_pos] - SCMI_B_batch[idx_pos], beta=beta)).sum()
	loss_neg = (F.softplus(SCMI_B_batch[idx_neg] - margin[idx_neg] - SCMI_A_batch[idx_neg], beta=beta)).sum()
	loss_small = (torch.abs(SCMI_B_batch[idx_small]-SCMI_A_batch[idx_small])).sum()
	loss = (loss_pos + loss_neg + loss_small)/bsz
	return loss, idx_small.sum()/bsz

def compute_accuracy(SCMI_A_batch, SCMI_B_batch, margin):
	return (((torch.sign((SCMI_A_batch-SCMI_B_batch)*margin)+1).sum())/(2*len(margin))).item()



def train_epoch(model, train_loader, f_optim, dsf_optim, epoch_id, args, feedback_loader=None):
	if args['scheduler'] == 'cyclic':
		lr_start, lr_end = scheduled_cyclic_lr(epoch_id, args), scheduled_cyclic_lr(epoch_id + 1, args)
		iters = len(train_loader)	
		lrs = np.interp(np.arange(iters), [0, iters], [lr_start, lr_end])

	train_loader.sampler.set_epoch(epoch_id)
	feedback_iter = []
	model.train()
	if feedback_loader: 
		for f in feedback_loader:
			if f is None:
				feedback_iter.append(None)
			else:
				feedback_iter.append(iter(f))
	batches_processed=0; num_batches=len(train_loader)

	if is_main_process():
		start_time = time.time()
	count=0
	batch_timer, step_timer  = 0. , 0.
	logger_maps = {0:'vanilla', 1:'finegrained', 2:'nnkmeans', 3:'FL', 4: 'balanced', 5:'matroids', 6:'remaining'}
	for D_A_idx, D_B_idx, D_A, D_B, _, _, margin, FL_M, FL_E in (train_loader):
		if args['scheduler'] == 'cyclic':
			for param_group in f_optim.param_groups:
				param_group['lr'] = lrs[count]
			for param_group in dsf_optim.param_groups:
				param_group['lr'] = lrs[count] * args['roof_lr_scale']

		final_loss=0; count+=1
		batch_time = time.time()

		if args['n_views'] == 1:
			peripteral_loss, loss_activity_norm, set_classification_accuracy, gate = dewrap(model, True).compute_loss(D_A[0], D_B[0], margin*args['MAX'])
		else:
			if args['model_type'] == 'set_transformer':	
				peripteral_loss, roof_consistency_loss, gain_loss, singleton_roof_consistency_loss, singleton_gain_loss,nested_peripteral_loss_with_augmentation, set_classification_accuracy, gate, loss_activity_norm = dewrap(model, True).compute_loss_augmented_nviews_set_transformer(D_A, D_B, margin*args['MAX'], FL_M*args['MAX'], FL_E*args['MAX'])
			else:
				peripteral_loss, roof_consistency_loss, gain_loss, singleton_roof_consistency_loss, singleton_gain_loss,nested_peripteral_loss_with_augmentation, set_classification_accuracy, gate, loss_activity_norm = dewrap(model, True).compute_loss_augmented_nviews(D_A, D_B, margin*args['MAX'], FL_M*args['MAX'], FL_E*args['MAX'])

		batch_timer += time.time() - batch_time 
		
  
		if args['nesting']:
			logger = {f"Peripteral Loss@Nesting {dewrap(model, True).nesting_list[i]}": peripteral_loss[i].item() for i, nesting in enumerate(dewrap(model, True).nesting_list)}
			logger = {**logger, **{f"Set Accuracy@Nesting {dewrap(model, True).nesting_list[i]}": accuracy for i, accuracy in enumerate(set_classification_accuracy)}}
		else:
			logger = {
			"Peripteral Loss": peripteral_loss.item(), 
			"Set Accuracy": set_classification_accuracy,
			"gate":gate
			}

		if args['n_views']  > 1:
			logger = {**logger, 'augmentation_consistency_loss': roof_consistency_loss.mean().item(), 'cross_augmentation_loss': peripteral_loss.mean().item(), "singleton_roof_consistency_loss":singleton_roof_consistency_loss.item(), "nested_peripteral_loss_with_augmentation":nested_peripteral_loss_with_augmentation.item()}

		final_loss += peripteral_loss.sum()
		if args['n_views']>1:
			final_loss += args['roof_consistency_loss']*(roof_consistency_loss.sum())
			final_loss += args['singleton_roof_consistency_loss']*singleton_roof_consistency_loss
			final_loss+=args['nested_peripteral_loss_with_augmentation']*nested_peripteral_loss_with_augmentation   

			if args['gain_loss']>0.:	
				final_loss += args['gain_loss']*gain_loss.sum()
			
			if args['singleton_gain_loss']>0.:		
				final_loss += args['singleton_gain_loss']*singleton_gain_loss
			
		final_loss+=args['activity_reg_coefficient']*loss_activity_norm

		

		if args['activity_reg_coefficient']>0.:
			logger['loss_activity_norm']=loss_activity_norm.item()

		
		if feedback_loader:
			for loader_idx in range(len(feedback_loader)):
				if feedback_loader[loader_idx] is None:
					continue 
				try:
					_, _, D_A_sf, D_B_sf, _, _, margin_sf, _, _ = next(feedback_iter[loader_idx])
				except StopIteration as e:
					print("Resetting Feedback Iterator as it has ended")
					feedback_iter[loader_idx] = iter(feedback_loader[loader_idx])
					_, _, D_A_sf, D_B_sf, _, _, margin_sf, _, _ = next(feedback_iter[loader_idx])
	 
				if args['model_type'] == 'set_transformer':
					peripteral_loss_SF, _, set_classification_accuracy_SF, _ = dewrap(model, True).compute_loss_set_transformer(D_A_sf[0], D_B_sf[0], margin_sf*args['MAX_sf'])
				else:
					peripteral_loss_SF, _, set_classification_accuracy_SF, _ = dewrap(model, True).compute_loss(D_A_sf[0], D_B_sf[0], margin_sf*args['MAX_sf'])


				logger = {**logger, **{f"{logger_maps[loader_idx]} SF Peripteral Loss@Nesting {dewrap(model, True).nesting_list[i]}": peripteral_loss_SF[i].item() for i, _ in enumerate(dewrap(model, True).nesting_list)}}
				logger = {**logger, **{f"{logger_maps[loader_idx]} SF Set Accuracy@Nesting {dewrap(model, True).nesting_list[i]}": set_classification_accuracy_SF[i] for i, _ in enumerate(dewrap(model, True).nesting_list)}}
		
				final_loss += args['feedback_coefficient']*peripteral_loss_SF.sum()

		step_time = time.time()
		final_loss.backward()

		if (batches_processed+1)%args['accumulation_steps']==0:
			if not args['freeze_pillar']:
				f_optim.step()
			if not args['freeze_dsf']:
				dsf_optim.step()
				
			if not (args['model_type'] == 'deepset' or args['model_type'] == 'set_transformer'):	
				dewrap(model, True).project()
			f_optim.zero_grad(set_to_none=True)
			dsf_optim.zero_grad(set_to_none=True)
			step_timer+= time.time() -  step_time
		elif not (args['model_type'] == 'deepset' or args['model_type'] == 'set_transformer'):
			dewrap(model, True).project()

		logger['final_loss']=final_loss.item()
		if args['wandb'] and is_main_process():
			wandb.log(logger)

		batches_processed+=1

		if batches_processed%25==0 and is_main_process():
			print("Average Time per batch ", (time.time()-start_time)/batches_processed)
			print("Average Time per loss computation ", batch_timer/batches_processed)
			print("Average Time per Gradient step ", step_timer/batches_processed)
			print ('Epoch [{}/{}], Step [{}/{}], Loss: {}' 
				   .format(epoch_id+1, args['n_epochs'], batches_processed, num_batches, final_loss.item()))

	return model