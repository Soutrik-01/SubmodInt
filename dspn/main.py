import torch 
import torch.distributed as dist
import numpy as np
import os 
from argparse import ArgumentParser
from arg_utils import add_args
from uuid import uuid4
from utils import *
from dspn_network import DSPN

def prepare_model_optimizer_scheduler(args, device, local_rank):

	def load_model_from_ckpt_if_any(model, f_optim, dsf_optim, f_scheduler, dsf_scheduler, args, device):
		start = 0
		if args['load_from_ckpt']:
			with torch.serialization.safe_globals([np._core.multiarray.scalar, np.dtype, np.dtypes.Float64DType]):
				save_state_dict = torch.load(args['load_from_ckpt'], map_location = device)
			ckpt = save_state_dict['net']; ckpt = {k[7:]: ckpt[k] for k in ckpt.keys()}
			dewrap(model, True).load_state_dict(ckpt)

			f_optim.load_state_dict(save_state_dict['f_optim'])
			dsf_optim.load_state_dict(save_state_dict['dsf_optim'])
			if not (args['scheduler'] == 'cyclic' or args['scheduler'] is None):
				f_scheduler.load_state_dict(save_state_dict['f_scheduler'])
				dsf_scheduler.load_state_dict(save_state_dict['dsf_scheduler'])
			start = save_state_dict['epoch']+1
		return model, f_optim, dsf_optim, f_scheduler, dsf_scheduler, start
	nesting_list=[i for i in range(10, args['set_size']+args['nesting_interval'], args['nesting_interval'])]

	def initialize_schedulers_if_any(f_optim, dsf_optim, args):
		if args['scheduler'] == 'cosine':	
			f_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(f_optim, eta_min= 0*args['lr'] ,T_max=args["n_epochs"])
			dsf_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(dsf_optim, eta_min= 0*args['lr'] ,T_max=args["n_epochs"])
		elif args['scheduler'] == 'triangular':
			f_scheduler = TriangularWarmRestartLR(f_optim, args['lr'], 5e-2*args['lr'], 10, mult_factor=2, gamma = args['lr_gamma'], verbose=True)
			dsf_scheduler = TriangularWarmRestartLR(dsf_optim,args['lr']*0.25, 5e-2*args['lr']*0.25, 10, mult_factor=2, gamma = args['lr_gamma'], verbose=True)
		elif args['scheduler'] == 'cyclic':
			f_scheduler, dsf_scheduler = None, None
			if is_main_process():	
				print("Cyclic Scheduler will be handled separately in the code")
		else:
			print("Unknown fixed learning rate!!!")

		return f_scheduler, dsf_scheduler

	def freeze_components_if_any(args, model):
		if args['freeze_dsf']:
			print("Freezing roof.")
			for p in model.dsf.parameters():
				p.requires_grad = False

		if args['freeze_pillar']:
			print("Freezing Pillar.")
			for p in model.feat.parameters():
				p.requires_grad = False

		return model 

	def create_optim(args, model):
		f_optim = torch.optim.AdamW(dewrap(model, True).feat.parameters(), lr = args['lr'], weight_decay=args['f_weight_decay'])
		dsf_optim = torch.optim.AdamW(dewrap(model, True).dsf.parameters(), lr = args['lr']*0.1, weight_decay=args['dsf_weight_decay'])
		return f_optim, dsf_optim

	if args['model_type'] == 'deepset':
		if args['dset'] in ['IN100', 'CIFAR100']:
			model = DSPN(out_dims=[2048], concave=['relu', 'relu', 'relu', 'relu'], args=args, nesting_list=nesting_list) 
		else:
			model = DSPN(out_dims=[512, 10], concave=['relu', 'relu', 'relu', 'relu'], args=args, nesting_list=nesting_list) 
	else:
		if args['dset'] in ['IN100', 'CIFAR100']:
			model = DSPN(out_dims=[2048], concave=['log1p', 'exp', 'sqrt', 'linear'], args=args, nesting_list=nesting_list)
		else:
			model = DSPN(out_dims=[512, 10], concave=['log1p', 'exp', 'sqrt', 'linear'], args=args, nesting_list=nesting_list)

	model.to(device)
	model.set_device(device)
	torch.cuda.set_device(device)	
	model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

	f_optim, dsf_optim = create_optim(args, model)
	f_scheduler, dsf_scheduler = initialize_schedulers_if_any(f_optim, dsf_optim, args)

	model, f_optim, dsf_optim, f_scheduler, dsf_scheduler, start = load_model_from_ckpt_if_any(model, f_optim, dsf_optim, f_scheduler, dsf_scheduler, args, device)


	torch.cuda.empty_cache()
	
				
	model = freeze_components_if_any(args, model)

	return model, f_optim, dsf_optim, f_scheduler, dsf_scheduler, start


def main(local_rank, args):
	
	world_size = torch.cuda.device_count()
	device = torch.device(f'cuda:{local_rank}')
	if os.environ.get('DSPN_GPU_MEMORY_FRACTION'):
		torch.cuda.set_per_process_memory_fraction(float(os.environ['DSPN_GPU_MEMORY_FRACTION']), device)
	args = vars(args)
	args['out_dim'] = 2048 if args['dset'] in ['IN100', 'CIFAR100'] else 512
	if is_main_process():
		uid = str(uuid4())
		if not (args['path_directory'] is None):
			uid =  args['path_directory']	
		print("*"*50, "TRAINING CONFIGURATION", "*"*50, "\n")
		print(args, "\n")
		print("*"*50, "TRAINING CONFIGURATION", "*"*50, "\n")

	
	dist.init_process_group("nccl", rank=local_rank, world_size=world_size)

	set_random_seed(args['seed'])

	print(f"Local Rank: {local_rank}, World Size: {world_size}")
	print("Preparing model, optimizer and scheduler..")
	model, f_optim, dsf_optim, f_scheduler, dsf_scheduler, start = prepare_model_optimizer_scheduler(args, device, local_rank)

	print("Instantiating loaders and target FL object..")
	train_loader, FL, dataset_, X_train, Y_train, D_M_idx_full, D_E_idx_full = create_train_test_loaders(args, device, dewrap(model, True).nesting_list, local_rank, world_size)

	if is_main_process():
		prepare_metadata(args, model, uid)

	
	dist.barrier()

	train_loader_sf=None
	
	logger_maps = {0:'vanilla', 1:'finegrained', 2:'nnkmeans', 3:'FL', 4: 'balanced', 5:'matroids', 6:'remaining'}
	train_loader_sf, margin_full_sf = None, None
	last_epoch = last_training_epoch(args)
	for epoch in (range(start, last_epoch)):
		model.train()
		torch.cuda.empty_cache() 
		current_lr = get_current_learning_rate(dsf_scheduler, args, epoch)
		if args['wandb'] and is_main_process():
			wandb.log({'Current DSF Learning Rate': current_lr})

		train_loader_sf, margin_full_sf = instantiate_feedback_loaders_at_epoch(epoch, D_M_idx_full, D_E_idx_full, X_train, Y_train, model, dataset_, args, device, FL, logger_maps, train_loader_sf, margin_full_sf)

		dist.barrier()
	
		train_epoch(model, train_loader, f_optim, dsf_optim, epoch, args, feedback_loader=train_loader_sf)

		if not ((args['scheduler'] is None) or (args['scheduler'] == 'cyclic')):	
			f_scheduler.step()
			dsf_scheduler.step()
		
		if is_main_process() and (is_saving_epoch(epoch, args['save_every']) or is_last_epoch(epoch, last_epoch)):
			save_checkpoint(model, f_optim, dsf_optim, f_scheduler, dsf_scheduler, epoch, args, uid)
	
		dist.barrier()
		
	if args['wandb']  and is_main_process():
		wandb.finish()
	
	dist.destroy_process_group()


parser = ArgumentParser()
add_args(parser)
args = parser.parse_args()
local_rank = int(os.environ["LOCAL_RANK"]) 
main(local_rank, args)
