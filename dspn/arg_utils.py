import random 
from tqdm import tqdm
import numpy as np
import sys
from argparse import ArgumentParser
import os 

def add_data_args(parser):

	parser.add_argument('--seed', type=int, default=10,
					help='The random seed.')
	parser.add_argument('--proj_name', type=str, default='DSPN_Large',
					help='DSF Concave Function')
	parser.add_argument('--wandb', action='store_true',
						help='wandb')
	parser.add_argument('--project_name', type=str, default='DSPN',
					help='Name of wandb project')
	parser.add_argument('--save_every', type=int, default=5,
						help='Number of epochs after which one saves the model')

	parser.add_argument('--root', type=str, required=True,
					help='Data Location Folder')
	parser.add_argument('--dset', type=str, required=True,
						help='Name of the Dataset')
	parser.add_argument('--K', type=int, default=10,
						help='num clusters/classes')
	parser.add_argument('--set_size', type=int, default=100,
						help='training set size')
	parser.add_argument('--nesting', action='store_true',
						help='If we want to use nested chain of sets for the training')
	parser.add_argument('--nesting_interval', type=int, default=10,
						help='Nesting Set interval')

	parser.add_argument('--submodular_feedback', type=int, default=-1,
						help='After how many epoch we will start doing submodular feedback?')
	parser.add_argument('--r_size', type=int, default=None,
						help='Random subset size for Stochastic greedy methods.')
	parser.add_argument('--sg_budget', type=int, default=100,
						help='Number of random sets in the stoch. greedy')
	parser.add_argument('--feedback_every', type=int, default=1,
						help='Create new fb generator after n epochs')
	parser.add_argument('--feedback_coefficient', type=float, default=0.1,
						help='Coefficient of submodular Feedback Loss.')
	parser.add_argument('--prop_fullGS', type=float, default=0.2,
						help='How many of feedback examples should be concerned with full ground set comparison.')
	parser.add_argument('--p_min', type=float, default=0.9,
						help='Minimum swap probability for the altmaxmin feedback')
	parser.add_argument('--use_training_pair_fb', action='store_true',
						help='If we want to use training E, M pairs in the submodular feedback')


	parser.add_argument('--class_balanced_feedback', action='store_true',
						help='wandb')
	parser.add_argument('--matroid_v_non_matroid', action='store_true',
						help='wandb')
	parser.add_argument('--use_matroid_rank_in_feedback', action='store_true',
						help='use_matroid_rank_in_feedback')
	parser.add_argument('--all_remaining', action='store_true',
						help='all_remaining')
		
	parser.add_argument('--nnkmeans_feedback', action='store_true',
						help='If we want to use target feedback.')
	parser.add_argument('--target_feedback', action='store_true',
						help='If we want to use target feedback.')



	parser.add_argument('--beta', type=float, default=1.,
						help='Softplus Scaling')
	parser.add_argument('--tau', type=float, default=1.,
						help='Constant in Softplus')
	parser.add_argument('--MAX', type=float, default=10.,
						help='Multiply margin scores by this constant')
	
	parser.add_argument('--MAX_sf', type=float, default=10.,
						help='Multiply Feedback margin scores by this constant')
	
	parser.add_argument('--kw', type=float, default=0.1,
						help='Kernel Width for the Structured Margin FL Target')
	
	parser.add_argument('--matroid_rank', type=int, default=-1,
						help='To use Matroid Rank based margin or not')
	parser.add_argument('--matroid_rank_tradeoff', type=float, default=0.,
						help='Weightage of matroid rank in margin computation')
	parser.add_argument('--target_responsibility', type=float, default=1.,
						help='Weightage of FL target in margin computation')


	parser.add_argument('--n_views', type=int, default=1, help='how many augmented views to generate of each set')


def add_architectural_args(parser):
	parser.add_argument('--freeze_dsf', action='store_true',
						help='')
	parser.add_argument('--freeze_pillar', action='store_true',
						help='')
	parser.add_argument('--model_type', type=str, default='DSPN',
						help='Model Type, one of - DSPN, deepset, set_transformer, submodint, submodint_f, hybrid (w submodint + (1 - w) DSF)')
	parser.add_argument('--submodint_gamma', type=float, default=2.5,
						help='Submod-Int gamma; sets the curvature C = gamma/2 log(gamma (set_size - 1) + 1) of phi_inter')
	parser.add_argument('--hybrid_w_init', type=float, default=0.5,
						help='model_type hybrid: initial w of w SubmodInt + (1 - w) DSF; w is learned')
	parser.add_argument('--anneal_from', type=int, default=None,
						help='Cyclic scheduler: after this epoch decay the learning rate linearly to 0 over --anneal_epochs, then stop')
	parser.add_argument('--anneal_epochs', type=int, default=25,
						help='Length of the --anneal_from phase')
	parser.add_argument('--fl_on_the_fly', action='store_true',
						help='Compute the target FL kernel from features on the GPU instead of storing the n x n matrix')
	

def add_training_args(parser):
	parser.add_argument('--load_from_ckpt', type=str, default=None,
					help='If want to load from ckpt')
	parser.add_argument('--bsz', type=int, default=16,
						help='Batch Size')
	parser.add_argument('--target_computation_bsz', type=int, default=40,
						help='Batch Size')
	parser.add_argument('--accumulation_steps', type=int, default=1,
						help='For Gradient Accumulation')
	parser.add_argument('--local_rank', type=int, default=0,
						help='GPU idx in DDP')
	parser.add_argument('--lr', type=float, default=1e-3,
						help='Initial Learning rate')
	parser.add_argument('--n_epochs', type=int, default=500,
						help='Number of Epochs')
	parser.add_argument('--scheduler', type=str, default=None,
						help='Can support Cosine/exponential (reduces after every 5 epochs)')
	parser.add_argument('--path_directory', type=str, default=None,
					help='path_directory for logging')

	parser.add_argument('--lr_gamma', type=float, default=1.,
						help='Exponentially decays learning rate by Multiply with this factor. By default, no reduction')
	parser.add_argument('--min_lr', type=float, default=1e-4,
						help='Minimum Learning Rate')
	parser.add_argument('--roof_lr_scale', type=float, default=1.,
						help="With --scheduler cyclic, the set function's learning rate as a multiple of the scheduled rate")

	parser.add_argument('--activity_reg_coefficient', type=float, default=0.,
						help='Activity reg')
	parser.add_argument('--dsf_weight_decay', type=float, default=0.,
						help='wd for dsf')
	parser.add_argument('--f_weight_decay', type=float, default=0.,
						help='wd for feature')
	parser.add_argument('--baseline', type=str, default='DSPN',
						help='One of DSPN, L2, max_margin')
	
	parser.add_argument('--roof_consistency_loss', type=float, default=0.,
						help='\lambda_1 in the Objective mentioned in the paper Eq 6')
	
	parser.add_argument('--gain_loss', type=float, default=0.,
						help='\lambda_3 in the Objective mentioned in the paper Eq 6')
	
	
	parser.add_argument('--nested_peripteral_loss_with_augmentation', type=float, default=0.,
						help='Recall that we have multiple views, so we can contrast E of view 1 with M of view 2. This while is not needed for smaller datasets, for larger datasets, having this doesnt do harm')
	
	parser.add_argument('--singleton_roof_consistency_loss', type=float, default=0.,
						help='\lambda_2 in the Objective mentioned in the paper Eq 6')
	
	parser.add_argument('--singleton_gain_loss', type=float, default=0.,
						help='\lambda_4 in the Objective mentioned in the paper Eq 6')
	
def add_args(parser):
	add_data_args(parser)
	add_architectural_args(parser)
	add_training_args(parser)