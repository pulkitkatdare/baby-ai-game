import copy
import glob
import os
import time
import operator
from functools import reduce

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from common.arguments import get_args
from agent import PPO, A2C, ACKTR
from vec_env.dummy_vec_env import DummyVecEnv
from vec_env.subproc_vec_env import SubprocVecEnv
from common.envs import make_env
from common.kfac import KFACOptimizer
from common.model import RecMLPPolicy, MLPPolicy, CNNPolicy
from common.storage import RolloutStorage
from common.visualize import visdom_plot

args = get_args()

assert args.algo in ['a2c', 'ppo', 'acktr']
if args.recurrent_policy:
	assert args.algo in ['a2c', 'ppo'], 'Recurrent policy is not implemented for ACKTR'

num_updates = int(args.num_frames) // args.num_steps // args.num_processes

torch.manual_seed(args.seed)
if args.cuda:
	torch.cuda.manual_seed(args.seed)

try:
	os.makedirs(args.log_dir)
except OSError:
	files = glob.glob(os.path.join(args.log_dir, '*.monitor.csv'))
	for f in files:
		os.remove(f)

def main():
	print("#######")
	print("WARNING: All rewards are clipped or normalized so you need to use a monitor (see envs.py) or visdom plot to get true rewards")
	print("#######")

	os.environ['OMP_NUM_THREADS'] = '1'

	if args.vis:
		from visdom import Visdom
		viz = Visdom()
		win = None

	envs = [make_env(args.env_name, args.seed, i, args.log_dir)
				for i in range(args.num_processes)]

	if args.num_processes > 1:
		envs = SubprocVecEnv(envs)
	else:
		envs = DummyVecEnv(envs)

	# Maxime: commented this out because it very much changes the behavior
	# of the code for seemingly arbitrary reasons
	#if len(envs.observation_space.shape) == 1:
	#    envs = VecNormalize(envs)

	obs_shape = envs.observation_space.shape
	obs_shape = (obs_shape[0] * args.num_stack, *obs_shape[1:])

	obs_numel = reduce(operator.mul, obs_shape, 1)

	if len(obs_shape) == 3 and obs_numel > 1024:
		actor_critic = CNNPolicy(obs_shape[0], envs.action_space, args.recurrent_policy)
	elif args.recurrent_policy:
		actor_critic = RecMLPPolicy(obs_numel, envs.action_space)
	else:
		actor_critic = MLPPolicy(obs_numel, envs.action_space)


	# Maxime: log some info about the model and its size
	# call function PPO.modelsize() for this to happen
	'''
	modelSize = 0
	for p in actor_critic.parameters():
		pSize = reduce(operator.mul, p.size(), 1)
		modelSize += pSize
	'''

	if envs.action_space.__class__.__name__ == "Discrete":
		action_shape = 1
	else:
		action_shape = envs.action_space.shape[0]

	if args.cuda:
		actor_critic.cuda()
	rollouts = RolloutStorage(args.num_steps, args.num_processes, obs_shape, envs.action_space, actor_critic.state_size)

	if args.algo == 'a2c':
		Agent = A2C(actor_critic, rollouts, args.lr, 
				args.eps, args.num_processes, obs_shape, args.use_gae, args.gamma,
				 args.tau, args.recurrent_policy, args.num_mini_batch, args.cuda, 
				 args.log_interval, args.vis, args.env_name, args.log_dir, args.entropy_coef,
				args.num_stack, args.num_steps, args.ppo_epoch, args.clip_param, 
				args.max_grad_norm, args.alpha, args.save_dir, args.vis_interval, 
				args.save_interval, num_updates, action_shape, args.value_loss_coef)

	elif args.algo == 'ppo':
		Agent = PPO(actor_critic, rollouts, args.lr, 
				args.eps, args.num_processes, obs_shape, args.use_gae, args.gamma,
				 args.tau, args.recurrent_policy, args.num_mini_batch, args.cuda, 
				 args.log_interval, args.vis, args.env_name, args.log_dir, args.entropy_coef,
				args.num_stack, args.num_steps, args.ppo_epoch, args.clip_param, 
				args.max_grad_norm, args.save_dir, args.vis_interval, args.save_interval, 
				num_updates, action_shape, args.value_loss_coef)

	elif args.algo == 'acktr':
		Agent = ACKTR(actor_critic, rollouts, args.lr, 
				args.eps, args.num_processes, obs_shape, args.use_gae, args.gamma,
				 args.tau, args.recurrent_policy, args.num_mini_batch, args.cuda, 
				 args.log_interval, args.vis, args.env_name, args.log_dir, args.entropy_coef,
				args.num_stack, args.num_steps, args.ppo_epoch, args.clip_param, 
				args.max_grad_norm, args.alpha, args.save_dir, args.vis_interval, 
				args.save_interval, num_updates, action_shape, args.value_loss_coef)
	print(str(actor_critic))
	print('Total model size: %d' % Agent.modelsize())

	obs = envs.reset()
	Agent.update_current_obs(obs, envs)
	Agent.rollouts.observations[0].copy_(Agent.current_obs)

	# These variables are used to compute average rewards for all processes.
	Agent.train(envs)

if __name__ == "__main__":
	main()
