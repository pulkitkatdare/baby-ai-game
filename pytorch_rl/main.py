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

from arguments import get_args
from vec_env.dummy_vec_env import DummyVecEnv
from vec_env.subproc_vec_env import SubprocVecEnv
from envs import make_env
from kfac import KFACOptimizer
from model import RecMLPPolicy, MLPPolicy, CNNPolicy
from storage import RolloutStorage
from visualize import visdom_plot
import preProcess

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
    
    #to be deleted after debug
    global envs,obs
    
    
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
    modelSize = 0
    for p in actor_critic.parameters():
        pSize = reduce(operator.mul, p.size(), 1)
        modelSize += pSize
    print(str(actor_critic))
    print('Total model size: %d' % modelSize)

    if envs.action_space.__class__.__name__ == "Discrete":
        action_shape = 1
    else:
        action_shape = envs.action_space.shape[0]

    if args.cuda:
        actor_critic.cuda()

    if args.algo == 'a2c':
        optimizer = optim.RMSprop(actor_critic.parameters(), args.lr, eps=args.eps, alpha=args.alpha)
    elif args.algo == 'ppo':
        optimizer = optim.Adam(actor_critic.parameters(), args.lr, eps=args.eps)
    elif args.algo == 'acktr':
        optimizer = KFACOptimizer(actor_critic)

    maxSizeOfMissionsSelected=200
    rollouts = RolloutStorage(args.num_steps, args.num_processes, obs_shape, envs.action_space, actor_critic.state_size,maxSizeOfMissions=maxSizeOfMissionsSelected)
    current_obs = torch.zeros(args.num_processes, *obs_shape)
    
    preProcessor=preProcess.PreProcessor()
    current_missions=torch.zeros(args.num_processes, maxSizeOfMissionsSelected)

    
    def update_current_obs(obs,missions):
        #print('top')
        shape_dim0 = envs.observation_space.shape[0]
        #img,txt = torch.from_numpy(np.stack(obs[:,0])).float(),np.stack(obs[:,1])

        images = torch.from_numpy(obs)
        if args.num_stack > 1:
            current_obs[:, :-shape_dim0] = current_obs[:, shape_dim0:]
        current_obs[:, -shape_dim0:] = images
        current_missions = missions

    obsF = envs.reset()
#    print('init')
#    print(obs)

    #print('obs : ', obs)
#    print(len(obs))
#    print(obs[0])
    
    #obsF,reward,done,info=envs.step(np.ones((args.num_processes)))
    #print('after 1 step')
    #print(obs)

    obs=np.array([preProcessor.preProcessImage(dico['image']) for dico in obsF])
    missions=torch.stack([preProcessor.stringEncoder(dico['mission']) for dico in obsF])
    bestActions=Variable(torch.stack( [ torch.Tensor(dico['bestActions']) for dico in obsF ] ))
    #print(missions)
    #print('missions size',missions.size())
    #print(len(obs[0]))
    #print(obs)
    
    
    def getMissionsAsVariables(step,end=False):
        '''
        Allow to convert from list of ASCII codes to pytorch Variables
        the argument step allows point-wise selection in the rollout
        the argument end allows to access a whole part of the memory according to
        missions[step:end]
        '''
        
        #get the missions as ASCII codes
        if end is not False:
            tmpMissions=rollouts.missions[step:end].view(-1,200)
            #convert them to pytorch tensors using the language model
            tmpMissions=preProcessor.adaptToTorchVariable(tmpMissions)
            #convert them as Variables
            missionsVariable=Variable(tmpMissions)       
        else:
            tmpMissions=rollouts.missions[step]
            #convert them to pytorch tensors using the language model
            tmpMissions=preProcessor.adaptToTorchVariable(tmpMissions)
            #convert them as Variables
            missionsVariable=Variable(tmpMissions,volatile=True)
       
      
        
        #check if cuda is available
        if args.cuda:
            missionsVariable=missionsVariable.cuda()
        return(missionsVariable)
    
    def correctReward(reward, cpu_actions,cpu_teaching_actions):
        '''
        defines the correction on the reward to apply in order to take account of the fact
        that in mode teacher the agent might choose wrong actions while actually 
        applying right actions, because actions are overwriten in the teacher mode
        '''
        si=len(cpu_actions)
        output=0
        #print('chosen ',cpu_actions)
        #print('teaching ',cpu_teaching_actions)
        #print('reward', reward)
        for i in range(si):
            if int(cpu_actions[i]) != int (cpu_teaching_actions[i]):
                reward[i]-=2
        return(output)
                
#    
#    
    #envs.getText()
    #print(txt)
    update_current_obs(obs,missions)

    rollouts.observations[0].copy_(current_obs)
    rollouts.missions[0].copy_(current_missions)
    # These variables are used to compute average rewards for all processes.
    episode_rewards = torch.zeros([args.num_processes, 1])
    final_rewards = torch.zeros([args.num_processes, 1])

    if args.cuda:
        current_obs = current_obs.cuda()
        current_missions=current_missions.cuda()
        bestActions=bestActions.cuda()
        rollouts.cuda()

    start = time.time()
    for j in range(num_updates):
        for step in range(args.num_steps):
            
            if step%2==0:
                useInfo=True
            else:
                useInfo=False
            useInfo=False
            
            #preprocess the missions to be used by the model
            if useInfo:
                missionsVariable=getMissionsAsVariables(step)
            else:
                missionsVariable=False
            
            # Sample actions
            value, action, action_log_prob, states = actor_critic.act(
                Variable(rollouts.observations[step], volatile=True),
                Variable(rollouts.states[step], volatile=True),
                Variable(rollouts.masks[step], volatile=True),
                missions=missionsVariable
            )
            
            cpu_actions = action.data.squeeze(1).cpu().numpy()
            
            cpu_teaching_actions=bestActions.data.squeeze(1).cpu().numpy()
            
            # Obser reward and next obs
            #print('actions',cpu_actions)
            if useInfo:
                obsF, reward, done, info = envs.step(cpu_teaching_actions)
                correctReward(reward,cpu_actions,cpu_teaching_actions)
                #print('corrected reward', reward)

            else:
                obsF, reward, done, info = envs.step(cpu_actions)
            
            ## get the image and mission observation from the observation dictionnary
            obs=np.array([preProcessor.preProcessImage(dico['image']) for dico in obsF])
            missions=torch.stack([preProcessor.stringEncoder(dico['mission']) for dico in obsF])
            bestActions=Variable(torch.stack( [ torch.Tensor(dico['bestActions']) for dico in obsF ] ))
            if args.cuda:
                bestActions=bestActions.cuda()


            reward = torch.from_numpy(np.expand_dims(np.stack(reward), 1)).float()
            episode_rewards += reward

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            final_rewards *= masks
            final_rewards += (1 - masks) * episode_rewards
            episode_rewards *= masks

            if args.cuda:
                masks = masks.cuda()

            if current_obs.dim() == 4:
                current_obs *= masks.unsqueeze(2).unsqueeze(2)
            else:
                current_obs *= masks
            current_missions *= masks

                
            #update current observation and save it in the storage memory
            update_current_obs(obs,missions)
            rollouts.insert(step, current_obs, current_missions, states.data, action.data, action_log_prob.data, value.data, reward, masks)

        if useInfo:
            missionsVariable=getMissionsAsVariables(-1)
        else:
            missionsVariable=False
            
        next_value = actor_critic(
            Variable(rollouts.observations[-1], volatile=True),
            Variable(rollouts.states[-1], volatile=True),
            Variable(rollouts.masks[-1], volatile=True),
            missions=missionsVariable
        )[0].data

        rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.tau)

        if args.algo in ['a2c', 'acktr']:
            
            missionsVariable=getMissionsAsVariables(0,end=-1)
            values, action_log_probs, dist_entropy, states = actor_critic.evaluate_actions(
                Variable(rollouts.observations[:-1].view(-1, *obs_shape)),
                Variable(rollouts.states[:-1].view(-1, actor_critic.state_size)),
                Variable(rollouts.masks[:-1].view(-1, 1)),
                Variable(rollouts.actions.view(-1, action_shape)),
                missions=missionsVariable
            )

            values = values.view(args.num_steps, args.num_processes, 1)
            action_log_probs = action_log_probs.view(args.num_steps, args.num_processes, 1)

            advantages = Variable(rollouts.returns[:-1]) - values
            value_loss = advantages.pow(2).mean()

            action_loss = -(Variable(advantages.data) * action_log_probs).mean()

            if args.algo == 'acktr' and optimizer.steps % optimizer.Ts == 0:
                # Sampled fisher, see Martens 2014
                actor_critic.zero_grad()
                pg_fisher_loss = -action_log_probs.mean()

                value_noise = Variable(torch.randn(values.size()))
                if args.cuda:
                    value_noise = value_noise.cuda()

                sample_values = values + value_noise
                vf_fisher_loss = -(values - Variable(sample_values.data)).pow(2).mean()

                fisher_loss = pg_fisher_loss + vf_fisher_loss
                optimizer.acc_stats = True
                fisher_loss.backward(retain_graph=True)
                optimizer.acc_stats = False

            optimizer.zero_grad()
            (value_loss * args.value_loss_coef + action_loss - dist_entropy * args.entropy_coef).backward()

            ## CLIP THE GRADIENT 
            if args.algo == 'a2c':
                nn.utils.clip_grad_norm(actor_critic.parameters(), args.max_grad_norm)

            optimizer.step()
        elif args.algo == 'ppo':
            advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

            for e in range(args.ppo_epoch):
                if args.recurrent_policy:
                    data_generator = rollouts.recurrent_generator(advantages, args.num_mini_batch)
                else:
                    data_generator = rollouts.feed_forward_generator(advantages, args.num_mini_batch)

                for sample in data_generator:
                    observations_batch, missions_batch, states_batch, actions_batch, \
                       return_batch, masks_batch, old_action_log_probs_batch, \
                            adv_targ = sample

                    # Reshape to do in a single forward pass for all steps
                    values, action_log_probs, dist_entropy, states = actor_critic.evaluate_actions(
                        Variable(observations_batch),
                        Variable(states_batch),
                        Variable(masks_batch),
                        Variable(actions_batch)
                    )

                    adv_targ = Variable(adv_targ)
                    ratio = torch.exp(action_log_probs - Variable(old_action_log_probs_batch))
                    surr1 = ratio * adv_targ
                    surr2 = torch.clamp(ratio, 1.0 - args.clip_param, 1.0 + args.clip_param) * adv_targ
                    action_loss = -torch.min(surr1, surr2).mean() # PPO's pessimistic surrogate (L^CLIP)

                    value_loss = (Variable(return_batch) - values).pow(2).mean()

                    optimizer.zero_grad()
                    (value_loss + action_loss - dist_entropy * args.entropy_coef).backward()
                    nn.utils.clip_grad_norm(actor_critic.parameters(), args.max_grad_norm)
                    optimizer.step()

        rollouts.after_update()

        if j % args.save_interval == 0 and args.save_dir != "":
            #print('current advice',envs.s)
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            # A really ugly way to save a model to CPU
            save_model = actor_critic
            if args.cuda:
                save_model = copy.deepcopy(actor_critic).cpu()

            save_model = [save_model,
                            hasattr(envs, 'ob_rms') and envs.ob_rms or None]

            torch.save(save_model, os.path.join(save_path, args.env_name + ".pt"))

        if j % args.log_interval == 0:
            end = time.time()
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            print("Updates {}, num timesteps {}, FPS {}, mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}, entropy {:.5f}, value loss {:.5f}, policy loss {:.5f}".
                format(j, total_num_steps,
                       int(total_num_steps / (end - start)),
                       final_rewards.mean(),
                       final_rewards.median(),
                       final_rewards.min(),
                       final_rewards.max(), dist_entropy.data[0],
                       value_loss.data[0], action_loss.data[0]))
        if args.vis and j % args.vis_interval == 0:
            try:
                # Sometimes monitor doesn't properly flush the outputs
                win = visdom_plot(viz, win, args.log_dir, args.env_name, args.algo)
            except IOError:
                pass

if __name__ == "__main__":
    main()
