import torch
import random
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical
from collections import namedtuple
from .utils import *
from .model import Model
from .optim import Optimizer
from .vec_env import DummyVecEnv, SubprocVecEnv
from functools import reduce
import operator
from itertools import count
import time


class Generic_ActorCritic_Agent_Base(object):

    # args: seed, cuda, gamma, tau, entropy_coef, batch_size, value_loss_coef

    def __init__(self, make_env_fn, args, device=None,
                 expt_dir='experiment', checkpoint_every=100, log_interval=100,
                 input_scale=1.0,
                 reward_scale=1.0,
                 reward_clip=False,
                 reward_min=-1,
                 reward_max=1,
                 process_string_func=None,
                 pad_sym='<pad>',
                 model_args=None):
        self.args = args
        torch.manual_seed(self.args.seed)
        self.cuda = self.args.cuda and torch.cuda.is_available()
        if self.cuda:
            torch.cuda.manual_seed(self.args.seed)

        self.model_args = model_args
        self.device = device
        self.input_scale = input_scale
        self.reward_scale = reward_scale
        self.reward_clip = reward_clip
        self.reward_min = reward_min
        self.reward_max = reward_max
        self.process_string = process_string_func

        if self.process_string is None:
            self.process_string = lambda x: [0]
        else:
            self.pad_sym = pad_sym
            res = self.process_string(pad_sym)
            self.pad_id = res[0]
            assert self.pad_id == 0, 'The pad_id must be zero.'

        self.model = None
        if model_args is not None:
            self.env = [make_env_fn(self.args.env, self.args.seed, i)
                        for i in range(self.args.num_processes)]
            self.model = Model(**model_args)
            if self.cuda:
                self.model.cuda(device)
        else:
            self.env = [make_env_fn(self.args.env, self.args.seed, i)
                        for i in range(1)]

        if len(self.env) == 1:
            self.env = DummyVecEnv(self.env)
        else:
            self.env = SubprocVecEnv(self.env)  # DummyVecEnv(self.env)  #

        if self.model is not None:
            modelSize = 0
            for p in self.model.parameters():
                pSize = reduce(operator.mul, p.size(), 1)
                modelSize += pSize
            print('Model size: %d' % modelSize)

        self.optimizer = None
        self.checkpoint_every = checkpoint_every
        self.log_interval = log_interval

        if not os.path.isabs(expt_dir):
            expt_dir = os.path.join(os.getcwd(), expt_dir)
        self.expt_dir = expt_dir
        if not os.path.exists(self.expt_dir):
            os.makedirs(self.expt_dir)

    def process_reward(self, rewards):
        def proc(reward):
            reward = reward / self.reward_scale
            if self.reward_clip:
                reward = min(max(reward, self.reward_min), self.reward_max)
            return reward
        return [proc(r) for r in rewards]

    def repackage_hidden_states(self, h):
        """ Wraps hidden states in new Variables,
            to detach them from their history.
        """
        if h is None:
            return None
        if isinstance(h, torch.autograd.Variable):
            return torch.autograd.Variable(h.data)
        else:
            return tuple(self.repackage_hidden_states(v) for v in h)

    def get_hidden_states_size(self, h):
        """ Wraps hidden states in new Variables,
            to detach them from their history.
        """
        if h is None:
            return None
        if isinstance(h, torch.autograd.Variable):
            return h.size()
        else:
            return tuple(self.get_hidden_states_size(v) for v in h)

    def get_hidden_states_data(self, h, ind, unsqueeze=False):
        """ Wraps hidden states in new Variables,
            to detach them from their history.
        """
        if h is None:
            return None
        if isinstance(h, torch.autograd.Variable):
            vdata = h.data[:, ind]
            if unsqueeze:
                vdata = vdata.unsqueeze(1)
            return vdata
        else:
            return tuple(
                self.get_hidden_states_data(
                    v, ind, unsqueeze) for v in h)

    def concat_hidden_states_data(self, hlist, toVariable=False):
        """ Wraps hidden states in new Variables,
            to detach them from their history.
        """
        if hlist is None:
            return None
        if len(hlist) == 0:
            return hlist
        if isinstance(hlist[0], torch.Tensor):
            vdata = torch.cat(hlist, 1)
            if toVariable:
                vdata = torch.autograd.Variable(vdata)
            return vdata
        else:
            ziphlist = zip(*hlist)
            return tuple(
                self.concat_hidden_states_data(
                    v, toVariable
                ) for v in ziphlist
            )

    def zero_like_hidden_states(self, h, volatile=False):
        """ Wraps hidden states in new Variables,
            to detach them from their history.
        """
        if h is None:
            return None
        if isinstance(h, torch.autograd.Variable):
            return torch.autograd.Variable(
                torch.zeros(h.data.size()).type_as(h.data),
                volatile=volatile)
        else:
            return tuple(self.zero_like_hidden_states(
                v, volatile) for v in h)

    def process_state(self, state, volatile=False):
        max_len_mission = []
        max_len_advice = []
        for i in range(len(state)):
            if not isinstance(state[i], dict):
                state[i] = {'image': state[i], 'mission': '', 'advice': ''}
            state[i]['image'] = np.expand_dims(
                np.transpose(state[i]['image'], (2, 0, 1)), 0)
            state[i]['mission'] = self.process_string(state[i]['mission'])
            state[i]['advice'] = self.process_string(state[i]['advice'])

            max_len_mission.append(len(state[i]['mission']))
            max_len_advice.append(len(state[i]['advice']))

        max_len_mission = max(max_len_mission)
        max_len_advice = max(max_len_advice)

        for i in range(len(state)):
            state[i]['mission'].extend(
                [self.pad_id] * (max_len_mission - len(state[i]['mission'])))
            state[i]['advice'].extend(
                [self.pad_id] * (max_len_advice - len(state[i]['advice'])))

        img = np.concatenate([state[i]['image']
                              for i in range(len(state))], axis=0)
        img = img / self.input_scale
        order = np.vstack([state[i]['mission']
                           for i in range(len(state))])
        advice = np.vstack([state[i]['advice']
                            for i in range(len(state))])

        # img = np.expand_dims(np.transpose(state['image'], (2, 0, 1)), 0)
        # img = img / self.input_scale
        # order = np.expand_dims((state['mission']), 0)
        # advice = np.expand_dims((state['advice']), 0)

        if self.cuda:
            img = torch.from_numpy(img).type(torch.cuda.FloatTensor)
            order = torch.from_numpy(order).type(torch.cuda.LongTensor)
            advice = torch.from_numpy(advice).type(torch.cuda.LongTensor)
        else:
            img = torch.from_numpy(img).type(torch.FloatTensor)
            order = torch.from_numpy(order).type(torch.LongTensor)
            advice = torch.from_numpy(advice).type(torch.LongTensor)

        return State(
            Variable(img, volatile=volatile),
            Variable(order, volatile=volatile),
            Variable(advice, volatile=volatile)
        )

    def compute_roullout_returns(self, values, rewards, masks):
        num_processes = rewards[0].size(0)
        returns = torch.zeros(
            len(rewards), num_processes, 1).type_as(rewards[0])

        # Generalized Advantage Estimataion
        gae = 0
        for i in reversed(range(len(rewards))):
            delta_t = rewards[i] + self.args.gamma * \
                masks[i + 1].data * values[i + 1].data - values[i].data
            gae = gae * masks[i + 1].data * \
                self.args.gamma * self.args.tau + delta_t
            returns[i] = gae + values[i].data

        return returns

    def recurrent_ppo_minibatch_generator(
            self, states, actions, log_probs, returns, advantages,
            hidden_states, masks, num_mini_batch):
        num_processes = returns.size(1)
        num_envs_per_batch = num_processes // num_mini_batch
        perm = torch.randperm(num_processes)

        bactions = torch.cat(actions, 0).view(len(actions), *actions[0].size())
        bmasks = torch.cat(masks, 0).view(len(masks), *masks[0].size())
        blog_probs = torch.cat(log_probs, 0).view(
            len(log_probs), *log_probs[0].size())

        bstates = State(*zip(*states))
        bstates_image = torch.cat(bstates.image, 0).data.view(
            len(states), *bstates.image[0].size())
        max_order_len = max([t.size(1) for t in bstates.mission])
        max_advice_len = max([t.size(1) for t in bstates.advice])
        bstates_mission = torch.zeros(
            len(bstates.mission), num_processes,
            max_order_len
        ).type_as(bstates.mission[0].data)
        bstates_advice = torch.zeros(
            len(bstates.advice), num_processes,
            max_advice_len
        ).type_as(bstates.advice[0].data)
        for i in range(len(states)):
            bstates_mission[
                i, :, 0:bstates.mission[i].size(1)
            ] = bstates.mission[i].data

            bstates_advice[
                i, :, 0:bstates.advice[i].size(1)
            ] = bstates.advice[i].data

        for start_ind in range(0, num_processes, num_envs_per_batch):
            state_image_batch = []
            state_mission_batch = []
            state_advice_batch = []
            hidden_states_batch = []
            actions_batch = []
            return_batch = []
            masks_batch = []
            log_probs_batch = []
            adv_targ = []

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]
                state_image_batch.append(bstates_image[:, ind])
                state_mission_batch.append(bstates_mission[:, ind])
                state_advice_batch.append(bstates_advice[:, ind])
                hidden_states_batch.append(
                    self.get_hidden_states_data(hidden_states[0], ind, True))
                actions_batch.append(bactions.data[:, ind])
                return_batch.append(returns[:, ind])
                masks_batch.append(bmasks.data[:-1, ind])
                log_probs_batch.append(blog_probs.data[:, ind])
                adv_targ.append(advantages[:, ind])

            state_image_batch = torch.cat(state_image_batch, 0)
            state_mission_batch = torch.cat(state_mission_batch, 0)
            state_advice_batch = torch.cat(state_advice_batch, 0)
            hidden_states_batch = self.concat_hidden_states_data(
                hidden_states_batch, True)

            actions_batch = torch.cat(actions_batch, 0)
            return_batch = torch.cat(return_batch, 0)
            masks_batch = torch.cat(masks_batch, 0)
            log_probs_batch = torch.cat(log_probs_batch, 0)
            adv_targ = torch.cat(adv_targ, 0)

            state_batch = State(
                Variable(state_image_batch),
                Variable(state_mission_batch),
                Variable(state_advice_batch),
            )

            yield state_batch, hidden_states_batch, actions_batch, \
                return_batch, masks_batch, log_probs_batch, adv_targ

    def a2c_optim_loss(self, values, log_probs, rewards, entropies, masks):
        R = values[-1]
        gae = torch.zeros(R.size(0), 1).type_as(R.data)

        # Base A3C Loss
        policy_loss, value_loss = 0, 0

        # Performing update
        for i in reversed(range(len(rewards))):
            # Value function loss
            R = self.args.gamma * masks[i + 1] * R + Variable(rewards[i])
            value_loss = value_loss + 0.5 * (R - values[i]).pow(2)

            # Generalized Advantage Estimataion
            delta_t = rewards[i] + self.args.gamma * \
                masks[i + 1].data * values[i + 1].data - values[i].data
            gae = gae * masks[i + 1].data * \
                self.args.gamma * self.args.tau + delta_t

            # Computing policy loss
            policy_loss = policy_loss - \
                log_probs[i] * Variable(gae) - self.args.entropy_coef * entropies[i]

        policy_loss = policy_loss.mean()
        value_loss = value_loss.mean()

        loss_total = policy_loss + self.args.value_loss_coef * value_loss

        # print('optim')
        self.optimizer.zero_grad()

        # Back-propagation
        loss_total.backward()

        # Apply updates
        self.optimizer.step()

        return loss_total.cpu().data.numpy()[0]

    def ppo_optim_loss(
            self, states, actions, values, log_probs, rewards, entropies,
            hidden_states, masks):
        # num process
        num_processes = rewards[0].size(0)

        # computing the rollout returns
        returns = self.compute_roullout_returns(values, rewards, masks)
        batchVal = torch.cat(values[:-1], 0).view_as(returns)

        advantages = returns - batchVal.data
        advantages = (advantages - advantages.mean()) / \
            (advantages.std() + 1e-5)

        self.args.ppo_epoch = 4
        self.args.ppo_mini_batch = 32
        self.args.ppo_clip_param = 0.2

        nupdates = 0
        loss_mean = 0.0

        for e in range(self.args.ppo_epoch):
            data_generator = self.recurrent_ppo_minibatch_generator(
                states, actions, log_probs, returns, advantages,
                hidden_states, masks,
                min(self.args.ppo_mini_batch, num_processes)
            )
            for sample in data_generator:
                state_batch, hidden_states_batch, actions_batch, \
                    return_batch, masks_batch, log_probs_batch, adv_targ = sample

                # Reshape to do in a single forward pass for all steps
                logit, value, _ = self.model.forward_with_hidden_states(
                    state_batch, hidden_states_batch, Variable(masks_batch))

                blog_probs = F.log_softmax(logit, dim=1)
                bprobs = F.softmax(logit, dim=1)
                baction_log_probs = blog_probs.gather(
                    1, Variable(actions_batch))
                bdist_entropy = -(blog_probs * bprobs).sum(-1).mean()

                adv_targ = Variable(adv_targ)
                ratio = torch.exp(
                    baction_log_probs -
                    Variable(log_probs_batch))
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.args.ppo_clip_param,
                    1.0 + self.args.ppo_clip_param
                ) * adv_targ

                # PPO's pessimistic surrogate (L^CLIP)
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = (Variable(return_batch) - value).pow(2).mean()

                total_loss = value_loss + policy_loss - bdist_entropy * self.args.entropy_coef

                # print('optim')
                self.optimizer.zero_grad()

                # Back-propagation
                total_loss.backward()

                # Apply updates
                self.optimizer.step()

                nupdates += 1

                loss_mean += total_loss.cpu().data.numpy()[0]

        return loss_mean / max(1, nupdates)

    def optimize_model(
            self,
            states,
            actions,
            values,
            log_probs,
            rewards,
            entropies,
            hidden_states,
            masks):

        # # loss
        # total_loss = self.a2c_optim_loss(
        #     values, log_probs, rewards, entropies, masks)

        total_loss = self.ppo_optim_loss(
            states, actions, values, log_probs, rewards, entropies,
            hidden_states, masks
        )

        return total_loss

    def _train_episodes(self, max_iters, start_epoch, start_step,
                        max_fwd_steps=None, path='.'):

        step = start_step
        step_elapsed = 0
        updatecount = 0
        updateprintcount = 0

        chekoutindex = int(step / self.checkpoint_every) + 1  # 1

        self.model.train()

        current_episode = start_epoch
        current_episode_reward = torch.zeros(self.args.num_processes, 1)
        final_rewards = torch.zeros(self.args.num_processes, 1)
        current_episode_step = 0
        total_loss = 0
        print_loss_total = 0  # Reset every log_interval
        epoch_loss_total = 0  # Reset every epoch

        start = time.time()

        state = self.env.reset()
        state = self.process_state(state)
        self.model.reset_hidden_states()
        hidden_states = None

        episode_ts = np.zeros(self.args.num_processes, dtype='int')

        prev_terminal_end = False
        print_loss_log = False
        runningReward = 0

        if max_fwd_steps is None:
            max_fwd_steps = int(
                max_iters - step) // self.args.num_processes // 500

        while step < max_iters:

            # collect the data for the current episode
            episode_length = 0
            values = []
            log_probs = []
            rewards = []
            entropies = []
            hidden_states_list = []
            mask_list = []
            actions = []
            states = []
            curr_reward = torch.zeros(self.args.num_processes, 1)

            # kind of truncated backprop through the time
            self.model.detach_hidden_states()
            hidden_states = self.repackage_hidden_states(hidden_states)
            masks = torch.ones(self.args.num_processes, 1)
            if self.args.cuda:
                masks = masks.cuda()
            masks = Variable(masks)

            local_elapsed_steps = 0

            while True:
                episode_length += 1
                current_episode_step += self.args.num_processes
                step_elapsed += self.args.num_processes
                local_elapsed_steps += self.args.num_processes

                logit, value, next_hidden_states = self.model.forward_with_hidden_states(
                    state, hidden_states, masks)

                # Calculate entropy from action probability distribution
                action_probs = F.softmax(logit, dim=1)
                action_log_prob = F.log_softmax(logit, dim=1)
                entropy = -(action_log_prob * action_probs).sum(1)

                # Take an action from distribution
                m = Categorical(action_probs)
                actionV = m.sample()
                action = actionV.data.cpu().numpy()
                select_action_log_proba = action_log_prob.gather(
                    1, actionV.view(-1, 1))

                # Perform the action on the environment
                nextstate, reward, done, _ = self.env.step(action)
                next_state = self.process_state(nextstate)
                reward = self.process_reward(reward)
                np_reward = np.expand_dims(np.array(reward), 1)
                reward = torch.from_numpy(np_reward).type_as(logit.data)
                episode_ts += 1
                current_episode_step = np.sum(episode_ts)
                for (i, done_) in enumerate(done):
                    if done_:
                        episode_ts[i] = 0
                        current_episode_step -= 1
                # If done then clean the history of observations.
                previous_masks = masks
                masks = torch.FloatTensor(
                    [[0.0] if done_ else [1.0] for done_ in done])
                # Cumul reward
                curr_reward += reward
                current_episode_reward += reward
                final_rewards *= masks
                final_rewards += (1 - masks) * current_episode_reward
                current_episode_reward *= masks

                masks = Variable(masks)

                if hidden_states is None:
                    if next_hidden_states is not None:
                        hidden_states = self.zero_like_hidden_states(
                            next_hidden_states)

                # Store informations
                states.append(state)
                actions.append(actionV.view(-1, 1))
                values.append(value)
                rewards.append(reward)
                entropies.append(entropy)
                log_probs.append(select_action_log_proba)
                hidden_states_list.append(hidden_states)
                mask_list.append(previous_masks)

                state = next_state
                hidden_states = next_hidden_states
                if (max_fwd_steps is not None) and (
                        episode_length >= max_fwd_steps):
                    break

            _, value, _ = self.model.forward_with_hidden_states(
                state, hidden_states, masks)
            R = value.data
            values.append(Variable(R))
            mask_list.append(masks)

            sum_reward = sum([torch.sum(a) for a in rewards]) / len(rewards)

            if sum_reward > 0:
                print('SUCCESS Reward Optim: ', sum_reward,
                      ' - ', current_episode)

            runningReward = runningReward * 0.99 + sum_reward * 0.01

            loss = self.optimize_model(
                states, actions, values, log_probs, rewards,
                entropies, hidden_states_list, mask_list
            )

            # if print_loss_log or sum(rewards) > 0:
            #     print('loss: ', current_episode, ': ', loss,
            #           'rew: ', sum(rewards))

            # Record average loss
            total_loss += loss
            print_loss_total += loss
            epoch_loss_total += loss

            step += local_elapsed_steps
            updatecount += 1
            updateprintcount += 1
            epoch_loss_avg = epoch_loss_total / updatecount

            current_episode_reward_mean = torch.mean(current_episode_reward)

            # log interval
            if (current_episode % self.log_interval == 0) and (
                    step_elapsed > 0):
                print_loss_avg = print_loss_total / updateprintcount
                print_loss_total = 0
                updateprintcount = 0
                print_loss_log = False
                log_msg = 'Progress: %d %d%%, pr Loss: %.4f, Ep Loss: %.4f' % (
                    step,
                    step / max_iters * 100,
                    print_loss_avg,
                    epoch_loss_avg
                )
                print(log_msg)
                ep_step_max = max_fwd_steps * self.args.num_processes
                print(
                    'Ep {}:{}/{} -> Rew {:.3f} mean/median {:.3f}/{:.3f}, min/max {:.3f}/{:.3f} Ep Loss {} Run Rew {}'. format(
                        current_episode,
                        current_episode_step,
                        ep_step_max,
                        current_episode_reward_mean,
                        final_rewards.mean(),
                        final_rewards.median(),
                        final_rewards.min(),
                        final_rewards.max(),
                        epoch_loss_total,
                        runningReward))

            prev_terminal_end = False
            if current_episode_reward_mean > 0:
                prev_terminal_end = True
                print_loss_log = True
            current_episode += 1
            epoch_loss_total = 0
            updatecount = 0
            self.optimizer.update(epoch_loss_avg, current_episode)

            # Checkpoint
            if step >= chekoutindex * self.checkpoint_every or step >= max_iters:
                Checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=current_episode,
                    step=step).save(
                    self.expt_dir)
                chekoutindex += 1

        pass

    def train(
            self, max_iters, max_fwd_steps=None,
            resume=False, optimizer=None, path='.'):
        # If training is set to resume
        if resume:
            latest_checkpoint_path = Checkpoint.get_latest_checkpoint(
                self.expt_dir)
            resume_checkpoint = Checkpoint.load(latest_checkpoint_path)
            self.model = resume_checkpoint.model
            if self.cuda and torch.cuda.is_available():
                self.model.cuda(self.device)
            self.optimizer = resume_checkpoint.optimizer

            # A walk around to set optimizing parameters properly
            resume_optim = self.optimizer.optimizer
            defaults = resume_optim.param_groups[0]
            defaults.pop('params', None)
            self.optimizer.optimizer = resume_optim.__class__(
                self.model.parameters(), **defaults)

            start_epoch = resume_checkpoint.epoch
            step = resume_checkpoint.step
        else:
            start_epoch = 1
            step = 0
            if optimizer is None:
                optimizer = Optimizer(
                    optim.Adam(
                        self.model.parameters(), lr=self.args.lr),
                    max_grad_norm=self.args.max_grad_norm)
            self.optimizer = optimizer

        self._train_episodes(
            max_iters, start_epoch, step, max_fwd_steps, path=path)

        return self.model

    def test(self, max_iters=None, render=False):

        self.model.eval()

        state = self.env.reset()
        state = self.process_state(state)
        self.model.reset_hidden_states()
        hidden_states = None
        num_envs = state.image.size(0)
        mask = torch.ones(num_envs, 1)
        if self.args.cuda:
            masks = masks.cuda()

        current_episode_reward = torch.zeros(self.args.num_processes, 1)
        final_rewards = torch.zeros(self.args.num_processes, 1)
        episode_ts = np.zeros(self.args.num_processes, dtype='int')

        step = 0

        start_time = time.time()

        while True:
            step += num_envs

            logit, value, hidden_states = self.model.forward_with_hidden_states(
                state, hidden_states, Variable(masks))

            # Calculate entropy from action probability distribution
            action_probs = F.softmax(logit, dim=1)
            m = Categorical(action_probs)
            action = m.sample()
            action = action.data.cpu().numpy()

            nextstate, reward, done, _ = self.env.step(action)
            episode_ts += 1
            episode_length = np.sum(episode_ts)
            at_leat_one_done = False
            for (i, done_) in enumerate(done):
                if done_:
                    episode_ts[i] = 0
                    episode_length -= 1
                    at_leat_one_done = True
            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            reward = self.process_reward(reward)
            reward = torch.from_numpy(
                np.expand_dims(np.array(reward), 1)).float()
            next_state = self.process_state(nextstate)

            current_episode_reward += reward
            final_rewards *= masks
            final_rewards += (1 - masks) * current_episode_reward
            current_episode_reward *= masks

            if render:
                self.env.render()
                time.sleep(0.1)

            state = next_state

            if at_leat_one_done:

                print(
                    "Time {}, steps: {}, FPS: {:.0f}, rew mean/median {:.3f}/{:.3f}, min/max {:.3f}/{:.3f} , len: {}".format(
                        time.strftime("%Hh %Mm %Ss",
                                      time.gmtime(time.time() - start_time)
                                      ),
                        step,
                        step / (time.time() - start_time),
                        final_rewards.mean(),
                        final_rewards.median(),
                        final_rewards.min(),
                        final_rewards.max(),
                        episode_length
                    ), ' ', mission
                )

                time.sleep(1)

            if (max_iters is not None) and (step >= max_iters):
                break
