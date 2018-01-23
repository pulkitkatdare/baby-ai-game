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
from functools import reduce
import operator
from itertools import count
import time


class Minimalist_ActorCritic_Agent(object):

    # args: seed, cuda, gamma, tau, entropy_coef, batch_size, value_loss_coef

    def __init__(self, env, args, device=None,
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
        self.env = env
        self.model_args = model_args
        self.cuda = self.args.cuda and torch.cuda.is_available()
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

        if hasattr(self.env, 'seed'):
            self.env.seed(self.args.seed)
        self.model = None
        if model_args is not None:
            self.model = Model(**model_args)
            if self.cuda:
                self.model.cuda(device)

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

    def process_reward(self, reward):
        reward = reward / self.reward_scale
        if self.reward_clip:
            reward = min(max(reward, self.reward_min), self.reward_max)
        return reward

    def process_state(self, state, volatile=False):
        if not isinstance(state, dict):
            state = {'image': state, 'mission': '', 'advice': ''}

        state['mission'] = self.process_string(state['mission'])
        state['advice'] = self.process_string(state['advice'])

        img = np.expand_dims(np.transpose(state['image'], (2, 0, 1)), 0)
        img = img / self.input_scale
        order = np.expand_dims((state['mission']), 0)
        advice = np.expand_dims((state['advice']), 0)

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

    def optimize_model(self, values, log_probs, rewards, entropies):

        R = values[-1]
        gae = torch.zeros(1, 1).type_as(R.data)

        # Base A3C Loss
        policy_loss, value_loss = 0, 0

        # Performing update
        for i in reversed(range(len(rewards))):
            # Value function loss
            R = self.args.gamma * R + rewards[i]
            value_loss = value_loss + 0.5 * (R - values[i]).pow(2)

            # Generalized Advantage Estimataion
            delta_t = rewards[i] + self.args.gamma * \
                values[i + 1].data - values[i].data
            gae = gae * self.args.gamma * self.args.tau + delta_t

            # Computing policy loss
            policy_loss = policy_loss - \
                log_probs[i] * Variable(gae) - self.args.entropy_coef * entropies[i]

        # print('optim')
        self.optimizer.zero_grad()

        # Back-propagation
        total_loss = (policy_loss + self.args.value_loss_coef * value_loss)
        total_loss.backward()

        # Apply updates
        self.optimizer.step()
        return total_loss.cpu().data.numpy()[0, 0]

    def _train_episodes(self, max_iters, start_epoch, start_step,
                        max_fwd_steps=None, path='.'):

        step = start_step
        step_elapsed = 0
        updatecount = 0
        updateprintcount = 0

        chekoutindex = int(step / self.checkpoint_every) + 1  # 1

        self.model.train()

        current_episode = start_epoch
        current_episode_reward = 0.0
        current_episode_step = 0
        total_loss = 0
        print_loss_total = 0  # Reset every log_interval
        epoch_loss_total = 0  # Reset every epoch

        start = time.time()

        state = self.env.reset()
        state = self.process_state(state)
        self.model.reset_hidden_states()

        prev_terminal_end = False
        print_loss_log = False
        runningReward = 0

        while step < max_iters:

            # collect the data for the current episode
            episode_length = 0
            values = []
            log_probs = []
            rewards = []
            entropies = []
            # actions = []
            # states = []
            curr_reward = 0.0
            terminal_end = False
            # kind of truncated backprop through the time
            self.model.detach_hidden_states()

            while True:
                episode_length += 1
                current_episode_step += 1
                step_elapsed += 1

                logit, value = self.model(state)

                # Calculate entropy from action probability distribution
                action_probs = F.softmax(logit, dim=1)
                action_log_prob = F.log_softmax(logit, dim=1)
                entropy = -(action_log_prob * action_probs).sum(1)

                # Take an action from distribution
                m = Categorical(action_probs)
                actionV = m.sample()
                action = actionV.data.cpu().numpy()[0]
                select_action_log_proba = action_log_prob.gather(
                    1, actionV.view(-1, 1))

                # Perform the action on the environment
                nextstate, reward, done, _ = self.env.step(action)
                next_state = self.process_state(nextstate)
                reward = self.process_reward(reward)

                # Cumul reward
                curr_reward += reward
                current_episode_reward += reward

                # Store informations
                # # states.append(state)
                # # actions.append(action)
                values.append(value)
                rewards.append(reward)
                entropies.append(entropy)
                log_probs.append(select_action_log_proba)

                if done:
                    terminal_end = True
                    state = self.env.reset()
                    state = self.process_state(state)
                    self.model.reset_hidden_states()

                    break
                else:
                    state = next_state
                    if (max_fwd_steps is not None) and (
                            episode_length >= max_fwd_steps):
                        break

            R = torch.zeros(1, 1).type_as(values[-1].data)
            if not terminal_end:
                _, value = self.model(state)
                R = value.data
            values.append(Variable(R))

            if sum(rewards) > 0:
                print('SUCCESS Reward Optim: ', sum(rewards),
                      ' - ', current_episode)

            runningReward = runningReward * 0.99 + sum(rewards) * 0.01

            loss = self.optimize_model(values, log_probs, rewards, entropies)

            # if print_loss_log or sum(rewards) > 0:
            #     print('loss: ', current_episode, ': ', loss,
            #           'rew: ', sum(rewards))

            # Record average loss
            total_loss += loss
            print_loss_total += loss
            epoch_loss_total += loss

            step += episode_length
            updatecount += 1
            updateprintcount += 1
            epoch_loss_avg = epoch_loss_total / updatecount

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
                print('Episode {}:{} -> Cur Rew {} Episode Loss {} Run Rew {}'.
                      format(current_episode, current_episode_step,
                             current_episode_reward, epoch_loss_total,
                             runningReward))

            prev_terminal_end = False
            if terminal_end:
                if current_episode_reward > 0:
                    prev_terminal_end = True
                    print_loss_log = True
                current_episode += 1
                epoch_loss_total = 0
                updatecount = 0
                current_episode_reward = 0.0
                current_episode_step = 0
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

        step = 0
        episode_length = 0
        reward_sum = 0

        start_time = time.time()

        while True:
            episode_length += 1
            step += 1

            logit, value = self.model(state)

            # Calculate entropy from action probability distribution
            action_probs = F.softmax(logit, dim=1)
            m = Categorical(action_probs)
            action = m.sample()
            action = action.data.cpu().numpy()[0]

            nextstate, reward, done, _ = self.env.step(action)
            if render:
                self.env.render()
                time.sleep(0.1)
            next_state = self.process_state(nextstate)

            if max_iters is not None:
                done = done or step >= max_iters
            reward_sum += reward

            if not done:
                state = next_state
            else:
                print(
                    "Time {}, steps: {}, FPS: {:.0f}, rew: {}, len: {}".format(
                        time.strftime("%Hh %Mm %Ss",
                                      time.gmtime(time.time() - start_time)
                                      ),
                        step,
                        step / (time.time() - start_time),
                        reward_sum,
                        episode_length
                    ), ' ', mission
                )
                reward_sum = 0
                episode_length = 0
                state = self.env.reset()
                state = self.process_state(state)
                self.model.reset_hidden_states()
                time.sleep(1)

            if (max_iters is not None) and (step >= max_iters):
                break