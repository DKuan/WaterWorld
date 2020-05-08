# Time: 2019-11-05
# Author: Zachary 
# Name: waterworld
# File func: abstract class for maddpg
import time
import os

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

from replay_buffer import ReplayBuffer
from model import actor_agent, critic_agent

class MADDPG_Agent():
    def __init__(self, num_agents, obs_shape_n, action_shape_n, args):
        self.var = 1.0
        self.min_var = 0.08
        self.obs_size = []
        self.action_size = []
        self.update_cnt = 0
        self.last_update_episode = 0
        self.num_agents = num_agents
        self.obs_shape_n = obs_shape_n
        self.action_shape_n = action_shape_n
        self.memory = ReplayBuffer(args.memory_size)
        head_o, head_a, end_o, end_a = 0, 0, 0, 0
        for obs_shape, action_shape in zip(obs_shape_n, action_shape_n):
            end_o = end_o + obs_shape
            end_a = end_a + action_shape 
            range_o = (head_o, end_o)
            range_a = (head_a, end_a)
            self.obs_size.append(range_o)
            self.action_size.append(range_a)
            head_o = end_o
            head_a = end_a

    def init_trainers(self, args):
        """
        init the trainers or load the old model
        """
        self.actors_cur = [None for _ in range(self.num_agents)]
        self.critics_cur = [None for _ in range(self.num_agents)]
        self.actors_tar = [None for _ in range(self.num_agents)]
        self.critics_tar = [None for _ in range(self.num_agents)]
        self.optimizers_c = [None for _ in range(self.num_agents)]
        self.optimizers_a = [None for _ in range(self.num_agents)]
        input_size_global = sum(self.obs_shape_n) + sum(self.action_shape_n)

        if args.restore == True: # restore the model
            game_step = int(args.old_model_name.split('_')[-1][:-1])
            for idx in range(self.num_agents):
                self.actors_cur[idx] = torch.load(args.old_model_name+'a_c_{}.pt'.format(idx))
                self.actors_tar[idx] = torch.load(args.old_model_name+'a_t_{}.pt'.format(idx))
                self.critics_cur[idx] = torch.load(args.old_model_name+'c_c_{}.pt'.format(idx))
                self.critics_tar[idx] = torch.load(args.old_model_name+'c_t_{}.pt'.format(idx))
                self.optimizers_a[idx] = optim.Adam(self.actors_cur[idx].parameters(), args.lr_a)
                self.optimizers_c[idx] = optim.Adam(self.critics_cur[idx].parameters(), args.lr_c)
            self.var = self.var - (game_step-args.learning_start_episode*args.per_episode_max_len)*args.var_discount
            self.var = self.min_var if self.var < self.min_var else self.var
            old_data = {'game_step':game_step, 'episode_gone_old':int(game_step/args.per_episode_max_len)}

        # Note: if you need load old model, there should be a procedure for juding if the trainers[idx] is None
        for i in range(self.num_agents):
            self.actors_cur[i] = actor_agent(self.obs_shape_n[i], self.action_shape_n[i], \
                args).to(args.device)
            self.critics_cur[i] = critic_agent(sum(self.obs_shape_n), sum(self.action_shape_n), \
                args).to(args.device)
            self.actors_tar[i] = actor_agent(self.obs_shape_n[i], self.action_shape_n[i], \
                args).to(args.device)
            self.critics_tar[i] = critic_agent(sum(self.obs_shape_n), sum(self.action_shape_n), \
                args).to(args.device)
            self.optimizers_a[i] = optim.Adam(self.actors_cur[i].parameters(), args.lr_a)
            self.optimizers_c[i] = optim.Adam(self.critics_cur[i].parameters(), args.lr_c)

        # return the old data, no need to update the trainers
        if args.restore == True: return old_data

        self.actors_tar = self.update_trainers(self.actors_cur, self.actors_tar, 1.0) # update the target par using the cur
        self.critics_tar = self.update_trainers(self.critics_cur, self.critics_tar, 1.0) # update the target par using the cur
    
    def load_trainers(self, args):
        trainers_cur = []
        trainers_tar = []
        optimizers = []
        input_size = [8, 10, 10] # the obs size
        input_size_global = [23, 25, 25] # cal by README

        """ load the model """
        self.actors_cur = [torch.load(args.old_model_name+'a_c_{}.pt'.format(agent_idx), map_location=args.device) \
            for agent_idx in range(self.num_agents)]

    def update_trainers(self, agents_cur, agents_tar, tao):
        """
        update the trainers_tar par using the trainers_cur
        This way is not the same as copy_, but the result is the same
        out:
        |agents_tar: the agents with new par updated towards agents_current
        """
        for agent_c, agent_t in zip(agents_cur, agents_tar):
            key_list = list(agent_c.state_dict().keys())
            state_dict_t = agent_t.state_dict()
            state_dict_c = agent_c.state_dict()
            for key in key_list:
                state_dict_t[key] = state_dict_c[key]*tao + \
                        (1-tao)*state_dict_t[key] 
            agent_t.load_state_dict(state_dict_t)
        return agents_tar

    def select_actions(self, obs_n, args):
        out = [agent(torch.from_numpy(obs).to(args.device, torch.float), training=False).detach().cpu().numpy() \
            for agent, obs in zip(self.actors_cur, obs_n)]
        action_n = np.array([a + self.var*np.random.randn(a.shape[0]) for a in out])
        action_n = np.clip(action_n, -1, 1)
        return action_n

    def agents_train(self, game_step, episode_now, args):
        """ 
        use this func to make the "main" func clean
        par:
        |input: the data for training
        |output: the data for next update
        """
        # update all trainers, if not in display or benchmark mode
        if episode_now < args.learning_start_episode: return 
        if self.update_cnt > 0 and self.var >= self.min_var: self.var *= args.var_discount
        #if episode_now > self.last_update_episode and (episode_now - args.learning_start_episode) % args.learning_fre == 0:
        if game_step % args.learning_fre_step == 0:
            if self.update_cnt == 0: print('\r=start training ...'+' '*100)
            self.last_update_episode = episode_now
            self.update_cnt += 1

            # update every agent in different memory batch
            for agent_idx, (actor_c, actor_t, critic_c, critic_t, opt_a, opt_c) in \
                enumerate(zip(self.actors_cur, self.actors_tar, self.critics_cur, \
                    self.critics_tar, self.optimizers_a, self.optimizers_c)):
                # del if opt_c == None: continue # jump to the next model update

                # sample the experience
                _obs_n_o, _action_n, _rew_n, _obs_n_n, _done_n = self.memory.sample( \
                    args.batch_size, agent_idx) # Note_The func is not the same as others
                    
                # --use the date to update the CRITIC
                rew = torch.tensor(_rew_n, device=args.device, dtype=torch.float) # set the rew to gpu
                done_n = torch.tensor(~_done_n, dtype=torch.float, device=args.device) # set the rew to gpu
                action_cur_o = torch.from_numpy(_action_n).to(args.device, torch.float)
                obs_n_o = torch.from_numpy(_obs_n_o).to(args.device, torch.float)
                obs_n_n = torch.from_numpy(_obs_n_n).to(args.device, torch.float)

                action_tar = torch.cat([a_t(obs_n_n[:, self.obs_size[idx][0]:self.obs_size[idx][1]]).detach() \
                    for idx, a_t in enumerate(self.actors_tar)], dim=1)
                q = critic_c(obs_n_o, action_cur_o).reshape(-1) # q 
                q_ = critic_t(obs_n_n, action_tar).reshape(-1) # q_ 
                q_ = q_*args.gamma*done_n + rew*torch.tensor(args.reward_scale_par, device=args.device) # q_*gamma*done + reward
                loss_c = torch.nn.MSELoss()(q, q_.detach()) # bellman equation
                opt_c.zero_grad()
                loss_c.backward()
                nn.utils.clip_grad_norm_(critic_c.parameters(), args.max_grad_norm)
                opt_c.step()

                # --use the data to update the ACTOR
                # There is no need to cal other agent's action
                opt_c.zero_grad()
                model_out, policy_c_new = actor_c( \
                    obs_n_o[:, self.obs_size[agent_idx][0]:self.obs_size[agent_idx][1]], model_original_out=True)
                # update the aciton of this agent
                action_cur_o[:, self.action_size[agent_idx][0]:self.action_size[agent_idx][1]] = policy_c_new 
                loss_pse = torch.mean(torch.pow(model_out, 2))
                loss_a = torch.mul(torch.tensor(-1.0, device=args.device), torch.mean(critic_c(obs_n_o, action_cur_o)))

                opt_a.zero_grad()
                (2e-3*loss_pse+loss_a).backward()
                #loss_a.backward()
                nn.utils.clip_grad_norm_(actor_c.parameters(), args.max_grad_norm)
                opt_a.step()

            # save the model to the path_dir ---cnt by update number
            #if self.update_cnt > args.start_save_model and self.update_cnt % args.fre4save_model == 0:
            if self.update_cnt > args.start_save_model and self.update_cnt % args.fre4save_model_step == 0:
                time_now = time.strftime('%y%m_%d%H%M')
                print('=time:{} step:{}        save'.format(time_now, game_step))
                model_file_dir = os.path.join(args.save_dir, '{}_{}_{}'.format( \
                    args.scenario_name, time_now, game_step))
                if not os.path.exists(model_file_dir): # make the path
                    os.mkdir(model_file_dir)
                for agent_idx, (a_c, a_t, c_c, c_t) in \
                    enumerate(zip(self.actors_cur, self.actors_tar, self.critics_cur, self.critics_tar)):
                    torch.save(a_c, os.path.join(model_file_dir, 'a_c_{}.pt'.format(agent_idx)))
                    torch.save(a_t, os.path.join(model_file_dir, 'a_t_{}.pt'.format(agent_idx)))
                    torch.save(c_c, os.path.join(model_file_dir, 'c_c_{}.pt'.format(agent_idx)))
                    torch.save(c_t, os.path.join(model_file_dir, 'c_t_{}.pt'.format(agent_idx)))

            # update the tar par
            self.actors_tar = self.update_trainers(self.actors_cur, self.actors_tar, args.tao) 
            self.critics_tar = self.update_trainers(self.critics_cur, self.critics_tar, args.tao) 
