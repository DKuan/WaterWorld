# Time: 2019-11-05
# Author: Zachary 
# Name: MADDPG_torch
# File func: main func
import os

import time
import torch
import pickle
import argparse
import numpy as np

from arguments import parse_args
from madrl_environments.pursuit import MAWaterWorld_mod
from maddpg_agent import MADDPG_Agent

def make_env(scenario_name, args, benchmark=False):
    """ 
    create the environment from script 
    """
    n_evaders = 10
    n_poison = 3
    radius = 0.05
    n_coop = 2
    n_sensors = 60
    world = MAWaterWorld_mod(n_pursuers=2, n_evaders=n_evaders, radius=radius,
                n_poison=n_poison, n_sensors=n_sensors, obstacle_radius=0.0,
                food_reward=10, poison_reward=-1,
                encounter_reward=0.01, n_coop=n_coop,
                sensor_range=1.5, obstacle_loc=None, )
    print('n_evaders:{} n_poison:{} radius:{} n_coop:{}'.format(n_evaders, n_poison, radius, n_coop))
    world.seed(996)
    np.random.seed(996)
    return world 

def train(args):
    """
    init the env, agent and train the agents
    """
    """step1: create the environment and get the info """
    env = make_env(args.scenario_name, args, args.benchmark)
    num_agents = env.agents.__len__()
    obs_shape_n = [env.agents[i].observation_space.shape[0] for i in range(num_agents)]
    action_shape_n = [env.agents[i].action_space.shape[0] for i in range(num_agents)] # no need for stop bit

    print('=============================')
    print('=1 Env {} is right ...'.format(args.scenario_name))
    print('=============================')

    """step2: create agents"""
    m_agent = MADDPG_Agent(num_agents, obs_shape_n, action_shape_n, args) # obs_size not check
    old_data = m_agent.init_trainers(args)
    
    print('=2 The {} agents are inited ...'.format(num_agents))
    print('=============================')

    """step3: init the pars """
    game_step = 0 if args.restore == False else old_data['game_step']
    episode_cnt = 0 
    reward_get_cnt = 0
    episode_gone_old = 0 if args.restore == False else old_data['episode_gone_old']
    t_start = time.time()
    rew_n_old = [0.0 for _ in range(num_agents)] # set the init reward
    agent_info = [[[]]] # placeholder for benchmarking info
    episode_rewards = [] # sum of rewards for all agents
    agent_rewards = [[] for _ in range(num_agents)] # individual agent reward
    
    print('=3 starting iterations ...')
    print('=============================')

    for episode_gone in range(episode_gone_old, args.max_episode_num):
        obs_n = env.reset()
        episode_rewards.append(0.0)
        for a_r in agent_rewards:
            a_r.append(0.0)
        # cal the reward print the debug data
        if game_step > 1:   
            mean_agents_r = [round(np.mean(agent_rewards[idx][-20:-1]), 2) for idx in range(num_agents)]
            mean_ep_r = round(np.mean(episode_rewards[-20:-1]), 2)
            print(" "*38 + 'episode reward:{} agents mean reward:{} reward_get_cnt:{} var:{}'.format( \
                mean_ep_r, mean_agents_r, reward_get_cnt, round(m_agent.var, 2)), end='\r')
        print('=Training: steps:{} episode:{}'.format(game_step, episode_gone), end='\r')

        for epi_in_cnt in range(args.per_episode_max_len):
            # get action
            actions = m_agent.select_actions(obs_n, args) 

            # interact with env
            new_obs_n, rew_n, done, info_n = env.step(actions)

            # save the experience
            done_n = [done for _ in range(num_agents)]
            m_agent.memory.add(obs_n, np.concatenate(actions), rew_n , new_obs_n, done_n)
            reward_get_cnt += info_n['reward_get']
            episode_rewards[-1] += np.sum(rew_n)
            for i, rew in enumerate(rew_n): agent_rewards[i][-1] += rew

            # train our agents 
            m_agent.agents_train(game_step, episode_gone, args)

            # update the obs_n
            game_step += 1
            obs_n = new_obs_n

            # check the end flag
            if done or epi_in_cnt >= args.per_episode_max_len-1:
                break

if __name__ == '__main__':
    args = parse_args()
    train(args)
