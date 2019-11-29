import os
import sys

import torch
import multiagent.scenarios as scenarios
from multiagent.environment import MultiAgentEnv
from madrl_environments.pursuit import MAWaterWorld_mod

from model import actor_agent, critic_agent
from arguments import parse_args

def make_env(scenario_name, arglist, benchmark=False):
    """ 
    create the environment from script 
    """
    world = MAWaterWorld_mod(n_pursuers=2, n_evaders=50,
                         n_poison=50, obstacle_radius=0.04,
                         food_reward=10,
                         poison_reward=-1,
                         encounter_reward=0.01,
                         n_coop=2,
                         sensor_range=0.2, obstacle_loc=None, )
    world.seed(123)
    return worlv

def get_trainers(env, arglist):
    trainers_cur = []
    trainers_tar = []
    optimizers = []
    input_size = [8, 10, 10] # the obs size
    input_size_global = [23, 25, 25] # cal by README

    """ load the model """
    actors_tar = [torch.load(arglist.old_model_name+'a_c_{}.pt'.format(agent_idx), map_location=arglist.device) \
        for agent_idx in range(env.n)]

    return actors_tar

def enjoy(arglist):
    """ 
    This func is used for testing the model
    """
    episode_step = 0
    """ init the env """
    env = make_env(arglist.scenario_name, arglist, arglist.benchmark)

    obs_shape_n = [env.agents[i].observation_space.shape[0] for i in range(env.agents.__len__())]
    #action_shape_n = [env.agents[i].action_space.n-1 for i in range(env.agents.__len__())] # no need for stop bit
    action_shape_n = [2, 2]
    #num_adversaries = min(env.agents.__len__(), arglist.num_adversaries)
    actors_cur = get_trainers(env, arglist)

    """ interact with the env """
    obs_n = env.reset()
    while(1):
        # update the episode step number
        episode_step += 1
        # get action
        # action_0 = [1, 0, 0, 0, 0]
        # action_1 = [1, 0.1, 0, 0, 0]
        # action_2 = [10, -0.1, 0, -0.1, 0]
        try:
            action_n = []
            # action_n = [agent.actor(torch.from_numpy(obs).to(arglist.device, torch.float)).numpy() \
            # for agent, obs in zip(trainers_cur, obs_n)]
            for actor, obs in zip(actors_tar, obs_n):
                action = torch.clamp(actor(torch.from_numpy(obs).to(arglist.device, torch.float)), -1, 1)
                action_n.append(action)
        except:
            print(obs_n)

        # interact with env
        new_obs_n, rew_n, done_n, info_n = env.step(action_n)
        done = all(done_n)
        terminal = (episode_step >= arglist.max_episode_len)

        # reset the env
        if done or terminal: 
            episode_step = 0
            obs_n = env.reset()

        new_obs_n, rew_n, done_n, info_n = env.step(action_n)
        # env._get_obs(env.agents[0])
        print(rew_n)
        #env.render()

if __name__ == '__main__':
    arglist = parse_args()
    enjoy(arglist)