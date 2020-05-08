# Time: 2019-11-05
# Author: Zachary 
# Name: MADDPG_torch
import torch
import torch.nn as nn
import torch.nn.functional as F

class actor_agent(nn.Module):
    def __init__(self, num_inputs, action_size, args):
        super(actor_agent, self).__init__()
        self.linear_a1 = nn.Linear(num_inputs, args.num_units_a[0])
        self.linear_a2 = nn.Linear(args.num_units_a[0], args.num_units_a[1])
        self.linear_a3 = nn.Linear(args.num_units_a[1], action_size)
        self.reset_parameters()

        self.LRuLU = nn.LeakyReLU(0.01)
        self.tanh= nn.Tanh()
        self.train()
    
    def reset_parameters(self):
        #gain = nn.init.calculate_gain('leaky_relu')
        #gain_tanh = nn.init.calculate_gain('tanh')
        nn.init.xavier_uniform_(self.linear_a1.weight)
        nn.init.xavier_uniform_(self.linear_a2.weight)
        nn.init.xavier_uniform_(self.linear_a3.weight)
    
    def forward(self, input, training=True, model_original_out=False):
        """
        The forward func defines how the data flows through the graph(layers)
        """
        x = self.linear_a1(input)
        x = self.LRuLU(x)
        x = self.LRuLU(self.linear_a2(x))
        model_out = self.linear_a3(x)
        policy = self.tanh(model_out)
        if model_original_out == True: return model_out, policy
        return policy 

class critic_agent(nn.Module):
    def __init__(self, obs_shape_n, action_shape_n, args):
        super(critic_agent, self).__init__()
        self.linear_c1 = nn.Linear(obs_shape_n, args.num_units_c[0])
        self.linear_c2 = nn.Linear(args.num_units_c[0]+action_shape_n, args.num_units_c[1])
        self.linear_c3 = nn.Linear(args.num_units_c[1], args.num_units_c[2])
        self.linear_c4 = nn.Linear(args.num_units_c[2], 1)
        self.bn1 = nn.BatchNorm1d(args.num_units_c[1], affine=True, track_running_stats=True)
        self.bn2 = nn.BatchNorm1d(args.num_units_c[2], affine=True, track_running_stats=True)
        self.reset_parameters()

        self.LReLU = nn.LeakyReLU(0.01)
        self.train()
    
    def reset_parameters(self):
        #gain = nn.init.calculate_gain('leaky_relu')
        #gain_tanh = nn.init.calculate_gain('tanh')
        nn.init.xavier_uniform_(self.linear_c1.weight)
        nn.init.xavier_uniform_(self.linear_c2.weight)
        nn.init.xavier_uniform_(self.linear_c3.weight)
        nn.init.xavier_uniform_(self.linear_c4.weight)

    def forward(self, obs_n, action_n):
        """
        input_g: input_global, input features of all agents
        """
        x = self.LReLU(self.linear_c1(obs_n))
        x = self.linear_c2(torch.cat([x, action_n], dim=1))
        x = self.bn1(x)
        x = self.LReLU(x)
        x = self.linear_c3(x)
        x = self.bn2(x)
        x = self.LReLU(x)
        value = self.linear_c4(x)
        return value
