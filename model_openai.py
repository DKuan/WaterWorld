# Time: 2019-11-05
# Author: Zachary 
# Name: MADDPG_torch
import torch
import torch.nn as nn
import torch.nn.functional as F

class abstract_agent(nn.Module):
    def __init__(self):
        super(abstract_agent, self).__init__()
    
    def act(self, input):
        policy, value = self.forward(input) # flow the input through the nn
        return policy, value

class actor_agent(abstract_agent):
    def __init__(self, num_inputs, action_size, args):
        super(actor_agent, self).__init__()
        self.device = args.device
        self.linear_a1 = nn.Linear(num_inputs, 500)
        self.linear_a2 = nn.Linear(500, 128)
        self.linear_a = nn.Linear(128, action_size)
        self.reset_parameters()
        # Activation func init
        self.LReLU = nn.LeakyReLU(0.01)
        self.tanh= nn.Tanh()
        self.train()
    
    def reset_parameters(self):
        gain = nn.init.calculate_gain('leaky_relu')
        gain_tanh = nn.init.calculate_gain('tanh')
        nn.init.xavier_uniform_(self.linear_a1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_a2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_a.weight)
    
    def forward(self, input, model_original_out=False, device=None):
        """
        The forward func defines how the data flows through the graph(layers)
        """
        if device == None : device = self.device # set the device for restore
        x = self.LReLU(self.linear_a1(input))
        x = self.LReLU(self.linear_a2(x))
        model_out = self.linear_a(x)
        # u = torch.rand_like(model_out, device=device)
        # policy = F.softmax(model_out - torch.log(-torch.log(u)), dim=-1)
        policy = self.tanh(model_out)
        if model_original_out == True:   return model_out, policy # for model_out criterion
        return policy 

class critic_agent(abstract_agent):
    def __init__(self, obs_shape_n, action_shape_n, args):
        super(critic_agent, self).__init__()
        self.linear_c1 = nn.Linear(obs_shape_n, 1024)
        self.linear_c2 = nn.Linear(1024+action_shape_n, 512)
        self.linear_c3 = nn.Linear(512, 300)
        self.linear_c = nn.Linear(300, 1)
        self.reset_parameters()

        self.LReLU = nn.LeakyReLU(0.01)
        self.train()
    
    def reset_parameters(self):
        gain = nn.init.calculate_gain('leaky_relu')
        nn.init.xavier_uniform_(self.linear_c1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_c2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_c3.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_c.weight)

    def forward(self, obs_input, action_input, device=None):
        """
        input_g: input_global, input features of all agents
        """
        o_out = self.LReLU(self.linear_c1(obs_input))
        x = torch.cat([o_out, action_input], dim=1)
        x = self.LReLU(self.linear_c2(x))
        x = self.LReLU(self.linear_c3(x))
        value = self.linear_c(x)
        return value

class openai_critic(abstract_agent):
    def __init__(self, obs_shape_n, action_shape_n, args):
        super(openai_critic, self).__init__()
        self.LReLU = nn.LeakyReLU(0.01)
        self.linear_c1_o1 = nn.Linear(obs_shape_n, args.num_units_o_1)
        self.linear_c1_o2 = nn.Linear(args.num_units_o_1, args.num_units_o_2)
        self.linear_c1_a = nn.Linear(action_shape_n, args.num_units_a_1)
        self.linear_c2 = nn.Linear(args.num_units_o_2+args.num_units_a_1, args.num_units_2)
        self.linear_c = nn.Linear(args.num_units_2, 1)

        self.reset_parameters()
        self.train()
    
    def reset_parameters(self):
        gain = nn.init.calculate_gain('leaky_relu')
        # nn.init.xavier_uniform_(self.linear_c1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        # nn.init.xavier_uniform_(self.linear_c2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        # nn.init.xavier_uniform_(self.linear_c3.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_c1_o1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_c1_o2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_c1_a.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_c2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_c.weight, gain=nn.init.calculate_gain('leaky_relu'))

    def forward(self, obs_input, action_input):
        """
        input_g: input_global, input features of all agents
        """
        o_out = self.LReLU(self.linear_c1_o1(obs_input))
        o_out = self.LReLU(self.linear_c1_o2(o_out))
        a_out = self.LReLU(self.linear_c1_a(action_input))
        x_cat = self.LReLU(self.linear_c2(torch.cat([o_out, a_out], dim=1)))
        value = self.linear_c(x_cat)
        return value

class openai_actor(abstract_agent):
    def __init__(self, num_inputs, action_size, args):
        super(openai_actor, self).__init__()
        self.device = args.device
        self.tanh= nn.Tanh()
        self.LReLU = nn.LeakyReLU(0.01)
        self.linear_a1 = nn.Linear(num_inputs, args.num_units_1)
        self.linear_a2 = nn.Linear(args.num_units_1, args.num_units_2)
        self.linear_a = nn.Linear(args.num_units_2, action_size)

        self.reset_parameters()
        self.train()
    
    def reset_parameters(self):
        gain = nn.init.calculate_gain('leaky_relu')
        gain_tanh = nn.init.calculate_gain('tanh')
        nn.init.xavier_uniform_(self.linear_a1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_a2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        #nn.init.xavier_uniform_(self.linear_a3.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_a.weight, gain=nn.init.calculate_gain('leaky_relu'))
    
    def forward(self, input, model_original_out=False):
        """
        The forward func defines how the data flows through the graph(layers)
        flag: 0 sigle input 1 batch input
        """
        x = self.LReLU(self.linear_a1(input))
        x = self.LReLU(self.linear_a2(x))
        #x = self.LReLU(self.linear_a3(x))
        model_out = self.linear_a(x)
        u = torch.rand_like(model_out, device=self.device)
        policy = F.softmax(model_out - torch.log(-torch.log(u)), dim=-1)
        if model_original_out == True:   return model_out, policy # for model_out criterion
        return policy
