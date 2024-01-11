import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class ActorNN(nn.Module):
    def __init__(self, input_dims, n_actions, fc1_dims, fc2_dims, lr):
        # Input state space, output policy (dim = dim of action space)
        super(ActorNN, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims #fc stands for fully connected layer, also known as "linear layer"
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims) # aristerisk means any number of dimensions
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.fc1_dims,self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)
        self.optimizer = optim.Adam(self.parameters(),lr=lr)
        self.loss = 0
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

        
class CriticNN(nn.Module):
    # Input state space, output value of the current state (dim=1)
    def __init__(self, input_dims, fc1_dims, fc2_dims, lr):
        # input_dims is the dimension of the state space
        super(CriticNN, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims #fc stands for fully connected layer, also known as "linear layer"
        self.fc2_dims = fc2_dims

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims) # aristerisk means any number of dimensions
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.fc1_dims,self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, 1)
        self.optimizer = optim.Adam(self.parameters(),lr=lr)
        self.loss = 0
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

class Agent():
    def __init__(self, input_dims, n_actions, fc1_dims, fc2_dims, gamma, lr):
        self.actor = ActorNN(input_dims, n_actions, fc1_dims, fc2_dims, lr)
        self.critic = CriticNN(input_dims, fc1_dims, fc2_dims, lr)

        actions = torch.empty(n_actions)
        self.pi = nn.init.uniform_(actions,a=0.0,b=1.0)
        self.pi_prob = torch.distributions.Categorical(probs=self.pi)

        self.total_reward = 0
        self.gamma = gamma

    
    def take_action(self):
        action = self.pi_prob.sample()
        return action

    def train(self, action, state, state_, reward, done):

        state_value = self.critic.forward(state)
        state_value_ = self.critic.forward(state_)
        delta = reward + self.gamma*state_value_*(1-int(done)) - state_value   #state value at terminal state is 0

        # Take actions
        self.pi = self.actor.forward(state)
        self.pi_prob = torch.distributions.Categorical(probs=self.pi)
        
        pi_log_prob = self.pi_prob.log_prob(action)

        self.critic.loss = torch.pow(delta, 2)

        self.critic.optimizer.zero_grad()
        self.critic.loss.backward()
        self.critic.optimizer.step()

        self.actor.loss = -pi_log_prob*delta

        self.actor.optimizer.zero_grad()
        self.actor.loss.backward()
        self.actor.optimizer.step()



    # def take_action(self, state):
        

    #     return action, pi_log_prob
