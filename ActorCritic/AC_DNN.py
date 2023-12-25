import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class ActorNN(nn.Module):
    def __init__(self, input_dims, n_actions, fc1_dims, fc2_dims, lr):
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
        self.loss = nn.MSELoss() # as in the DQN paper
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

        
class CriticNN(nn.Module):
    def __init__(self, input_dims, n_actions, fc1_dims, fc2_dims, lr):
        super(CriticNN, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims #fc stands for fully connected layer, also known as "linear layer"
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims) # aristerisk means any number of dimensions
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.fc1_dims,self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)
        self.optimizer = optim.Adam(self.parameters(),lr=lr)
        self.loss = nn.MSELoss() # as in the DQN paper
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

class Agent():
    def __init__(self, gamma):
        self.actor = ActorNN()
        self.critic = CriticNN()

        self.gamma = gamma
    

    def train(self, state, state_, reward, pi_log_prob, done):
        state_value = self.critic.forward(state)
        state_value_ = self.critic.forward(state_)
        delta = reward + self.gamma*state_value_*(1-int(done)) - state_value   #state value at terminal state is 0

        actor_loss = -pi_log_prob*delta
        critc_loss = np.power(delta, 2)

        
        

    def take_action(self, state):
        pi = self.actor.forward(state)
        pi_prob =  torch.distributions.Categorical(probs=pi)
        action = pi_prob.sample()
        pi_log_prob = pi_prob.log_prob(action)

        return action, pi_log_prob
