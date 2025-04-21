import numpy as np
import torch
import torch.nn as nn

class PPOMemory():
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
    
    def clear_memory(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []

    def add_memory(self,state,action,log_prob,value,reward):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)

    def sample(self):
        states = torch.FloatTensor(np.array(self.states))
        actions = torch.FloatTensor(np.array(self.actions))
        log_probs = torch.FloatTensor(np.array(self.log_probs))
        values = torch.FloatTensor(np.array(self.values))
        rewards = torch.FloatTensor(np.array(self.rewards))
        return states,actions,log_probs,values,rewards

class AC(nn.Module):
    def __init__(self,state_dim,action_dim):
        super(AC, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    def forward(self,x):
        action_prob = self.actor(x)
        dist = torch.distributions.Categorical(action_prob)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        value = self.critic(x)
        return action,log_prob,value
    
class PPO():
    def __init__(self,state_dim,action_dim,gamma,clip_ratio,K_epochs,batch_size):
        self.ac = AC(state_dim,action_dim)
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.K_epochs = K_epochs
        self.memory = PPOMemory()
    
    def add_memory(self,state,action,log_prob,value,reward):
        self.memory.add_memory(state,action,log_prob,value,reward)
        
    def clear_memory(self):
        self.memory.clear_memory()

    def choose_action(self,state):
        state = torch.FloatTensor(state)
        action,log_prob,value = self.ac(state)
        return action
        
    def update(self):
        states,actions,log_probs,values,rewards = self.memory.sample()
        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                returns[t] = rewards[t]
                advantages[t] = rewards[t] - values[t]
            else:
                returns[t] = rewards[t] + self.gamma * returns[t+1]
                advantages[t] = returns[t] - values[t]
        
        for _ in range(self.K_epochs):
            indices = np.arange(len(states))
            np.random.shuffle(indices)
            for start in range(len(states),self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_values = values[batch_indices]
                batch_log_probs = log_probs[batch_indices]
                batch_reward = rewards[batch_indices]
                batch_returns = returns[batch_indices]

                _,new_log_probs,new_value = self.ac(batch_states)