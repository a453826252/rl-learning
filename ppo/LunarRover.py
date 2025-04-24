import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter


logger = SummaryWriter("logs")
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

class AC:
    def __init__(self,state_dim,action_dim):
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
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
    def get_action_and_value(self,x):
        action_prob = self.actor(x)
        dist = torch.distributions.Categorical(action_prob)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        value = self.critic(x)
        return action,log_prob,value.squeeze(-1)
    def evaluate(self,state,action):
        action_prob = self.actor(state)
        dist = torch.distributions.Categorical(action_prob)
        log_prob = dist.log_prob(action)
        value = self.critic(state)
        return log_prob,value.squeeze(-1)
    
class PPO():
    def __init__(self,state_dim,action_dim,gamma,clip_ratio,K_epochs,batch_size):
        self.ac = AC(state_dim,action_dim)
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.K_epochs = K_epochs
        self.batch_size = batch_size
        self.ac_optimizer = torch.optim.Adam([
                        {'params': self.ac.actor.parameters(), 'lr': 0.001},
                        {'params': self.ac.critic.parameters(), 'lr': 0.0003}
                    ])
        self.memory = PPOMemory()
        self.mse_loss = nn.MSELoss()
    
    def add_memory(self,state,action,log_prob,value,reward):
        self.memory.add_memory(state,action,log_prob,value,reward)
        
    def clear_memory(self):
        self.memory.clear_memory()

    def choose_action(self,state):
        state = torch.FloatTensor(state).unsqueeze(0)  # 添加batch维度，因为模型的输入是(batch_size, state_dim)
        with torch.no_grad():  # 不计算梯度，因为我们只需要得到动作，不需要反向传播
            action,log_prob,value = self.ac.get_action_and_value(state)
        return action.item(),log_prob.item(),value.item()

    def save(self):
        torch.save(self.ac.actor.state_dict(),"ac_actor.pth")
        torch.save(self.ac.critic.state_dict(),"ac_critic.pth")

    def load(self):
        self.ac.actor.load_state_dict(torch.load("ac_actor.pth"))
        self.ac.critic.load_state_dict(torch.load("ac_critic.pth"))

    def update(self):
        states,actions,log_probs,values,rewards = self.memory.sample()
        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                returns[t] = rewards[t]
            else:
                returns[t] = rewards[t] + self.gamma * returns[t+1]
            advantages[t] = returns[t] - values[t]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        total_loss = []
        for _ in range(self.K_epochs):
            indices = np.arange(len(states))
            np.random.shuffle(indices)
            for start in range(0,len(states),self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_log_probs = log_probs[batch_indices]
                batch_returns = returns[batch_indices]

                new_log_probs,new_value = self.ac.evaluate(batch_states,batch_actions)
                radio = (new_log_probs - batch_log_probs).exp()
                clip_radio = torch.clamp(radio,1-self.clip_ratio,1+self.clip_ratio)
                loss = -torch.min(radio*batch_advantages,clip_radio*batch_advantages).mean() + 0.5*self.mse_loss(new_value,batch_returns)
                total_loss.append(loss.item())
                self.ac_optimizer.zero_grad()
                loss.backward()
                self.ac_optimizer.step()
        return np.mean(total_loss)

def train():
    env = gym.make('LunarLander-v3')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    ppo = PPO(state_dim, action_dim,0.99,0.2,80,64)
    for episode in range(200):
        state,_ = env.reset()
        done = False
        truncated = False
        total_reward = 0
        while not done and not truncated:
            action,log_prob,value = ppo.choose_action(state)
            next_state,reward,done,truncated,_ = env.step(action)
            ppo.add_memory(state,action,log_prob,value,reward)
            state = next_state
            total_reward += reward
        ave_loss = ppo.update()
        ppo.clear_memory()
        logger.add_scalar("reward",total_reward,episode)
        logger.add_scalar("loss",ave_loss,episode)
        if episode % 10 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward},loss:{ave_loss}")
        
        if episode % 100 == 0:
            ppo.save()  # 保存模型参数
    env.close()
    logger.close()

def test():
    env = gym.make('LunarLander-v3',render_mode="human")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    ppo = PPO(state_dim, action_dim,0.99,0.2,10,64)
    ppo.load()  # 加载模型参数
    state,_ = env.reset()
    done = False
    while not done:
        action,log_prob,value = ppo.choose_action(state)
        next_state,reward,done,_,_ = env.step(action)
        state = next_state
if __name__ == "__main__":
    # train()
    test()