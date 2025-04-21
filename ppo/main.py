import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
torch.autograd.set_detect_anomaly(True)

# 定义策略网络（Actor）
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.mu_head = nn.Linear(64, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        mu = torch.softmax(self.mu_head(x),dim=-1)
  
        dist = torch.distributions.Categorical(mu)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob


# 定义价值网络（Critic）
class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.v_head = nn.Linear(64, 1)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.v_head(x)


# PPO算法实现
class PPO:
    def __init__(self, state_dim, action_dim, gamma=0.99, clip_ratio=0.2,
                 actor_lr=3e-4, critic_lr=1e-3, train_epochs=10, batch_size=64):
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.train_epochs = train_epochs
        self.batch_size = batch_size

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action, log_prob = self.actor(state)
        return action.item(), log_prob.item()

    def update(self, states, actions, log_probs, rewards, dones):
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        log_probs = torch.FloatTensor(log_probs)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)

        values = self.critic(states)
        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)

        running_return = 0
        running_advantage = 0
        for t in reversed(range(len(rewards))):
            if dones[t]:
                running_return = 0
                running_advantage = 0
            running_return = rewards[t] + self.gamma * running_return
            returns[t] = running_return
            running_advantage = rewards[t] + self.gamma * values[t + 1] - values[t] if t < len(rewards) - 1 else rewards[t] - values[t]
            advantages[t] = running_advantage

        # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.train_epochs):
            indices = np.arange(len(states))
            # np.random.shuffle(indices)
            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]

                state_batch = states[batch_indices]
                action_batch = actions[batch_indices]
                old_log_prob_batch = log_probs[batch_indices]
                advantage_batch = advantages[batch_indices]
                return_batch = returns[batch_indices]

                new_action, new_log_prob = self.actor(state_batch)
                ratio = (new_log_prob - old_log_prob_batch).exp()
                surr1 = ratio * advantage_batch
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantage_batch
                actor_loss = -torch.min(surr1, surr2).mean()

                value = self.critic(state_batch)
                critic_loss = nn.MSELoss()(value, return_batch)

                self.actor_optimizer.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()
                break


# 主训练循环
def train():
    env = gym.make('LunarLander-v3')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    ppo = PPO(state_dim, action_dim)

    num_episodes = 1000
    max_steps = 1000
    for episode in range(num_episodes):
        state, _ = env.reset()
        states, actions, log_probs, rewards, dones = [], [], [], [], []
        for step in range(max_steps):
            action, log_prob = ppo.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            dones.append(done)
            state = next_state
            if done:
                break
        ppo.update(states, actions, log_probs, rewards, dones)
        if episode % 10 == 0:
            total_reward = sum(rewards)
            print(f"Episode {episode}: Total Reward = {total_reward}")
    env.close()


if __name__ == "__main__":
    train()
