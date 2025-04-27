import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

class RewardLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardLoggerCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.current_reward = 0
        self.overCount = 0

    def _on_step(self) -> bool:
        reward = self.locals['rewards'][0]
        self.current_reward += reward
        if self.locals['dones'][0]:
            self.over = self.current_reward > 200
            if self.over:
                self.overCount += 1
            else:
                self.overCount = 0    
            self.episode_rewards.append(self.current_reward)
            print(f"Episode {len(self.episode_rewards)}: Total Reward = {self.current_reward}")
            self.current_reward = 0
        return self.overCount < 5


env = gym.make('LunarLander-v3',render_mode="rgb_array")
model = PPO('MlpPolicy', env,n_epochs=10, verbose=1)
model.learn(total_timesteps=1000000000,callback=RewardLoggerCallback())
env.close()
# env = gym.make('LunarLander-v3',render_mode="human")
# obs,_ = env.reset()
# dones = False
# while not dones:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info,_ = env.step(action)
    
