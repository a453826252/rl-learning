import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make('LunarLander-v3',render_mode="rgb_array")
model = PPO('MlpPolicy', env,n_epochs=100, verbose=1)
model.learn(total_timesteps=10000)
env.close()
env = gym.make('LunarLander-v3',render_mode="human")
obs,_ = env.reset()
dones = False
while not dones:
    action, _states = model.predict(obs)
    obs, rewards, dones, info,_ = env.step(action)
