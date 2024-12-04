import gym
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

environment_name = 'LunarLander-v2'

# Create the environment
env = DummyVecEnv([lambda: gym.make(environment_name, render_mode='human')])

model = A2C.load("A2C_model", env=env)

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()