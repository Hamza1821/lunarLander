import gym
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

environment_name = 'LunarLander-v2'

# Create the environment
env = DummyVecEnv([lambda: gym.make(environment_name, render_mode='human')])

episodes = 10

# model = A2C('MlpPolicy', env, verbose = 1)

# model.learn(total_timesteps=100000)

# model.save("A2C_model")

for episode in range(1, episodes+1):
    state = env.reset()  # Reset environment
    state = state[0]  # Extract the state from the batch
    done = False
    score = 0 
    
    while True:
        env.render()  # Render the environment
        
        action = env.action_space.sample()  # Sample random action
        
        # Take a step and inspect the result
        n_state, reward, done, info = env.step([action])  # Pass the action as a list
        
        # Inspect the result
        
        # Unpack after inspecting the structure
        # Use [0] to extract values if necessary
        
        score += reward
          # Check if the episode is done
        
    print(f'Episode: {episode} Score: {score}')



env.close()



# import gym
# from stable_baselines3 import A2C
# from stable_baselines3.common.vec_env import DummyVecEnv
# from stable_baselines3.common.evaluation import evaluate_policy
# import time

# environment_name = 'LunarLander-v2'

# # Create the environment
# env = DummyVecEnv([lambda: gym.make(environment_name, render_mode='human')])

# episodes = 10
# total_timesteps = 14000
# save_frequency = 100 # Save the model every N timesteps
# reward_threshold = 100  # Example threshold to stop if reward goes below this (adjust as needed)

# # Initialize the model
# model = A2C('MlpPolicy', env, verbose=1)

# # Variables to track the performance
# previous_reward = 0  # Initialize previous reward for overfitting check
# reward_history = []  # To track the reward over episodes
# patience = 5  # Number of episodes to tolerate a decrease in reward
# stop_early = False

# for t in range(total_timesteps):
#     model.learn(total_timesteps=1, reset_num_timesteps=False)
    
#     if t % 100 == 0:  # Evaluate every 100 timesteps
#         mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=5)
#         print(f"Step {t}/{total_timesteps}, Reward: {mean_reward}")
        
#         reward_history.append(mean_reward)
        
#         # If we have enough history, check for a decrease in reward
#         if len(reward_history) > patience:
#             # Check the average reward in the last `patience` episodes
#             recent_rewards = reward_history[-patience:]
#             avg_recent_reward = sum(recent_rewards) / len(recent_rewards)
            
#             if avg_recent_reward < previous_reward:
#                 print("Warning: Reward has decreased. The model may be overfitting.")
            
#             # Stop training early if the average reward goes below the threshold
           
        
#         previous_reward = mean_reward
    
#     # Save model periodically
#     if t % save_frequency == 0:
#         model.save(f"A2C_model_{t}")
#         print(f"Model saved at step {t}")
#         choice=input("enter s to stop")
#         if choice=='s':
#             print("stopping the training early")
#             break;

# After training is done, you can save the final model


# Close the environment after training
env.close()

