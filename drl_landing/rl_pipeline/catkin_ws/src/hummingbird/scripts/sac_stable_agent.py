#!/usr/bin/env python


# -*- coding: utf-8 -*-

import gym
import numpy as np
from gym.envs.registration import register
import multiprocessing as mp

#import stable_baselines.common.policies as pol
#from stable_baselines.common.policies import FeedForwardPolicy, register_policy
from stable_baselines.common.vec_env import DummyVecEnv

import tensorflow as tf
import rospy
import rospkg
# import our training environment
import openai_ros.task_envs.hummingbird.arbitrary_spot_landing

#import stable_baselines.ddpg.policies as pol

from stable_baselines.sac.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import SAC
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
import os



os.environ['ROS_MASTER_URI'] = "http://localhost:1135" + str(3) + '/'

env1_name = "ArbitrarySpotLanding-v0"
#openai_ros.openai_ros.src.openai_ros.task_envs
timestep_limit_per_episode = 100 # Can be any Value
register(
        id='ArbitrarySpotLanding-v0',
        entry_point='openai_ros:task_envs.hummingbird.arbitrary_spot_landing.HummingbirdLandBasi',
        timestep_limit=timestep_limit_per_episode
    )


print("registration done")



os.environ['ROS_MASTER_URI'] = "http://localhost:1135" + str(3) + '/'
    
#rospy.init_node('hummingbird_land_basic_test', anonymous=True, log_level=rospy.WARN)



best_mean_reward, n_steps = -np.inf, 0

def callback(_locals, _globals):
  """
  Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
  :param _locals: (dict)
  :param _globals: (dict)
  """
  global n_steps, best_mean_reward
  # Print stats every 10 calls

  if (n_steps + 1) % 100 == 0:
      # Evaluate policy performance
      _locals['self'].save('/home/imartinovic/snapshots/sac/checkpoint_model.pkl')

      x, y = ts2xy(load_results(log_dir), 'timesteps')
      if len(x) > 0:
          mean_reward = np.mean(y[-100:])
          print(x[-1], 'timesteps')
          print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(best_mean_reward, mean_reward))

          # New best model, you could save the agent here
          if mean_reward > best_mean_reward:
              best_mean_reward = mean_reward
              # Example for saving best model
              print("Saving new best model")
              _locals['self'].save(log_dir + 'best_model.pkl')
  n_steps += 1
  return True


# Create log dir
log_dir = "/home/imartinovic/snapshots/sac/"
os.makedirs(log_dir, exist_ok=True)




env = gym.make('ArbitrarySpotLanding-v0',env_id = 3)
env = Monitor(env, log_dir, allow_early_resets=True)
env = DummyVecEnv([lambda: env])

# policy_kwargs=dict(layers = [400,300,200])
model = SAC(MlpPolicy, env, verbose=1,tensorboard_log="/home/imartinovic/tboard_logs/sac_22_04/")
model.learn(total_timesteps=700000,tb_log_name = "sac_22_04_1",callback = callback)
model.save("/home/imartinovic/models/sac_22_04_1")
#model = PPO1.load("/home/imartinovic/models/best_model",env=env, tensorboard_log= "~/tboard_logs/")

rospy.logwarn("DONE TRAINING")
#rospy.logerr("TESTING THE LEARNED MODEL")
###input("Press Enter to continue...")
#obs = env.reset()
#steps_elapsed = 0
#while True:
#    action, _states = model.predict(observation=obs, deterministic = True)
#    obs, rewards, dones, info = env.step(action)
#    steps_elapsed+=1
#    if steps_elapsed>220:
#        steps_elapsed = 0
#        obs = env.reset()

