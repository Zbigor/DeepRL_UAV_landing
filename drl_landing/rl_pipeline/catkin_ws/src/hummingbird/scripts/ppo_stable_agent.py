#!/usr/bin/env python

# -*- coding: utf-8 -*-
import os
import gym
import rospy
import multiprocessing as mp
import numpy as np
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.policies import MlpLnLstmPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import PPO2
#from stable_baselines import PPO1
#from stable_baselines import TRPO
#from stable_baselines import SAC

from openai_ros.robot_envs import hummingbird_env
from gym.envs.registration import register

import openai_ros.task_envs.hummingbird.hummingbird_waypoints_landing

from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy

from stable_baselines.sac.policies import MlpPolicy as sac_mlp


env1_name = "SpecificSpotLanding-v0"
#openai_ros.openai_ros.src.openai_ros.task_envs
timestep_limit_per_episode = 180 # Can be any Value
register(
        id='SpecificSpotLanding-v0',
        entry_point='openai_ros:task_envs.hummingbird.hummingbird_waypoints_landing.HummingbirdLandBasic',
        timestep_limit=timestep_limit_per_episode
    )




print("registration done")


def my_gym_make(e_name,e_id, log):
#    temporary +1, to let node0 run the demo
    e_id+=0
    en = gym.make(e_name,env_id = e_id)
    en = Monitor(en, log, allow_early_resets=True)

    return en

log_dir = "/home/imartinovic/snapshots/ppo2/"
os.makedirs(log_dir, exist_ok=True)
best_mean_reward, n_steps = -np.inf, 0

def callback(_locals, _globals):
      """
      Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
      :param _locals: (dict)
      :param _globals: (dict)
      """
      log_dir = "/home/imartinovic/snapshots/ppo2/"

      global n_steps, best_mean_reward
      # Print stats every 1000 steps
#      print("callback called")
      if (n_steps + 1) % 50 == 0:
          # saving checkpoint
          
          _locals['self'].save('/home/imartinovic/snapshots/ppo2/checkpoint_modelppo12.pkl')
          if (_locals['self'].episode_reward[-1]>best_mean_reward):
              best_mean_reward = _locals['self'].episode_reward[-1]
              _locals['self'].save(log_dir + 'best_model1ppo2.pkl')
          # Evaluate policy performance
          print("EVAL PERFORMANCE")
          rospy.logwarn("EVAL PERFORMANCE")
          
          try:
              x, y = ts2xy(load_results(log_dir), 'timesteps')
              if len(x) > 0:
                  
                  mean_reward = np.sum(y[-100:])
                  print(x[-1], 'timesteps')
                  print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(best_mean_reward, mean_reward))
        
                  # New best model, you could save the agent here
                  if mean_reward > best_mean_reward:
                      best_mean_reward = mean_reward
                      # Example for saving best model
                      print("Saving new best model")
                      _locals['self'].save(log_dir + 'best_model1ppo2.pkl')
          except:
              rospy.logwarn("BAD INPUT")
            
              
      n_steps += 1
      return True



# multiprocess environment
if __name__ == '__main__':
    
    mp.freeze_support()    
    n_workers = 3
    env_name = "SpecificSpotLanding-v0"



    log_dir = "/home/imartinovic/snapshots/ppo2/"
    os.makedirs(log_dir, exist_ok=True)
    best_mean_reward, n_steps = -np.inf, 0

#    env = SubprocVecEnv(env_fns = [
#                                    lambda: my_gym_make(e_name = env_name,e_id = 0),
#                                    lambda: my_gym_make(e_name = env_name,e_id = 1),
#                                    lambda: my_gym_make(e_name = env_name,e_id = 2)
##                                    lambda:my_gym_make(e_name = env1_name,e_id = 3)
#                                   ],start_method='forkserver')
    
    env = SubprocVecEnv(env_fns = [lambda worker_id=worker_id: my_gym_make(e_name = env_name,e_id = worker_id,log=log_dir) for worker_id in range(n_workers)],
                                   start_method='forkserver')


    #TRAINING

#    print("DONE WITH ENV CREATION")
##    #rospy.logwarn('DONE with env creation')
######    p_kwargs=dict(net_arch = [400,300,200])
    model = PPO2(MlpLnLstmPolicy, env, verbose=1,tensorboard_log="/home/imartinovic/tboard_logs/ppo2_27_09/",full_tensorboard_log = False)
    print("STARTING TO LEARN")
    model.learn(total_timesteps=1500000,tb_log_name = "ppo2_27_09",callback=callback)
    model.save("/home/imartinovic/models/ppo2_27_09_ppo2")
###    
    print("DONE TRAINING")
    rospy.signal_shutdown("done training")
    
    
    # TESTING
    
    
#    model = PPO2.load("/home/imartinovic/tboard_logs/ppo2_20_08/ppo2_20_08_1/checkpoint_model",env=env, tensorboard_log= "~/tboard_logs/")
###    
#    rospy.logwarn("DONE TRAINING")
#    rospy.logerr("TESTING THE LEARNED MODEL")
##    input("Press Enter to continue...")
#    obs = env.reset()
#    rospy.logwarn("DONE RESET")
#    steps_elapsed = 0
#    while True:
#        action, _states = model.predict(observation=obs, deterministic = True)
#        obs, rewards, dones, info = env.step(action)
#######        steps_elapsed+=1
######        if steps_elapsed>160:
######            steps_elapsed = 0
######             obs = env.reset()