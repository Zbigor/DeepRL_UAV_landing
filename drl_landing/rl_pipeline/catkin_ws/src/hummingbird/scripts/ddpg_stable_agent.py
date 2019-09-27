#!/usr/bin/env python


# -*- coding: utf-8 -*-

import gym
import numpy as np

#import stable_baselines.common.policies as pol
from stable_baselines.ddpg.policies import MlpPolicy, LnCnnPolicy
#from stable_baselines.common.policies import FeedForwardPolicy, register_policy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines import DDPG
import tensorflow as tf
import rospy
import rospkg
# import our training environment
from openai_ros.task_envs.hummingbird import hummingbird_land_basic

import stable_baselines.ddpg.policies as pol

import os

#
#print(pol.get_policy_registry())
#input("Press Enter to continue...")


os.environ['ROS_MASTER_URI'] = "http://localhost:1135" + str(0) + '/'
    
rospy.init_node('hummingbird_land_basic_test', anonymous=True, log_level=rospy.WARN)

    
    
env_name = rospy.get_param("/ddpg/env")
my_seed = rospy.get_param("/ddpg/seed")
my_gamma = rospy.get_param("/ddpg/gamma")
my_epochs = rospy.get_param("/ddpg/epochs")
exp_name = 'ddpg'
    


env = gym.make(env_name)
#env = DummyVecEnv([lambda: env])

# the noise objects for DDPG
n_actions = env.action_space.shape[-1]
param_noise = None

## consider changing this to having a gaussian noise
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

#action_noise = NormalActionNoise(mean=np.zeros(n_actions),sigma=float(0.2) * np.ones(n_actions))


print("low = "+ str(env.action_space.low))
print("high = " + str(env.action_space.high))

model = DDPG(policy = MlpPolicy,env = env,verbose=1,policy_kwargs=dict(layers = [400,300,200]), param_noise=param_noise, 
             action_noise=action_noise,tensorboard_log="/home/imartinovic/tboard_logs/",nb_rollout_steps= 100, nb_train_steps= 50, 
             nb_eval_steps = 100, batch_size=128, normalize_observations=False)

model.learn(total_timesteps=100000, log_interval = 5000, tb_log_name = "test_run_12_04_1")
model.save("/home/imartinovic/models/test_run_12_04_1")

#del model # remove to demonstrate saving and loading

#model = DDPG.load("/home/imartinovic/models/test_run_12_04_3",env=env, tensorboard_log= "~/tboard_logs/" )
rospy.logwarn("DONE TRAINING")
#rospy.logerr("TESTING THE LEARNED MODEL")
###input("Press Enter to continue...")
#obs = env.reset()
#steps_elapsed = 0
#while True:
#    action, _states = model.predict(obs)
#    obs, rewards, dones, info = env.step(action)
#    steps_elapsed+=1
#    if steps_elapsed>220:
#        steps_elapsed = 0
#        obs = env.reset()
#        