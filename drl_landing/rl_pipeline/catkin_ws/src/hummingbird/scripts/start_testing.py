#!/usr/bin/env python
#from spinup.algos.ddpg import ddpg



import gym
import numpy
import time

from gym import wrappers
# ROS packages required
import rospy
import rospkg
# import our training environment
from openai_ros.task_envs.hummingbird import hummingbird_test1


if __name__ == '__main__':
    
#    my_ddpg = ddpg()
    rospy.init_node('hummingbird_test1_test', anonymous=True, log_level=rospy.WARN)

    # Create the Gym environment
    env = gym.make('HummingbirdTest1-v0')
    rospy.logwarn("Gym environment done")

    # Set the logging system
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('hummingbird')
    outdir = pkg_path + '/training_results'
    rospy.logwarn("Monitor Wrapper started")

    env = wrappers.Monitor(env, outdir, force=True)
    rospy.logwarn("Monitor Wrapper started")

    last_time_steps = numpy.ndarray(0)

    # Loads parameters from the ROS param server
    # Parameters are stored in a yaml file inside the config directory
    # They are loaded at runtime by the launch file
    Alpha = rospy.get_param("/drone/alpha")
    Epsilon = rospy.get_param("/drone/epsilon")
    Gamma = rospy.get_param("/drone/gamma")
    epsilon_discount = rospy.get_param("/drone/epsilon_discount")
    nepisodes = rospy.get_param("/drone/nepisodes")
    nsteps = rospy.get_param("/drone/nsteps")

    # Initialises the algorithm that we are going to use for learning
#    qlearn = qlearn.QLearn(actions=range(env.action_space.n),
#                           alpha=Alpha, gamma=Gamma, epsilon=Epsilon)
#    initial_epsilon = qlearn.epsilon
    initial_epsilon = 10
    start_time = time.time()
    highest_reward = 0
    rospy.logwarn("Main loop starting")

    # Starts the main training loop: the one about the episodes to do
    for x in range(nepisodes):
        rospy.logwarn("############### START EPISODE=>" + str(x))
                      
        done = False
#        if qlearn.epsilon > 0.05:
#            qlearn.epsilon *= epsilon_discount

        # Initialize the environment and get first state of the robot
        rospy.logwarn("RESET SIM STARTED")
        # CUMULATED REWARD FOR THE PREVIOUS EPISODE GETS PUBLISHED HERE
        observation = env.reset()
        rospy.logwarn("RESET SIM DONE")
        state = ''.join(map(str, observation))

        # Show on screen the actual situation of the robot
        # env.render()
        # for each episode, we test the robot for nsteps
        # dummy action for testing
        action = 1
        cumulated_reward = 0
        
        rospy.logwarn('Starting iterations')
       
        episode_start = time.time()
        for i in range(nsteps):
            rospy.logwarn("############### Start Step=>" + str(i))
            # Pick an action based on the current state
            if i % 15 == 0:
                action = -1 * action
                
#            action = qlearn.chooseAction(state)
            
            rospy.logwarn("Next action is:%d", action)
            # Execute the action in the environment and get feedback
            
            # IMPORTANT, NEEDS TO BE DEFINED EVEN FOR THE TEST
            observation, reward, done, info = env.step(action)
            
            rospy.logwarn(str(observation) + " " + str(reward))
            cumulated_reward += reward
            if highest_reward < cumulated_reward:
                highest_reward = cumulated_reward

            nextState = ''.join(map(str, observation))

            # Make the algorithm learn based on the results
            rospy.logwarn("# state we were=>" + str(state))
#            rospy.logwarn("# action that we took=>" + str(action))
            rospy.logwarn("# reward that action gave=>" + str(reward))
            rospy.logwarn("# episode cumulated_reward=>" + str(cumulated_reward))
            rospy.logwarn("# State in which we will start next step=>" + str(nextState))
#            qlearn.learn(state, action, reward, nextState)

            if not (done):
                rospy.logwarn("NOT DONE")
                state = nextState
            else:
                rospy.logwarn("DONE")
                last_time_steps = numpy.append(last_time_steps, [int(i + 1)])
                break
            rospy.logwarn("############### END Step=>" + str(i))
            #raw_input("Next Step...PRESS KEY")
            # rospy.sleep(2.0)
        m, s = divmod(int(time.time() - episode_start), 60)
        
        rospy.logwarn("EPISODE DURATION = " + str(m)+" min" + str(s)+ " sec")
        m, s = divmod(int(time.time() - start_time), 60)
        h, m = divmod(m, 60)
#        rospy.logerr(("EP: " + str(x + 1) + " - [alpha: " + str(round(qlearn.alpha, 2)) + " - gamma: " + str(
#            round(qlearn.gamma, 2)) + " - epsilon: " + str(round(qlearn.epsilon, 2)) + "] - Reward: " + str(
#            cumulated_reward) + "     Time: %d:%02d:%02d" % (h, m, s)))
        rospy.logerr(("EP: " + str(x + 1) + " - [alpha: " + str(round(Alpha, 2)) + " - gamma: " + str(
            round(Gamma, 2)) + " - epsilon: " + str(round(Epsilon, 2)) + "] - Reward: " + str(
            cumulated_reward) + "     Time: %d:%02d:%02d" % (h, m, s)))
    rospy.loginfo(("\n|" + str(nepisodes) + "|" + str(Alpha) + "|" + str(Gamma) + "|" + str(
        initial_epsilon) + "*" + str(epsilon_discount) + "|" + str(highest_reward) + "| PICTURE |"))

#    l = last_time_steps.tolist()
#    l.sort()

    # print("Parameters: a="+str)
    rospy.loginfo("Overall score: {:0.2f}".format(last_time_steps.mean()))
#    rospy.loginfo("Best 100 score: {:0.2f}".format(reduce(lambda x, y: x + y, l[-100:]) / len(l[-100:])))

    env.close()