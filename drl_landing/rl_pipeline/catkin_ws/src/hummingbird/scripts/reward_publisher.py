#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from hummingbird.msg import Rewards


def writer():
    pub = rospy.Publisher('/hummingbird/reward/', Rewards, queue_size=1)
    rospy.init_node('reward_publisher',anonymous=True)
    pub.publish()
    while not rospy.is_shutdown():
        rospy.logwarn_once("Started the publisher")
    
    
if __name__ == '__main__':
    try:
        writer()
    except rospy.ROSInterruptException:
        pass    