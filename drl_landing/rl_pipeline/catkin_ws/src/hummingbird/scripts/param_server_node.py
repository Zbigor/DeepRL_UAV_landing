#!/usr/bin/env python
# license removed for brevity
# -*- coding: utf-8 -*-

import rospy
#from std_msgs.msg import String

def talker():
    rospy.init_node('param_server', anonymous=True)
    while not rospy.is_shutdown():
        pass

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass