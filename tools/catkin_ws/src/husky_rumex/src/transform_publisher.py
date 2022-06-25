#!/usr/bin/env python
import rospy

if __name__ == "__main__":
    rospy.init_node("transform_publisher")
    rospy.loginfo("Starting transform_publisher.")

    
    while not rospy.is_shutdown():
        message_pub = rospy.Publisher("/tf/", type, queue_size=10)
        
        pass
