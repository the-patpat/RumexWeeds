#!/usr/bin/env python
import rospy
import glob
import os 
from std_msgs.msg import Time
from rosgraph_msgs.msg import Clock
from time import sleep

def set_ros_time(seq_dir, use_sim_time=True):
    message_pub = rospy.Publisher("clock", Clock, queue_size=10)
    rospy.init_node("set_ros_sim_time", anonymous=True)
        
    file_list = glob.glob(os.path.join(seq_dir,"imgs/*.png"))
    times = [int(os.path.splitext(os.path.split(x)[-1])[0].split('_')[-1]) for x in file_list]
    times_iter = iter(sorted(times))
    rate = rospy.Rate(5)

    while not rospy.is_shutdown():
        try:
            if use_sim_time:
                now = next(times_iter)
                rospy.loginfo("Set rospy time to {}. Output of rospy.Time.now() is: {}".format(now, rospy.Time.now()))
                message_pub.publish(Clock(rospy.Time(nsecs=now)))
                sleep(0.2)
            else:
                now = next(times_iter)
                rospy.Time.set(nsecs=now)
                rospy.loginfo("Set rospy time to {}. Output of rospy.Time.now() is: {}".format(now, rospy.Time.now()))
                rate.sleep()
        except StopIteration:
            rospy.signal_shutdown("Iterator exhausted")
if __name__ == '__main__':
    try:
        set_ros_time(rospy.get_param('/seq_dir'), rospy.get_param("/use_sim_time"))
    except rospy.ROSInterruptException:
        pass 