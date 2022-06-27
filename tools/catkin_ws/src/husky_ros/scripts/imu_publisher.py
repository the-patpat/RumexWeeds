#!/usr/bin/env python
# license removed for brevity
from collections import OrderedDict
import rospy
from std_msgs.msg import Header 
from geometry_msgs.msg import Quaternion, Vector3
from datetime import time
#message
from sensor_msgs.msg import Imu 
import json
import copy
import os


def talker(seq_dir):
    
    #This will be variable or replaced by either loc/seq or just loc with all seqs
    with open(os.path.join(seq_dir, 'imu.json'), 'r') as f:
        imu_hist = json.load(f, object_pairs_hook=OrderedDict)

    keys = list(imu_hist.keys())
    keys = iter(sorted(keys))
    pub = rospy.Publisher('imu_msg', Imu, queue_size=10)
    rospy.init_node('imu_publisher', anonymous=True)
    rate = rospy.Rate(5) # 10hz
    while not rospy.is_shutdown():

        try:
            frame = next(keys)
            dict_message = imu_hist[frame]
            # rospy.loginfo("Keys of dict: %s in frame %s" % (str(dict_message.keys()), frame))

        except StopIteration:
            rospy.signal_shutdown('Iterator exhausetd')

        header = Header(stamp=rospy.Time(**dict_message['header']['stamp']),
                seq=dict_message['header']['seq'], 
                frame_id=dict_message['header']['frame_id'])
        lin_acc = Vector3(**dict_message['linear_acceleration'])
        ang_vel = Vector3(**dict_message['angular_velocity'])
        orient = Quaternion(**dict_message['orientation'])
        
        
        #Deleting the extracted key-value pairs does not work, will result in an error
        imu_msg = Imu(linear_acceleration=lin_acc, header=header,
                        angular_velocity=ang_vel, orientation=orient, 
                        linear_acceleration_covariance = dict_message['linear_acceleration_covariance'],
                        angular_velocity_covariance = dict_message['angular_velocity_covariance'], 
                        orientation_covariance = dict_message['orientation_covariance'])
        pub.publish(imu_msg) 
        rate.sleep()

if __name__ == '__main__':
    try:
        talker(rospy.get_param('/seq_dir'))
    except rospy.ROSInterruptException:
        pass
