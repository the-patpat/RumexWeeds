#!/usr/bin/env python
# license removed for brevity
from collections import OrderedDict
import rospy
from std_msgs.msg import Header 
#GPS message
from sensor_msgs.msg import NavSatFix, NavSatStatus
import json
import os

def talker(seq_dir):
    
    #This will be variable or replaced by either loc/seq or just loc with all seqs
    with open('/home/pat/onedrive/Thesis/Experiments/RumexWeeds/data/20210806_hegnstrup/seq0/gps.json', 'r') as f:
        gps_hist = json.load(f, object_pairs_hook=OrderedDict)

    keys = list(gps_hist.keys())
    keys = iter(sorted(keys))
    pub = rospy.Publisher('gps_msg', NavSatFix, queue_size=10)
    rospy.init_node('gps_publisher', anonymous=True)
    rate = rospy.Rate(5) # 10hz
    while not rospy.is_shutdown():
        try:
            dict_message = gps_hist[next(keys)]
        except StopIteration:
            rospy.signal_shutdown('Iterator exhausetd')
        gps_message = NavSatFix()
        header = Header(stamp=rospy.Time(**dict_message['header']['stamp']),
                        seq=dict_message['header']['seq'], 
                        frame_id=dict_message['header']['frame_id'])
        gps_message.header = header 
        gps_message.status = NavSatStatus(**dict_message['status'])
        gps_message.latitude = dict_message['latitude']
        gps_message.longitude = dict_message['longitude']
        gps_message.altitude = dict_message['altitude']
        gps_message.position_covariance = dict_message['position_covariance']
        gps_message.position_covariance_type = dict_message['position_covariance_type']
        pub.publish(gps_message)
        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
