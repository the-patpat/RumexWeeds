#!/usr/bin/env python
# license removed for brevity
import rospy
from std_msgs.msg import Header
from sensor_msgs.msg import Image
#message
import json
import numpy as np
import cv2
import os
from cv_bridge import CvBridge


def talker(seq_dir):
    
    #This will be variable or replaced by either loc/seq or just loc with all seqs
    with open(os.path.join(seq_dir, 'rgb.json'), 'r') as f:
        rgb_hist = json.load(f)#, object_pairs_hook=OrderedDict)

    keys = list(rgb_hist.keys())
    keys = iter(sorted(keys))
    pub = rospy.Publisher('image', Image, queue_size=10)
    rospy.init_node('rgb_publisher', anonymous=True)
    rate = rospy.Rate(5) # 10hz
    bridge = CvBridge()

    while not rospy.is_shutdown():

        try:
            frame = next(keys)
            
            #Load image to the corresponding header
            parts = frame.split('_')
            img_path = "_".join(parts[0:2])+ "_rgb_" +"_".join(parts[2:])
            img_path = os.path.join(seq_dir, 'imgs', img_path+".png")
            img = cv2.imread(img_path)
 
            dict_message = rgb_hist[frame]

        except StopIteration:
            rospy.signal_shutdown('Iterator exhausetd')

        if img is not None:
            header = Header(stamp=rospy.Time(**dict_message['stamp']),
                    seq=dict_message['seq'], 
                    frame_id=dict_message['frame_id'].encode('utf8', 'ascii'))
            img_msg = bridge.cv2_to_imgmsg(img) 
            img_msg.header = header
            # img_msg = Image(header=header, height=img.shape[0], width=img.shape[1],
                # encoding="bgr8", step=3*img.shape[1], data=img.flatten().tolist())

            pub.publish(img_msg) 
        else:
            rospy.logwarn("Image file {} not found!".format(img_path))
        rate.sleep()

if __name__ == '__main__':
    try:
        talker(rospy.get_param('/seq_dir'))
    except rospy.ROSInterruptException:
        pass
