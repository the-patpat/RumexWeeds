#!/usr/bin/env python
import rospy
import tf
import json
from geometry_msgs.msg import Vector3, Quaternion
import os


def send_transform(br, tf_dict, dict_key, child, parent):
    br.sendTransform(translation=[tf_dict[dict_key]["translation"]["x"],tf_dict[dict_key]["translation"]["y"], tf_dict[dict_key]["translation"]["z"]], 
                    rotation=[tf_dict[dict_key]["rotation"]["x"], tf_dict[dict_key]["rotation"]["y"], tf_dict[dict_key]["rotation"]["z"], tf_dict[dict_key]["rotation"]["w"]], 
                    child=child, parent=parent, time=rospy.Time.now())


def pub_transform(seq_dir):
    rospy.init_node("tf_publisher", anonymous=True)
    br = tf.TransformBroadcaster()
    #Load the transforms from the json file
    #Is not located in the sequence folder but rather in the location folder
    with open(os.path.join(os.path.split(seq_dir)[0], "metadata.json"), 'r') as f:
        tf_dict = json.load(f)["transforms"]
    

    #Baselink transform
    while not rospy.is_shutdown():
        send_transform(br, tf_dict, "base_link", "base_link", "base_footprint")

        #IMU link transform
        send_transform(br, tf_dict, "imu_link", "imu_link", "base_link")

        #GNSS transform
        send_transform(br, tf_dict, "base_egnss", "gps", "base_link")

        #Camera transform
        send_transform(br, tf_dict, "camera_link", "camera_link", "base_link")

if __name__ == "__main__":
    try:
        pub_transform(rospy.get_param('/seq_dir'))
    except rospy.ROSInterruptException:
        pass