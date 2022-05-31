#!/usr/bin/env python
# license removed for brevity
import rospy
from std_msgs.msg import Header
from geometry_msgs.msg import PoseWithCovariance, TwistWithCovariance, Twist, Pose, Quaternion, Vector3
from nav_msgs.msg import Odometry
#message
import json


def talker():
    
    #This will be variable or replaced by either loc/seq or just loc with all seqs
    with open('/home/pat/onedrive/Thesis/Experiments/RumexWeeds/data/20210806_hegnstrup/seq0/odom.json', 'r') as f:
        odo_hist = json.load(f)#, object_pairs_hook=OrderedDict)

    keys = list(odo_hist.keys())
    keys = iter(sorted(keys))
    pub = rospy.Publisher('odo_msg', Odometry, queue_size=10)
    rospy.init_node('odo_publisher', anonymous=True)
    rate = rospy.Rate(5) # 10hz
    while not rospy.is_shutdown():

        try:
            frame = next(keys)
            dict_message = odo_hist[frame]
            # rospy.loginfo("Keys of dict: %s in frame %s" % (str(dict_message.keys()), frame))

        except StopIteration:
            rospy.signal_shutdown('Iterator exhausetd')

        header = Header(stamp=rospy.Time(**dict_message['header']['stamp']),
                seq=dict_message['header']['seq'], 
                frame_id=dict_message['header']['frame_id'])
        twist = TwistWithCovariance(twist=Twist(linear=Vector3(**dict_message['twist']['twist']["linear"]),
                                    angular=Vector3(**dict_message['twist']['twist']["angular"])),
                                    covariance=dict_message['twist']['covariance'])
        pose = PoseWithCovariance(pose=Pose(position=Vector3(**dict_message['pose']['pose']['position']), 
                                            orientation=Quaternion(**dict_message['pose']['pose']['orientation'])),
                                covariance=dict_message["pose"]["covariance"])
        
        # rospy.loginfo("pose orig: %s pose: %s" % (pose_orig, pose))        
    
        #Deleting the extracted key-value pairs does not work, will result in an error
        odo_msg = Odometry(header=header, twist=twist, pose=pose, child_frame_id=dict_message['child_frame_id'])
        
        pub.publish(odo_msg) 
        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
