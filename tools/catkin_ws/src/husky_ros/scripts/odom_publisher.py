#!/usr/bin/env python
# license removed for brevity
import rospy
from std_msgs.msg import Header
from geometry_msgs.msg import PoseWithCovariance, TwistWithCovariance, Twist, Pose, Quaternion, Vector3
from nav_msgs.msg import Odometry
import tf
#message
import json
import os

def talker(seq_dir):
    
    #This will be variable or replaced by either loc/seq or just loc with all seqs
    with open(os.path.join(seq_dir, 'odom.json'), 'r') as f:
        odo_hist = json.load(f)#, object_pairs_hook=OrderedDict)

    keys = list(odo_hist.keys())
    keys = iter(sorted(keys))
    pub = rospy.Publisher('odo_msg', Odometry, queue_size=10)
    rospy.init_node('odo_publisher', anonymous=True)
    rate = rospy.Rate(5) # 10hz
    br = tf.TransformBroadcaster()

    prev_stamp = None
    init_position = None

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
        
        if init_position is None:
            init_position = Vector3(**dict_message['pose']['pose']['position'])
            position = Vector3()
        else:
            position = Vector3(**dict_message['pose']['pose']['position'])
            position.x -= init_position.x
            position.y -= init_position.y
            position.z -= init_position.z 

        pose = PoseWithCovariance(pose=Pose(position=position, 
                                            orientation=Quaternion(**dict_message['pose']['pose']['orientation'])),
                                covariance=dict_message["pose"]["covariance"])
        br.sendTransform([pose.pose.position.x, pose.pose.position.y,pose.pose.position.z],
                        [pose.pose.orientation.x, pose.pose.orientation.y,pose.pose.orientation.z,pose.pose.orientation.w],
                        rospy.Time(**dict_message['header']['stamp']), 'base_footprint', 'odom')
        
        if prev_stamp is not None:
            tdiff =  rospy.Time(**dict_message['header']['stamp']) - prev_stamp
            #rospy.loginfo("Timediff: {:.2f}".format(tdiff.to_sec()))
            assert tdiff.to_sec() > 0, "Timediff should be positive"
            
        
        prev_stamp = rospy.Time(**dict_message['header']['stamp'])
        
        # rospy.loginfo("pose: %s" % pose.pose.position)        
    
        #Deleting the extracted key-value pairs does not work, will result in an error
        odo_msg = Odometry(header=header, twist=twist, pose=pose, child_frame_id=dict_message['child_frame_id'])
        
        pub.publish(odo_msg) 
        rate.sleep()

if __name__ == '__main__':
    try:
        talker(rospy.get_param('/seq_dir'))
    except rospy.ROSInterruptException:
        pass
