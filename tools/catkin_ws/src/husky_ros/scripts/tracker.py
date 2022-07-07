#!/usr/bin/env python3
import sys
import rospy
import numpy as np
import scipy as sp
import scipy.spatial 
import scipy.stats
import scipy.linalg
import copy
import cv2
#import torch
import time
import tf

#Little trick for mixed python2-python3 environments, cv_bridge is compiled with python3 too.
#This one uses python2 and cv_bridge, thus we need to make sure that the python2 version of cv_bridge is loaded
#sys.path.insert(0, '/opt/ros/melodic/lib/python2.7/dist-packages')

from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray
from cv_bridge import CvBridge
from std_msgs.msg import String, Float32
from husky_ros.msg import Detection2DArrayWithImage
from geometry_msgs.msg import Point

#TODO: Get the transform with tf listener (done)
#TODO: Get the bounding box coordinates in camera frame coordinates
#TODO: Transform the camera frame coordinates to odom frame
#TODO: Replace the kalman filter with odom frame coordinates
#TODO: integrate the odom movement as external vector for state prediction 


#from yolor/utils/general.py
def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y =  np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y
    

#Kalman filter implementation
class KalmanNd:
    def __init__(self, x_initial, P_initial, H, R, F):
        self.x_initial = x_initial
        self.P_initial = P_initial
        self.x = x_initial # mean_vector (states)
        self.P = P_initial # covariance matrix
        self.H = H #Observability matrix
        self.R = R #Measurement uncertainty
        self.F = F
        self.x_hist = [None]
        self.P_hist = [None]
        self.updated = [None]
        self.locked = False
        self.I = np.eye(x_initial.shape[0])#np.zeros((x_initial.shape[0], x_initial.shape[0]))
        self.z_hist = [None]
        #np.fill_diagonal(self.I, 1)
    
    def lock(self):
        print("Locking kalman filter")
        self.locked = True

    def unlock(self):
        print("Unlocking kalman filter")
        self.locked = False

    def update(self, Z):
        """Kalman filter measurement update
        Alters the object and updates the Kalman filter with a current measurement.
        Paramaters
        -----------
        Z : ndarray 
            The measurements to use for the update
        """
        if not self.locked:
            y = Z.reshape(-1,1) - np.dot(self.H, self.x)
            S = np.dot(np.dot(self.H,self.P), self.H.T) + self.R
            K = np.dot(np.dot(self.P,self.H.T), np.linalg.pinv(S))
            
            #Removed the update, but rather overwrite the prediction with the update
            self.x_hist[-1] = (copy.deepcopy(self.x))
            self.x = self.x + np.dot(K,y)
            self.P_hist[-1] = copy.deepcopy(self.P)
            self.P = np.dot((self.I - np.dot(K, self.H)), self.P)
            self.updated[-1] = True
            self.z_hist[-1] = copy.deepcopy(Z)
    def predict(self, u=None):
        if not self.locked:
            ### insert predict function
            if u is None:
                u = np.zeros_like(self.x)
            self.x_hist.append(copy.deepcopy(self.x))
            self.x = np.dot(self.F,self.x) + u
            self.P_hist.append(copy.deepcopy(self.P))
            self.P = np.dot(np.dot(self.F, self.P), self.F.T)
            self.updated.append(False)
            self.z_hist.append(None)
    def get_ellipsis(self, conf=0.95):
        """Returns the center point, minor and major axis of the ellipsis"""
        
        #Get the observables (columns/rows)
        axs = [x[0] for x in np.argwhere(self.H)]
        #Construct the custom covariance matrix
        P_sub = self.P[np.ix_(axs, axs)]
        #Get the eigenvectors
        ev, evec = scipy.linalg.eig(P_sub)
        conf_length = scipy.stats.chi2.ppf(conf, P_sub.shape[0])
        #Note that this is just half the length (so you can do center +- length)
        axis_length = np.sqrt(ev*conf_length)
        ord = np.argsort(axis_length)

        #center, minor, major (actually, need to adjust this to be generalized to n-d, not only 2d)
        return self.x[axs], axis_length[ord[0]]*(evec[ord[0]]/np.linalg.norm(evec[ord[0]])), axis_length[ord[1]]*(evec[ord[1]]/np.linalg.norm(evec[ord[1]]))

    def reset(self):
        self.__init__(self.x_initial, self.P_initial, self.H, self.R, self.F)

    def __repr__(self):
        str = "Kalman filter diagnostics\n"
        str +="\tCurrent state vector:\n\t"
        for i, val in enumerate(self.x):
            str += f"\tx[{i}]: {val}"
        str += "\n"
        str +="\tCurrent covariance:\n"
        str += f"\t\t{self.P}\n"
        if len(self.z_hist) > 0:
            str += "\tLast measurement:\n"
            str += f"\t\t{self.z_hist[-1].flatten() if self.z_hist[-1] is not None else self.z_hist[-1]}\n"
        if True in self.updated:
            str += f"\t\tLast measurement was {self.updated[::-1].index(True)} cycles ago\n"
        str += f"\tCurrent filter state is {'locked' if self.locked else 'unlocked'}"
        return str



class TrackingNode:
    def __init__(self):
        rospy.init_node("tracking_node")
        rospy.loginfo("Starting TrackingNode as name_node.")
        self._kmf = KalmanNd(np.asarray([[0], [0], [0], [0]]), 
                        1000*np.eye(4),
                        np.asarray([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]]),
                        np.asarray([[0.8, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0.8, 0], [0, 0, 0, 0]]),
                        np.asarray([[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]]))
        self._kmf.lock()

        print(self._kmf)
        
        #Feature tracking
        # self.feature_tracker = cv2.SIFT_create()
        self.feature_tracker = cv2.ORB_create(nfeatures=50000)
        FLANN_INDEX_LSH = 6
        index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 1) #2 
        # self._flann = (cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50)))
        self._flann = cv2.FlannBasedMatcher(index_params, dict(checks=50))
        self.image_sub = rospy.Subscriber("detections", Detection2DArrayWithImage, self.handle_image)
        self._bridge = CvBridge() 
        self.state = None
        self.des = None        
        self.kp = None
        self.img = None
        self.detection_count = 0
        self.draw_params = dict(matchColor = (0,255,0),
                            singlePointColor = (255,0,0),
                            matchesMask = None,
                            flags = cv2.DrawMatchesFlags_DEFAULT)
        self.vis_match_pub = rospy.Publisher("tracker/vis_match", Image, queue_size=10)
        self.state_pub = rospy.Publisher("tracker/tracker_state", String, queue_size=10)
        self.sift_time_pub = rospy.Publisher("tracker/sift_time/detect", Float32, queue_size=10)
        self.sift_time_match_pub = rospy.Publisher("tracker/sift_time/match", Float32, queue_size=10)
        self.detection_count_pub = rospy.Publisher("/tracker/detection_count", Float32, queue_size=10)
        self._dilateelement = cv2.getStructuringElement(cv2.MORPH_RECT, (201,201))
        self.tf_listener = tf.TransformListener()
        self.detection_database = np.zeros((0,0)) 
        self.kalman_x_pub = rospy.Publisher("/tracker/kalman/x", Float32, queue_size=10)
        self.kalman_y_pub = rospy.Publisher("/tracker/kalman/y", Float32, queue_size=10)
        self.kalman_pt_pub = rospy.Publisher("/tracker/kalman/pt", Point, queue_size=10)
        pass
    
    def handle_image(self, msg):

        img = self._bridge.imgmsg_to_cv2(msg.source_img.img)

        trans, rot = None, None
        try:
            trans, rot = self.tf_listener.lookupTransform('odom', 'camera_link', rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
           pass 
        #rospy.loginfo("Current location of camera is: x/y/z {}".format(trans)) 
    
        #Make kalman prediction
        self._kmf.predict()

        #Get length of detections
        if len(msg.detections) > 0 and abs(self.detection_count - len(msg.detections)) == 0 :
            #NMS + confidence thresholded detections available
            self.state = 'detected_same'
            self.detection_count = len(msg.detections)
        elif len(msg.detections) > 0 and  len(msg.detections) - self.detection_count < 0: 
            #Lost an instance, either false negative or lost false positive or plant out of the picture
            #This needs matching
            self.state = 'detected_loss'
            self.detection_count = len(msg.detections)
        elif len(msg.detections) > 0 and len(msg.detections) - self.detection_count  > 0: 
            #Got a new instance
            #Recalculate features as we have something new to track
            #This needs matching
            self.state = 'detected_new'
            self.detection_count = len(msg.detections)
            
            #Kalman filter is locked until we get the first detection
        elif len(msg.detections) == 0:
            #Transition to none_available state
            self.state = 'none_detected' 
            self.detection_count = 0
        
        if self.state in ["detected_new", "detected_loss", "none_detected"]:
            #Calculate features in ROIs (detected areas)
            
            #Empty array, holds bounding box candidates from recieved detections
            detections = []

            if self.state != "none_detected":
                box_mask = np.zeros((msg.source_img.img.height, msg.source_img.img.width), dtype=np.uint8)
            else:
                box_mask = None
            for detection in msg.detections:
                bbox = np.asarray([detection.bbox.center.x, detection.bbox.center.y,
                             detection.bbox.size_x, detection.bbox.size_y]).reshape(1,4)
                detections.append(bbox)
                tl_x, tl_y, br_x, br_y = np.squeeze(xywh2xyxy(bbox), axis=0).astype(int)
                box_mask[tl_y:(br_y+1), tl_x:(br_x+1)] = 1
                box_mask = cv2.dilate(box_mask, self._dilateelement)
            t0 = time.time()
            kp, des = self.feature_tracker.detectAndCompute(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), box_mask)
            t1 = time.time()
            self.sift_time_pub.publish(Float32(t1-t0))

            #Compare candidate boxes with boxes from previous pictures
            #TODO handle conflicts, resolution
            #For now, just use the first match
            if self.detection_database.shape[0] > 0 and len(detections) > 0:
                det_curr = np.asarray(detections).reshape(-1,4)
                dist = scipy.spatial.distance.cdist(det_curr[:, :2], self.detection_database[:, :2])
                if dist.shape[0] > dist.shape[1]:
                    #More detections now than in the previous frame
                    #Match direction: previous -> current. Unmatched are the new ones
                    
                    #Sorting column-wise, sort the column vector
                    closest_previous_to_now = np.argsort(dist, axis=0)
                    
                    #Get the minimum argument of each column vector (first row), make hist (count)
                    hist, _ = np.histogram(closest_previous_to_now[0, :], bins=np.max(closest_previous_to_now)+1)
                    
                    #Target conflict, multiple sources want to match to the same target
                    if (hist > 1).any():
                        rospy.logwarn("Multiple sources matched to one target")

                    for i,j in enumerate(closest_previous_to_now[0, :]):
                        print(f"matching previous detection {i} to current detection {j}")
                    self._kmf.update(np.asarray([det_curr[closest_previous_to_now[0,0], 0], 0,det_curr[closest_previous_to_now[0,0], 1], 0]))
                elif dist.shape[0] < dist.shape[1]:
                    #Other direction: current -> previous
                    closest_now_to_previous = np.argmin(dist, axis=1) 
                    hist, _ = np.histogram(closest_now_to_previous, bins=np.max(closest_now_to_previous)+1)
                    if (hist > 1).any():
                        rospy.logwarn("Multiple sources matched to one target")
                    for i,j in enumerate(closest_now_to_previous):
                        print(f"matching current detection {i} to previous detection {j}")
                    m = np.argwhere(closest_now_to_previous == 0)
                    if m.shape[0] > 0:
                        self._kmf.update(np.asarray([det_curr[m[0], 0], 0, det_curr[m[0], 1], 0]))
                    pass
                else:
                    pass

                print(dist)
            
            self.detection_database = np.asarray(detections).reshape(-1,4)
                

            if self.state == "detected_new":
                if self._kmf.locked:
                    #Initial lock, initialize kalman update
                    print("got first measurement, update kalman filter and unlock")
                    self._kmf.unlock()
                    self._kmf.update(np.asarray([self.detection_database[0,0], 0, self.detection_database[0, 2], 0]))
                self.kp, self.des = kp, des
                self.img = copy.deepcopy(img)

        if len(self._kmf.x_hist) > 0:
            img = cv2.circle(img, (int(self._kmf.x_hist[-1][0]), int(self._kmf.x_hist[-1][2])), 10, ((255, 0, 0) if self._kmf.updated[-1] else (0,0,255)), cv2.FILLED)
            center, minor, major = self._kmf.get_ellipsis(conf=0.5)
            center = center.astype(int).flatten()
            minor = minor.astype(int)
            major = major.astype(int)
            img = cv2.ellipse(img, center, (np.linalg.norm(major).astype(int), np.linalg.norm(minor).astype(int)), -np.arctan2(major[1], major[0]), 0, 360, ((255, 0, 0) if self._kmf.updated[-1] else (0,0,255)), 10)
        self.vis_match_pub.publish(self._bridge.cv2_to_imgmsg(img))

        if self.state in ["detected_loss", "none_detected"]:
            #Do matching

            t0 = time.time()
            matches = self._flann.knnMatch(self.des, des, k=2)
            t1 = time.time()
            self.sift_time_match_pub.publish(Float32(t1-t0))
            
            matchesMask = [[0,0] for i in range(len(matches))]
            # ratio test as per Lowe's paper
            matches = np.asarray(matches, dtype=object)
            if len(matches.shape) > 1:
                for i,(m,n) in enumerate(matches):
                    if m.distance < 0.7*n.distance:
                        matchesMask[i]=[1,0]
            self.draw_params["matchesMask"] = matchesMask
            # img_match = cv2.drawMatchesKnn(self.img,self.kp,img, kp, matches, None, **self.draw_params)
            # self.vis_match_pub.publish(self._bridge.cv2_to_imgmsg(img_match))

        self.state_pub.publish(self.state)
        self.detection_count_pub.publish(Float32(self.detection_count))
        self.kalman_x_pub.publish(self._kmf.x[0])
        self.kalman_y_pub.publish(self._kmf.x[2])
        self.kalman_pt_pub.publish(Point(self._kmf.x[0], self._kmf.x[2], 0))
        print(self._kmf)


        



if __name__ == "__main__":
    name_node = TrackingNode()
    rospy.spin()