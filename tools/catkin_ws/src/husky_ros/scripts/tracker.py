#!/usr/bin/env python3
from tkinter import W
import rospy
import numpy as np
import scipy as sp
import copy
import cv2
import torch


from sensor_msgs.msg import Image
from cv_bridge import CvBridge

#from yolor/utils/general.py
def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
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
        self.x_hist = []
        self.P_hist = []
        self.updated = []
        self.locked = False
        self.I = np.eye(x_initial.shape[0])#np.zeros((x_initial.shape[0], x_initial.shape[0]))
        #np.fill_diagonal(self.I, 1)
    
    def lock(self):
        self.locked = True

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
            self.P_hist.append(copy.deepcopy(self.P))
            self.P = np.dot((self.I - np.dot(K, self.H)), self.P)
            self.updated[-1] = True
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
    def get_ellipsis(self, conf=0.95):
        """Returns the center point, minor and major axis of the ellipsis"""
        
        #Get the observables (columns/rows)
        axs = [x[0] for x in np.argwhere(self.H)]
        #Construct the custom covariance matrix
        P_sub = self.P[np.ix_(axs, axs)]
        #Get the eigenvectors
        ev, evec = sp.linalg.eig(P_sub)
        conf_length = sp.stats.chi2.ppf(conf, P_sub.shape[0])
        #Note that this is just half the length (so you can do center +- length)
        axis_length = np.sqrt(ev*conf_length)
        ord = np.argsort(axis_length)

        #center, minor, major (actually, need to adjust this to be generalized to n-d, not only 2d)
        return self.x[axs], axis_length[ord[0]]*(evec[ord[0]]/np.linalg.norm(evec[ord[0]])), axis_length[ord[1]]*(evec[ord[1]]/np.linalg.norm(evec[ord[1]]))

    def reset(self):
        self.__init__(self.x_initial, self.P_initial, self.H, self.R, self.F)

class TrackingNode:
    def __init__(self):
        rospy.init_node("tracking_node")
        rospy.loginfo("Starting TrackingNode as name_node.")
        self._kmf = KalmanNd(np.asarray([[0], [0], [0], [0]]), 
                        1000*np.eye(4),
                        np.asarray([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]]),
                        np.asarray([[0.2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0.2, 0], [0, 0, 0, 0]]),
                        np.asarray([[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]]))
        
        #Feature tracking
        self.feature_tracker = cv2.SIFT_create()
        self._flann = (cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50)))
        self.image_pub = rospy.Publisher("sift_view", Image, queue_size=10)
        self.image_sub = rospy.Subscriber("detections", Image, self.handle_image)
        self._bridge = CvBridge() 
        self.des = None        
        self.kp = None
        self.img = None
        self.draw_params = dict(matchColor = (0,255,0),
                            singlePointColor = (255,0,0),
                            matchesMask = None,
                            flags = cv2.DrawMatchesFlags_DEFAULT)
        self.vis_match_pub = rospy.Publisher("vis_match", Image, queue_size=10)
        pass
    
    def handle_image(self, msg):
        
        kp = None
        des = None
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
        elif len(msg.detections) == 0:
            #Transition to none_available state
            self.state = 'none_detected' 
        
        if self.state in ["detected_new", "detected_loss", "none_detected"]:
            #Calculate features in ROIs (detected areas)
            box_mask = np.zeros((msg.detections[0].source_img.height, msg.detections[0].source_img.width), dtype=np.uint8)
            img = self._bridge.imgmsg_to_cv2(msg.detections[0].source_img)
            for detection in msg.detections:
                bbox = np.asarray([detection.bbox.center.x, detection.bbox.center.y,
                             detection.bbox.size_x, detection.bbox.size_y]).reshape(1,4)
                tl_x, tl_y, br_x, br_y = np.squeeze(xywh2xyxy(bbox), axis=0)
                box_mask[tl_y:(br_y+1), tl_x:(br_x+1)] = 1
            kp, des = self.feature_tracker.detectAndCompute(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), box_mask)
            if self.state == "detected_new":
                self.kp, self.des = kp, des
                self.img = copy.deepcopy(img)


        if self.state in ["detected_loss", "none_detected"]:
            #Do matching

            matches = self._flann.knnMatch(self.des, des, k=2)
            
            matchesMask = [[0,0] for i in range(len(matches))]
            # ratio test as per Lowe's paper
            for i,(m,n) in enumerate(matches):
                if m.distance < 0.7*n.distance:
                    matchesMask[i]=[1,0]
            self.draw_params["matchesMask"] = matchesMask
            img_match = cv2.drawMatchesKnn(self.img,self.kp,img, kp, matches, None, **self.draw_params)
            self.vis_match_pub.publish(self._bridge.cv2_to_imgmsg(img_match))


        



if __name__ == "__main__":
    name_node = TrackingNode()
    rospy.spin()