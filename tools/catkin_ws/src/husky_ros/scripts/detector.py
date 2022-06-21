#!/usr/bin/env python3
import queue
import rospy
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose, BoundingBox2D
from geometry_msgs.msg import Pose2D

import sys
sys.path.insert(0, "/home/pat/")
sys.path.insert(0, "/home/pat/yolor")
from yolor.models.models import *
from yolor.utils.general import non_max_suppression, xyxy2xywh

import torch

from cv_bridge import CvBridge

class DetectorNode:
    def __init__(self):
        rospy.init_node("yolor_detector")
        rospy.loginfo("Starting DetectorNode.")
        
        # Load model
        rospy.loginfo("Loading YOLOR model")
        self.model = Darknet('/home/pat/yolor/cfg/yolor_csp_rumex.cfg', (640, 640)).cuda()
        self.model.load_state_dict(torch.load('/home/pat/yolor/runs/train/yolor_csp5/weights/best_ap.pt', map_location='cpu')['model'])
        self.model.to('cuda').eval()
        self.model.half()  # to FP16
        self.model(torch.zeros((1,3,640,640), device='cuda'))
        rospy.loginfo("Loaded YOLOR model")
        self.subscriber = rospy.Subscriber("image", Image, self.handle_image, queue_size=10)
        self.publisher = rospy.Publisher("detections", Detection2DArray, queue_size=10)
        

        self.bridge = CvBridge()

    def handle_image(self, data):
        img = self.bridge.imgmsg_to_cv2(data)
        img = torch.from_numpy(img).to('cuda')
        img = img.half() 
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = self.model(img, augment=False)[0]

        pred = non_max_suppression(pred, 0.2, 0.5, classes=["rumex"], agnostic=False)

        #Pred shape is (batch, detections, 6(x1,y1,x2,y2,conf,cls))
        pred = pred.squeeze(0)
        detections = []
        msg = Detection2DArray(detections=[])
        for p in pred:
            obj_hyp = ObjectHypothesisWithPose(id=0, score=p[-2])
            (xc,yc,w,h) = xyxy2xywh(p[0:4])
            xc *= 640
            yc *= 640
            w *= 640
            h *= 640
            bbox = BoundingBox2D(center=Pose2D(x=xc, y=yc), size_x=w, size_y=h)
            msg.detections.append(Detection2D(bbox=bbox, results=[obj_hyp], source_img=data))
        
        self.publisher.publish(msg)
            

if __name__ == "__main__":
    try:
        name_node = DetectorNode()
        rospy.spin()
    except rospy.ROSInternalException:
        pass