#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose, BoundingBox2D
from geometry_msgs.msg import Pose2D
from std_msgs.msg import Float32

import sys
sys.path.insert(0, "/")

sys.path.insert(0, "/yolor")
from yolor.models.models import *
from yolor.utils.general import non_max_suppression, xyxy2xywh
from yolor.utils.datasets import letterbox
from yolor.utils.plots import plot_one_box
import numpy as np
import torch
from copy import deepcopy
import time
from husky_ros.msg import Detection2DArrayWithImage, ImageWithId

from cv_bridge import CvBridge

class DetectorNode:
    def __init__(self):
        print("Initializeing node")
        rospy.init_node("yolor_detector")
        print("initialized node")
        rospy.loginfo("Starting DetectorNode.")
        # Load model
        rospy.loginfo("Loading YOLOR model")
        self.model = Darknet('/yolor/cfg/yolor_csp_rumex.cfg', (640, 640)).cuda()
        self.model.load_state_dict(torch.load('/yolor/best_ap.pt', map_location='cpu')['model'])
        self.model.to('cuda').eval()
        self.model.training = False
        self.model.half()  # to FP16
        self.model(torch.zeros((1,3,640,640), device='cuda').half())
        rospy.loginfo("Loaded YOLOR model")
        self.subscriber = rospy.Subscriber("image_id", ImageWithId, self.handle_image, queue_size=10)
        self.publisher = rospy.Publisher("detections", Detection2DArrayWithImage, queue_size=10)
        self.vis_publisher = rospy.Publisher("visual_detections", Image, queue_size=10)
        self.bridge = CvBridge()
        self.inference_time_publisher = rospy.Publisher("detector/inference_time", Float32, queue_size=10)
        self.model_time_publisher = rospy.Publisher("detector/model_time", Float32, queue_size=10 )
        self.nms_time_publisher = rospy.Publisher("detector/nms_time", Float32, queue_size=10)
        self.mask_publisher = rospy.Publisher("detection_mask", Image, queue_size=10)
        rospy.loginfo("Initialization finished. Starting to listen....")

    def handle_image(self, data):
        t0 = time.time()
        img = self.bridge.imgmsg_to_cv2(data.img)
        im0 = deepcopy(img)

        img = letterbox(img, new_shape=(640,640))[0]
        rospy.logdebug(f"letterbox return np array with shape {img.shape}")
        img = np.ascontiguousarray(img[:, :, ::-1]) #BGR to RGB
        rospy.logdebug("Moving to GPU")
        img = torch.from_numpy(img).to('cuda')
        rospy.logdebug("Moved to GPU")
        img = img.half() 
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        img = torch.permute(img, (0,3,1,2))
        with torch.no_grad():
            rospy.logdebug(f"Passing tensor of size {img.shape} and dtype {img.dtype} to model")
            t01 = time.time()
            pred = self.model(img, augment=False)[0]
            t1 = time.time()
            self.model_time_publisher.publish(Float32(t1-t01))


        #classes=[0] -> rumex single class, id=0. [0] at the end: batch_size=1, and nms output is list with batch_size elements (nx6 tensors)
        t01 = time.time()
        pred = non_max_suppression(pred, 0.2, 0.5, classes=[0], agnostic=False)[0]
        t1 = time.time()
        self.nms_time_publisher.publish(Float32(t1-t01))

        #Pred shape is (batch, detections, 6(x1,y1,x2,y2,conf,cls))
        msg = Detection2DArrayWithImage(detections=[])
        mask = np.zeros_like(im0, dtype=np.uint8)
        for p in pred:
            obj_hyp = ObjectHypothesisWithPose(id=0, score=p[-2])
            p[:4] = scale_coords(img.shape[2:], p[:4].unsqueeze(0), im0.shape).round().squeeze(0)
            (xc,yc,w,h) = xyxy2xywh(p[0:4].unsqueeze(0)).squeeze(0)
            xc = int(xc)
            yc = int(yc)
            w = int(w)
            h = int(h)
            mask[(yc - h//8):(yc + h//8), (xc - w//8):(xc + w//8), :] = (128,64,128)
            # xc *= 640
            # yc *= 640
            # w *= 640
            # h *= 640
            bbox = BoundingBox2D(center=Pose2D(x=xc, y=yc), size_x=w, size_y=h)
            # Rescale boxes from img_size to im0 size
            msg.detections.append(Detection2D(bbox=bbox, results=[obj_hyp], source_img=self.bridge.cv2_to_imgmsg(deepcopy(im0))))
            
            plot_one_box(p[:4], im0, label="rumex: %.2f" % p[-2], color=(255,0,0), line_thickness=3)
        msg.source_img = ImageWithId(img=self.bridge.cv2_to_imgmsg(deepcopy(im0)), id=data.id)
        
        t1 = time.time()
        self.inference_time_publisher.publish(Float32(t1-t0))
        self.publisher.publish(msg)
        self.mask_publisher.publish(self.bridge.cv2_to_imgmsg(mask))
        self.vis_publisher.publish(self.bridge.cv2_to_imgmsg(im0))            

if __name__ == "__main__":
    try:
        name_node = DetectorNode()
        rospy.spin()
    except rospy.ROSInternalException:
        pass