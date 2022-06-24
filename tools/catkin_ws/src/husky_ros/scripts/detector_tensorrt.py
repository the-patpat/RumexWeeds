#!/usr/bin/env python3
import queue
import rospy
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose, BoundingBox2D
from geometry_msgs.msg import Pose2D

import sys
sys.path.insert(0, "/home/pat/scratch")
sys.path.insert(0, "/home/pat/scratch/yolor")
from yolor.models.models import *
from yolor.utils.general import non_max_suppression, xyxy2xywh
from yolor.utils.datasets import letterbox
from yolor.utils.plots import plot_one_box
import numpy as np
import torch
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

from copy import deepcopy

from cv_bridge import CvBridge

class DetectorNode:
    def __init__(self):
        print("Initializeing node")
        rospy.init_node("yolor_detector_tensorrt")
        print("initialized node")
        rospy.loginfo("Starting DetectorNode.")
        # Load model
        rospy.loginfo("Loading YOLOR model")
        self.model = self._predict
        rospy.loginfo("Loaded YOLOR model")
        self.subscriber = rospy.Subscriber("image", Image, self.handle_image, queue_size=10)
        self.publisher = rospy.Publisher("detections", Detection2DArray, queue_size=10)
        self.vis_publisher = rospy.Publisher("visual_detections", Image, queue_size=10)
        self.bridge = CvBridge()
        
        #Initialize tensorrt
        f = open("/home/pat/scratch_dtu_synced/yolor/runs/train/yolor_csp5/weights/best_ap.trt", "rb")
        self._runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING)) 
        self._engine = self.runtime.deserialize_cuda_engine(f.read())
        self._context = self.engine.create_execution_context()
        input_batch = np.zeros((1,3,640,640), dtype=np.float32)
        self.output = np.zeros((1,25200,6), dtype=np.float32)
        self._d_input = cuda.mem_alloc(1 * input_batch.nbytes)
        self._d_output = cuda.mem_alloc(1 * self.output.nbytes)
        self._bindings = [int(self._d_input), int(self._d_output)]
        self._stream = cuda.Stream()
        rospy.loginfo("Initialization finished. Starting to listen....")
            
    def _predict(self, batch): # result gets copied into output
        # transfer input data to device
        print("Copying batch to cuda device")
        cuda.memcpy_htod_async(self._d_input, batch, self._stream)
        # execute model
        print("Executing inference")
        self._context.execute_async_v2(self._bindings, self._stream.handle, None)
        # transfer predictions back
        print("transfering back")
        cuda.memcpy_dtoh_async(self.output, self._d_output, self._stream)
        # syncronize threads
        print("synchronizing")
        self._stream.synchronize()
        return self.output

    def handle_image(self, data):
        img = self.bridge.imgmsg_to_cv2(data)
        im0 = deepcopy(img)

        img = letterbox(img, new_shape=(640,640))[0]
        rospy.loginfo(f"letterbox return np array with shape {img.shape}")
        img = np.ascontiguousarray(img[:, :, ::-1]) #BGR to RGB
        #img = torch.from_numpy(img).to('cuda')
        #img = img.half() 
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        img = np.transpose(img, (0,3,1,2))
        rospy.loginfo(f"Passing ndarray of size {img.shape} and dtype {img.dtype} to model")
        pred = self.model(img)

        #classes=[0] -> rumex single class, id=0. [0] at the end: batch_size=1, and nms output is list with batch_size elements (nx6 tensors)
        pred = non_max_suppression(pred, 0.2, 0.5, classes=[0], agnostic=False)[0]

        #Pred shape is (batch, detections, 6(x1,y1,x2,y2,conf,cls))
        detections = []
        msg = Detection2DArray(detections=[])
        for p in pred:
            obj_hyp = ObjectHypothesisWithPose(id=0, score=p[-2])
            p[:4] = scale_coords(img.shape[2:], p[:4].unsqueeze(0), im0.shape).round().squeeze(0)
            (xc,yc,w,h) = xyxy2xywh(p[0:4].unsqueeze(0)).squeeze(0)
            # xc *= 640
            # yc *= 640
            # w *= 640
            # h *= 640
            bbox = BoundingBox2D(center=Pose2D(x=xc, y=yc), size_x=w, size_y=h)
            # Rescale boxes from img_size to im0 size
            msg.detections.append(Detection2D(bbox=bbox, results=[obj_hyp], source_img=deepcopy(im0)))
            
            plot_one_box(p[:4], im0, label="rumex: %.2f" % p[-2], color=(255,0,0), line_thickness=3)

            
        
        self.publisher.publish(msg)
        self.vis_publisher.publish(self.bridge.cv2_to_imgmsg(im0))
            

if __name__ == "__main__":
    try:
        name_node = DetectorNode()
        rospy.spin()
    except rospy.ROSInternalException:
        pass