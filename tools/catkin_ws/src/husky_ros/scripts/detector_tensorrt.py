#!/usr/bin/env python3
import queue
import rospy
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose, BoundingBox2D
from geometry_msgs.msg import Pose2D
from std_msgs.msg import Float32

import sys
sys.path.insert(0, "/home/galirumi/")
sys.path.insert(0, "/home/galirumi/yolor")
from yolor.models.models import *
from yolor.utils.general import non_max_suppression, xyxy2xywh
from yolor.utils.datasets import letterbox
from yolor.utils.plots import plot_one_box
import numpy as np
import torch
import tensorrt as trt
import pycuda.driver as cuda
# import pycuda.autoinit
import threading
import time

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
        self.inference_time_publisher = rospy.Publisher("inference_time", Float32, queue_size=10)
        self.vis_publisher = rospy.Publisher("visual_detections", Image, queue_size=10)
        self.bridge = CvBridge()

        self.half = rospy.get_param('half_precision')
        self.inference_dtype = np.float16 if self.half else np.float32
        if self.half: rospy.loginfo("Using half precision")
        
        #Initialize tensorrt
        cuda.init()
        self.device = cuda.Device(0)
        self.__context = self.device.make_context()
        self._runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING)) 
        self._trt_file = open("/home/galirumi/best_ap_fp16.trt" if self.half else "/home/galirumi/best_ap.trt", "rb")
        self._engine = self._runtime.deserialize_cuda_engine(self._trt_file.read())
        self._context = self._engine.create_execution_context()
        self.input_batch = np.zeros((1,3,640,640), dtype=self.inference_dtype)
        self.output = np.zeros((1,25200,6), dtype=self.inference_dtype) #might need to be reinitialized when used in another thread
        self._d_input = cuda.mem_alloc(1 * self.input_batch.nbytes)
        self._d_output = cuda.mem_alloc(1 * self.output.nbytes)
        self._bindings = [int(self._d_input), int(self._d_output)]
        self._stream = cuda.Stream()
        self.thread_id = threading.get_ident()
        self.__context.pop()

        # self._initialized = False
        
        rospy.loginfo("Warming up tensorrt....")
        rospy.loginfo("Current thread id: %s" % str(threading.get_ident()))
        # self._init_cuda()
        self.model(self.input_batch)
        self.model(self.input_batch)
        self.model(self.input_batch)
        rospy.loginfo("Warmed up tensorrt....")
        rospy.loginfo("Initialization finished. Starting to listen....")
        # context.pop()
            
    def _init_cuda(self):
        rospy.loginfo("Publishing image")
        image_pub = rospy.Publisher("image", Image, queue_size=10)
        img = np.zeros((1200, 1920, 3), dtype=np.uint8)
        rate = rospy.Rate(5)
        for i in range(5):
            rate.sleep()
            if i==2:
                image_pub.publish(self.bridge.cv2_to_imgmsg(img))

        rospy.loginfo("Published to image")
        rospy.loginfo("Waiting for TensorRT initialization...")
        image_pub.unregister()
        rospy.loginfo("Initialized")
        

    def _predict(self, batch): # result gets copied into output
        # transfer input data to device

        self.__context.push()
        rospy.loginfo("Making execution context")
        # self._context = self._engine.create_execution_context()
            # pop_thread = True
        # else:
            # pop_thread = False

        self.output = np.zeros((1,25200,6), dtype=self.inference_dtype) #might need to be reinitialized when used in another thread

        rospy.loginfo("Allocating memory")
        self._d_input = cuda.mem_alloc(1 * self.input_batch.nbytes)
        self._d_output = cuda.mem_alloc(1 * self.output.nbytes)
        
        rospy.loginfo("Creating bindings")
        self._bindings = [int(self._d_input), int(self._d_output)]

        print("Copying batch to cuda device, shape: %s" % str(batch.shape))
        print("Thread ID: %s" % str(threading.get_ident()))
        cuda.memcpy_htod_async(self._d_input, np.ascontiguousarray(batch ), self._stream)

        # execute model
        print("Executing inference")
        self._context.execute_async_v2(self._bindings, self._stream.handle, None)
        
        # transfer predictions back
        print("transfering back")
        res = cuda.memcpy_dtoh_async(self.output, self._d_output, self._stream)

        print("Result of back-transfer : %s\nSynchronizing" % str(res))
        # syncronize threads
        self._stream.synchronize()
        rospy.loginfo("Freeing memory")
        self._d_input.free()
        self._d_output.free()
        # if pop_thread:
        rospy.loginfo("Popping context")
        self.__context.pop()
        return self.output

    def handle_image(self, data):
        t0 = time.time()
        img = self.bridge.imgmsg_to_cv2(data)
        im0 = deepcopy(img)

        img = letterbox(img, new_shape=(640,640), auto=False)[0]
        rospy.loginfo(f"letterbox return np array with shape {img.shape}")
        img = np.ascontiguousarray(img[:, :, ::-1]) #BGR to RGB
        img = img.astype(self.inference_dtype)
        #img = torch.from_numpy(img).to('cuda')
        #img = img.half() 

        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = np.expand_dims(img, 0)

        img = np.transpose(img, (0,3,1,2))
        rospy.loginfo(f"Passing ndarray of size {img.shape} and dtype {img.dtype} to model")
        pred = self.model(img)

        #classes=[0] -> rumex single class, id=0. [0] at the end: batch_size=1, and nms output is list with batch_size elements (nx6 tensors)
        pred = torch.from_numpy(pred)
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

            
        t1 = time.time()
        self.inference_time_publisher.publish(Float32(t1-t0))
        self.publisher.publish(msg)
        self.vis_publisher.publish(self.bridge.cv2_to_imgmsg(im0))

            

if __name__ == "__main__":
    try:
        name_node = DetectorNode()
        # name_node._init_cuda()
        rospy.spin()
    except rospy.ROSInternalException:
        pass
