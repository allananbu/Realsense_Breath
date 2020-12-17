# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 03:00:31 2020

@author: Allan
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import json
import time
import h5py

jsonObj = json.load(open("HighResHighAccuracyPreset.json"))
json_string= str(jsonObj).replace("'", '\"')

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

#can also load width and height from json file
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

cfg = pipeline.start(config)
dev = cfg.get_device()
advnc_mode = rs.rs400_advanced_mode(dev)
advnc_mode.load_json(json_string)

size=(640,480)
#hf=h5py.File('data.h5','w')
try:
    while True:
        #wait for depth and color frames
        frames = pipeline.wait_for_frames()
        depth_f=frames.get_depth_frame()
        color_f=frames.get_color_frame()
        if not depth_f or not color_f:
            continue
        
        #convert frame data to array
        depth_image=np.asanyarray(depth_f.get_data())
        color_image=np.asanyarray(color_f.get_data())
        
        #black and white
#        depth_color=cv2.convertScaleAbs(depth_image,alpha=0.03)
        
        #apply colormap
        depth_color=cv2.applyColorMap(cv2.convertScaleAbs(depth_image,alpha=0.1),cv2.COLORMAP_JET)        
        #STACK color and depth images
        images=np.hstack((color_image,depth_color))
        #hf.create_dataset('dataset',depth_image)
        
        #write frame into video
        #result.write(depth_image)
        #show images
        cv2.namedWindow('Realsense',cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Realsense',images)
        key=cv2.waitKey(1)
        if key & 0xFF==ord('q') or key==27:
            cv2.destroyAllWindows()
            break
        hf.close()
finally:
    pipeline.stop()
    
        

        