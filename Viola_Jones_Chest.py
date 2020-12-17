# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 00:30:13 2020

@author: Allan
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import json
#from cubemos.core.nativewrapper import CM_TargetComputeDevice

jsonObj = json.load(open("HighResHighAccuracyPreset.json"))
json_string= str(jsonObj).replace("'", '\"')

#initialize pipeline
pipeline = rs.pipeline()
#Set configurations for stream
config = rs.config()
#rs.config.enable_device_from_file(config, "C:/Users/Allan/Desktop/JRF/Realsense/Python/test2.bag", repeat_playback=False)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)
#playback = profile.get_device().as_playback()
#playback.set_real_time(False)
dev = profile.get_device()
advnc_mode = rs.rs400_advanced_mode(dev)
advnc_mode.load_json(json_string)
#Post processing filters
decimation = rs.decimation_filter()
spatial = rs.spatial_filter()
temporal = rs.temporal_filter()
hole_filling = rs.hole_filling_filter()
depth_to_disparity = rs.disparity_transform(True)
disparity_to_depth = rs.disparity_transform(False)

#start pipeline based on configuration

profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()

#gets properties of depth sensor
depth_sensor = profile.get_device().first_depth_sensor()
#Get preset range
#preset_range = depth_sensor.get_option_range(rs.option.visual_preset)
#
##Set Preset - choose from 1-6 levels
#for i in range(int(preset_range.max)):
#    visualpreset=depth_sensor.get_option_value_description(rs.option.visual_preset,i)
#    if visualpreset == "High Accuracy":
#        depth_sensor.set_option(rs.option.visual_preset, i)
#        
#if depth_sensor.supports(rs.option.depth_units):
#    depth_sensor.set_option(rs.option.depth_units,0.00025)
#    depth_scale = depth_sensor.get_depth_scale()

#Align depth and color
align_to = rs.stream.depth
align = rs.align(align_to)
i=1
j=1

#Define license directory
sdk_path = os.environ["CUBEMOS_SKEL_SDK"]
def default_license_dir():
    return os.path.join(os.environ["LOCALAPPDATA"],"Cubemos","SkeletonTracking","logs")

try:
    while True:
        
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        #Get depth and color frames
        aligned_color_frame = aligned_frames.get_color_frame() 
        depth_frame = aligned_frames.get_depth_frame()
    
        if not aligned_color_frame or not depth_frame:
            continue
        #Post processing of frame
        frame = decimation.process(depth_frame) 
        frame = depth_to_disparity.process(depth_frame)
        frame = spatial.process(frame)
        frame = temporal.process(frame)
        filtered_depth_frame = disparity_to_depth.process(frame)
  
        #Frame object to image
        depth_image = np.asanyarray(filtered_depth_frame.get_data())
        color_image = np.asanyarray(aligned_color_frame.get_data())
        #Convert 1D depth image to 3D for aligning with color image
        depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
 
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        images = np.hstack((color_image, depth_colormap))
        
#
        
        cv2.namedWindow('Align Example', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Align Example', images)
#        cv2.imwrite("C:/Users/Allan/Desktop/JRF/Realsense/Python/Images/Depth/"+str(i)+".png",depth_image)
#        cv2.imwrite("C:/Users/Allan/Desktop/JRF/Realsense/Python/Images/Color/"+str(i)+".png",color_image) 
        i = i+1
        j = j+1
        key = cv2.waitKey(1)
    # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
finally:
    pipeline.stop()