# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 00:30:13 2020

@author: Allan
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import json
import matplotlib.pyplot as plt
#from cubemos.core.nativewrapper import CM_TargetComputeDevice

#Load Haar Cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
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
##Post processing filters
#decimation = rs.decimation_filter()
#spatial = rs.spatial_filter()
#temporal = rs.temporal_filter()
#hole_filling = rs.hole_filling_filter()
#depth_to_disparity = rs.disparity_transform(True)
#disparity_to_depth = rs.disparity_transform(False)

#start pipeline based on configuration

profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()

#gets properties of depth sensor and find depth scale
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

# We will be removing the background of objects more than
#  clipping_distance_in_meters meters away
clipping_distance_in_meters = 1 #1 meter
clipping_distance = clipping_distance_in_meters / depth_scale

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

mean_all=[]
roi_all=[]
color_all=[]
depth_all=[]
#Define license directory
#sdk_path = os.environ["CUBEMOS_SKEL_SDK"]
#def default_license_dir():
#    return os.path.join(os.environ["LOCALAPPDATA"],"Cubemos","SkeletonTracking","logs")

try:
    while True:
        
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        frame_no=frames.get_frame_number()
        #Get depth and color frames
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
    
        if not color_frame or not depth_frame:
            continue
        #Post processing of frame
#        frame = decimation.process(depth_frame) 
#        frame = depth_to_disparity.process(depth_frame)
#        frame = spatial.process(frame)
#        frame = temporal.process(frame)
#        filtered_depth_frame = disparity_to_depth.process(frame)
  
        #Frame object to image
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        #Convert 1D depth image to 3D for aligning with color image
        #depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
 
       #Convert uint to meters
        color_all.append(depth_image)
        #Remove background from image
        grey_color=153
        depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
        bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)
        depth_all.append(depth_image)
        color_all.append(color_image)
        #Face detection using Viola Jones algorithm
        #convert backgnd removed img to grayscale
        gray_bg=cv2.cvtColor(color_image,cv2.COLOR_BGR2GRAY)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        faces=face_cascade.detectMultiScale(gray_bg,1.1,5)
        for (x,y,w,h) in faces:
            img=cv2.rectangle(color_image,(x-50,y+250),(x+w,y+h+220),(255,0,0),2)
            img1=cv2.rectangle(depth_colormap,(x-50,y+250),(x+w,y+h+220),(0,255,0),2)
            roi_bg=color_image[y:y+h,x:x+w]
        #roi_bg_gray=cv2.cvtColor(roi_bg,cv2.COLOR_BGR2GRAY)
        roi_depth=depth_image[y+250:y+h+220,x-50:x+w]
        roi_all.append(roi_depth)
        roi_meter=roi_depth*0.001
        m=np.median(roi_depth[roi_depth>0])
        roi_depth[roi_depth==0]=m
        
        mean_depth=np.mean(roi_depth)
        mean_all.append(mean_depth)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
#        #images = np.hstack(( depth_colormap,bg_removed))
        cv2.namedWindow('Align Example', cv2.WINDOW_AUTOSIZE)
        img=np.hstack((color_image,depth_colormap))
        cv2.imshow('Align Example', depth_colormap)
        print("frame number",frame_no)
#        plt.plot(mean_all)
##        plt.pause(0.001)
##        cv2.imwrite("C:/Users/Allan/Desktop/JRF/Realsense/Python/Images/Depth/"+str(i)+".png",depth_image)
##        cv2.imwrite("C:/Users/Allan/Desktop/JRF/Realsense/Python/Images/Color/"+str(i)+".png",color_image) 
#        i = i+1
#        j = j+1
        key = cv2.waitKey(1)
#    # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break

finally:
    pipeline.stop()

ts=np.linspace(0,1/30,len(mean_out))
mean_out=[sum(mean_all[i:i+4])/4 for i in range(len(mean_all)-4+1)]
plt.plot(mean_out)