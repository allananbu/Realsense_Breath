# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 10:04:48 2020

@author: Allan
"""

import numpy as np
import pyrealsense2 as rs
import os
import time
import cv2
import matplotlib.pyplot as plt
images=[]
i = 0
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
mean_all=[]
frame_all=[]
try:
    config = rs.config()
    rs.config.enable_device_from_file(config, "C:/Users/Allan/Desktop/JRF/Realsense/Python/test5.bag", repeat_playback=False)
    pipeline = rs.pipeline()
    profile=pipeline.start(config)
    playback = profile.get_device().as_playback()
    playback.set_real_time(False)

    while True:
        frame_present, frames = pipeline.try_wait_for_frames()
        frame_no=frames.get_frame_number()
#        time_stamp=frames.get_timestamp()
        #playback.pause()
        color_frame=frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not depth_frame:
            continue
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        frame_all.append(depth_image)
        depth_color = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        gray_bg=cv2.cvtColor(color_image,cv2.COLOR_BGR2GRAY)
        faces=face_cascade.detectMultiScale(gray_bg,1.1,5)
        for (x,y,w,h) in faces:
            img=cv2.rectangle(color_image,(x-50,y+250),(x+w,y+h+220),(255,0,0),2)
            img1=cv2.rectangle(depth_color,(x-50,y+250),(x+w,y+h+220),(0,255,0),2)
            roi_bg=color_image[y:y+h,x:x+w]
        images.append(color_image)
        
        roi_depth=depth_image[y+250:y+h+220,x-50:x+w]
        #roi_all.append(roi_depth)
        roi_meter=roi_depth*0.001
        m=np.median(roi_depth[roi_depth>0])
        roi_depth[roi_depth==0]=m
        
        mean_depth=np.mean(roi_depth)
        mean_all.append(mean_depth)
        print("frame number",frame_no)
#        print("time stamp",time_stamp)
        #cv2.imwrite("C:/Users/Allan/Desktop/JRF/Realsense/Python" + str(i) + ".png", depth_image)
                # Show images
        img=np.hstack((img,img1))
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', img)
        key=cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
        if not frame_present:
            break
        i += 1
        #playback.resume()
except RuntimeError:
    print("There are no more frames left in the .bag file!")
        
finally:
    pass

mean_out=[sum(mean_all[i:i+4])/4 for i in range(len(mean_all)-4+1)]
plt.plot(mean_out)