# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 17:00:09 2020

@author: Admin
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
rs.config.enable_device_from_file(config, "C:/Users/Allan/Desktop/JRF/Realsense/Python/test1.bag", repeat_playback=False)


images=[]
i = 0
#Haar Classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
mean_all=[]
frame_all=[]
# Start streaming
profile=pipeline.start(config)
playback = profile.get_device().as_playback()
playback.set_real_time(False)

try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        frame_no=frames.get_frame_number() #Get frame number
        frame_time=frames.get_timestamp() #Get timestamp
        frame_all.append(frame_time)
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        #Convert color to gray for classifier
        gray_bg=cv2.cvtColor(color_image,cv2.COLOR_BGR2GRAY)
        depth_color = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        #Detect faces in grayed RGB frame
        faces=face_cascade.detectMultiScale(gray_bg,1.1,5)
        #Use coordinates to define ROI
        for (x,y,w,h) in faces:
            img=cv2.rectangle(color_image,(x-50,y+250),(x+w,y+h+220),(255,0,0),2)
            img1=cv2.rectangle(depth_color,(x-50,y+250),(x+w,y+h+220),(0,255,0),2)
            roi_bg=color_image[y:y+h,x:x+w]
        #images.append(color_image)
        #Use ROI to find chest in depth frame
        roi_depth=depth_image[y+250:y+h+220,x-50:x+w]
        #roi_all.append(roi_depth)
        #Replace zeros in depth with median value 
        roi_meter=roi_depth*0.001
        m=np.median(roi_depth[roi_depth>0])
        roi_depth[roi_depth==0]=m
        
        mean_depth=np.mean(roi_depth)
        mean_all.append(mean_depth)
        print("frame number",frame_no)
        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        
#        color_all.append(color_frame)
#         Stack both images horizontally
        images = np.hstack((color_image, depth_color))
        #print("frame no",frame_no)

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)
        key=cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
except RuntimeError:
    print("There are no more frames left in the .bag file!")

finally:

    # Stop streaming
    pipeline.stop()

ts=frame_all[len(frame_all)-1]-frame_all[0]
ts=ts*0.001
print("Total time(in sec) is ",ts)
#time=np.linspace(0,int(ts),num=len(frame_all))
mean_out=[sum(mean_all[i:i+4])/4 for i in range(len(mean_all)-4+1)]
mean_out=np.array(mean_out)
mean_out=mean_out[np.logical_not(np.isnan(mean_out))]
mean_out=(mean_out-mean_out.min())/(mean_out.max()-mean_out.min())
time=np.linspace(0,int(ts),num=len(mean_out))
plt.plot(time,mean_all)
plt.show()

#Plot belt respiration
#ref=pd.read_csv('C:/Users/Allan/Desktop/JRF/Realsense/Python/Reference/test1.csv')
#time2=ref['Time(s)']
#resp=ref['Force(N)']
#resp_norm=(resp-resp.min())/(resp.max()-resp.min())
#plt.plot(time2,resp_norm)