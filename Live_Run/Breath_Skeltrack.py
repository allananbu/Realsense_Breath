# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 22:32:59 2021

@author: Allan
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 10:04:48 2020

@author: Allan
"""

import numpy as np
import pyrealsense2 as rs
import cv2
import matplotlib.pyplot as plt
import scipy.io
#import scipy.signal as signal
from skeletontracker import skeletontracker


mean_all=[]
#mean_all=np.zeros([1500,1])
frame_all=[]
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
#rs.config.enable_device_from_file(config, "C:/Users/Allan/Desktop/JRF/Realsense/Python/test3.bag", repeat_playback=False)
pipeline = rs.pipeline()
profile=pipeline.start(config)
device = profile.get_device()
depth_sensor = device.query_sensors()[0]
set_laser = 50
depth_sensor.set_option(rs.option.laser_power, set_laser)
temp=np.zeros([2,1])
peaks=[]
valley=[]
FLAG=0
comb=[]
con=[]
i=0
j=0
k=0

#Load filter Coeffcients
a1=scipy.io.loadmat('filter_coef_2.mat')
a=a1['h']
a=np.transpose(a)
a=a[:,0]

#a=np.load('filt_coeff_1.npy')
#window buffer for mean depth signals
x_n=np.zeros([17,])

# Initialize the cubemos api with a valid license key in default_license_dir()
skeletrack = skeletontracker(cloud_tracking_api_key="")
joint_confidence = 0.2

try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        frame_no=frames.get_frame_number() #Get frame number
        frame_time=frames.get_timestamp() #Get timestamp
        frame_all.append(frame_no)
        i=i+1
        if frame_no<20:
            continue
        if frame_no>1500:
            break
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        #Convert color to gray for classifier
        
        depth_color = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        #Detect faces in grayed RGB frame

        #Use coordinates to define ROI
#        for (x,y,w,h) in faces:
##            img=cv2.rectangle(color_image,(x-50,y+250),(x+w,y+h+220),(255,0,0),2)
#             img1=cv2.rectangle(depth_color,(x-50,y+250),(x+w,y+h+220),(0,255,0),2)
        
       
        #Use ROI to find chest in depth frame
        skeletons = skeletrack.track_skeletons(color_image)
        
        skel_2D=skeletons[0]
        joints=skel_2D.joints
        
        x,y=joints[1]

        x=int(x)
        y=int(y)+50
        x1=int(x)-70
        # y=int(y) + 30
        
        y1=y+150
        x2=int(x)+80
        roi_depth=depth_image[y:y1,x1:x2]
        
        
        #Replace zeros in depth with median value 
        roi_meter=roi_depth*0.001
        m=np.median(roi_meter[roi_meter>0])
        roi_meter[roi_meter==0]=m
        
        mean_depth=np.mean(roi_meter)
        mean_all.append(mean_depth)

        
        try:

            x_n=np.roll(x_n,1)
            x_n[0]=mean_depth
            k=k+1
            y=a*x_n
            y=sum(y)
#            
            #y=signal.lfilter(a,1,x_n)
#            y=sum(y)
            con.append(y)
        except:
            continue

        if i>=1:
            diff_mean=np.sign(con[j]-con[j-1])
            j+=1
            temp[1]=diff_mean
        if temp[0]!=temp[1]:
            if temp[1]==-1:
                peaks.append(j-1)
                FLAG=1
                comb.append(10)
            elif temp[1]==1:
                valley.append(j-1)
                v1=0
                FLAG=0
                comb.append(0)
        if len(valley)>=len(peaks):
            comb.append(0)
        else:
            comb.append(10)
        temp[0]=temp[1]
        print("frame number",frame_no)
        
        
        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        
#        
#         Stack both images horizontally
#        images = np.hstack((color_image, depth_color))
        
        plt.figure(2)
        plt.plot(comb)
#        plt.plot(con)
#        plt.plot(mean_all[peaks],'x')
#        plt.plot(mean_all[valley],'o')
        plt.pause(0.0001)

#         Show images
#        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
#        cv2.imshow('RealSense', depth_color)
#        key=cv2.waitKey(1)
#        if key & 0xFF == ord('q') or key == 27:
#            cv2.destroyAllWindows()
#            break
except RuntimeError:
    print("There are no more frames left in the .bag file!")

finally:
    pipeline.stop()

# ts=frame_all[len(frame_all)-1]-frame_all[0]
# ts=ts*0.001
tot_time=frame_no/30
# print("Total time(in sec) is ",ts)
# time=np.linspace(0,tot_time,num=len(mean_all))
# mean_out=[sum(mean_all[i:i+3])/3 for i in range(len(mean_all)-3+1)]
# mean_out=np.array(mean_out)
# mean_out=mean_out[np.logical_not(np.isnan(mean_out))]
# mean_decimate=signal.decimate(mean_all,3)

# for i in np.arange(0,len(mean_all)):
#     if i>=1:
#         diff_mean=np.sign(mean_all[i]-mean_all[i-1])
#         temp[1]=diff_mean
#         if temp[0]!=temp[1]:
#             if temp[1]==-1:
#                 peaks.append(i-1)
#             elif temp[1]==1:
#                 valley.append(i-1)
#         temp[0]=temp[1]
#con=con[13:len(con)]
mean_tot=np.mean(con[13:len(con)])
con=np.array(con)
peaks=np.array(peaks)
valley=np.array(valley)
time1=np.linspace(0,tot_time,num=len(mean_all))
time2=np.linspace(0,tot_time,num=len(con))
time3=np.linspace(0,tot_time,num=len(comb))
plt.figure(3)
plt.plot(time1,mean_all)
#mean_all=np.array(mean_all)
#plt.plot(time1[peaks],mean_all[peaks],'x')
#plt.plot(time1[valley],mean_all[valley],'o')
plt.figure(4)
plt.plot(time2,con)
plt.plot(time2[peaks],con[peaks],'x')
plt.plot(time2[valley],con[valley],'o')
plt.figure(5)
plt.plot(time3,comb)
plt.show()
#plt.ylim([mean_tot+0.004,mean_tot-0.004])

# ##Plot belt respiration
# #ref=pd.read_csv('C:/Users/Allan/Desktop/JRF/Realsense/Python/Reference/test1.csv')
# #time2=ref['Time(s)']
# #resp=ref['Force(N)']
# #resp_norm=(resp-resp.min())/(resp.max()-resp.min())
# #plt.plot(time2,resp_norm)