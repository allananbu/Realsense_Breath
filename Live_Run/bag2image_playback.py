import numpy as np
import pyrealsense2 as rs
from scipy import signal
import cv2
import matplotlib.pyplot as plt
images=[]
i = 0
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
frame_all=[]
mean_all=[]
try:
    config = rs.config()
    rs.config.enable_device_from_file(config, "E:/Research/JRF_VideoBasedVitalSign_breathe_Paper2/Realsense/Realsense_Breath/Live_Run/bag4.bag", repeat_playback=False)
    pipeline = rs.pipeline()
    profile=pipeline.start(config)
    playback = profile.get_device().as_playback()
    playback.set_real_time(False)
    while True:
        frames = pipeline.wait_for_frames()
        frame_no=frames.get_frame_number()
        frame_all.append(frame_no)
        color_frame=frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not depth_frame or not color_frame:
            continue
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        # if frame_no<20:
        #     continue
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
        
        #Replace zeros in depth with median value 
        roi_meter=roi_depth*0.001
        m=np.median(roi_meter[roi_meter>0])
        roi_meter[roi_meter==0]=m
#        
        mean_depth=np.mean(roi_meter)
        mean_all.append(mean_depth)
        print("frame number",frame_no)
        #cv2.imwrite("C:/Users/Allan/Desktop/JRF/Realsense/Python" + str(i) + ".png", depth_image)
                # Show images
        img=np.hstack((color_image,depth_color))
#        plt.imshow(color_image)
#        plt.pause(0.05)
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', img)
        key=cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break

        i += 1
except RuntimeError:
    print("There are no more frames left in the .bag file!")
        
finally:
    pass

ts=frame_all[len(frame_all)-1]-frame_all[0]
ts=ts*0.001
tot_time=frame_no/30
print("Total time(in sec) is ",ts)
#time=np.linspace(0,int(ts),num=len(frame_all))
mean_out=[sum(mean_all[i:i+3])/3 for i in range(len(mean_all)-3+1)]
mean_out=np.array(mean_out)
mean_out=mean_out[np.logical_not(np.isnan(mean_out))]
mean_decimate=signal.decimate(mean_all,3)
time=np.linspace(0,tot_time,num=len(mean_decimate))

resp=np.load('belt_data.npy',allow_pickle=True)
force=resp[:,1]
time1=resp[:,0]
force.tolist()
f1=force.tolist()
t1=time1.tolist()
fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.plot(time,mean_decimate, color=color)
# ax2 = ax1.twinx()
color = 'tab:blue'
ax2.plot(t1,f1)