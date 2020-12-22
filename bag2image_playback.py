import numpy as np
import pyrealsense2 as rs
import os
import time
import cv2
import matplotlib.pyplot as plt
images=[]
i = 0
try:
    config = rs.config()
    rs.config.enable_device_from_file(config, "C:/Users/Allan/Desktop/JRF/Realsense/Python/test1.bag", repeat_playback=False)
    pipeline = rs.pipeline()
    profile=pipeline.start(config)
    playback = profile.get_device().as_playback()
    playback.set_real_time(False)

    while True:
        frames = pipeline.wait_for_frames()
       
        color_frame=frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not depth_frame:
            continue
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        depth_color = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        images.append(color_image)
        
        #cv2.imwrite("C:/Users/Allan/Desktop/JRF/Realsense/Python" + str(i) + ".png", depth_image)
                # Show images
        img=np.hstack((color_image,depth_color))
#        plt.imshow(color_image)
#        plt.pause(0.05)
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', img)
        key=cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:
            plt.clo
            break

        i += 1
except RuntimeError:
    print("There are no more frames left in the .bag file!")
        
finally:
    pass