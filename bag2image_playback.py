import numpy as np
import pyrealsense2 as rs
import os
import time
import cv2

images=[]
i = 0
try:
    config = rs.config()
    rs.config.enable_device_from_file(config, "C:/Users/Allan/Desktop/JRF/Realsense/Python/test2.bag", repeat_playback=False)
    pipeline = rs.pipeline()
    profile=pipeline.start(config)
    playback = profile.get_device().as_playback()
    playback.set_real_time(False)

    while True:
        frames = pipeline.wait_for_frames()
        playback.pause()
        depth_frame = frames.get_depth_frame()
        if not depth_frame:
            continue
        depth_image = np.asanyarray(depth_frame.get_data())

        color_image = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        images.append(color_image)

        cv2.imwrite("C:/Users/Allan/Desktop/JRF/Realsense/Python" + str(i) + ".png", color_image)
        i += 1
        playback.resume()
except RuntimeError:
    print("There are no more frames left in the .bag file!")
        
finally:
    pass