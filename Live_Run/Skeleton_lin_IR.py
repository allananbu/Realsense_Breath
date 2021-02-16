# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 21:42:02 2021

@author: Allan
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy.io

joints=scipy.io.loadmat('IR_joints.mat')
jts=joints['Joints']
depth=np.load('IR_image.npy')

#depth=cv2.applyColorMap(cv2.convertScaleAbs(depth_im, alpha=0.03), cv2.COLORMAP_JET)

#depth=cv2.cvtColor(depth_im,cv2.COLOR_BGR2RGB)
ptx_0,pty_0=(jts[0])
ptx_0=int(ptx_0)
pty_0=int(ptx_0)
ptx_1,pty_1=(jts[15])
ptx_1=int(ptx_1)
pty_1=int(pty_1)
ptx_2,pty_2=(jts[16])
ptx_2=int(ptx_2)
pty_2=int(pty_2)
ptx_3,pty_3=(jts[17])
ptx_3=int(ptx_3)
pty_3=int(pty_3)
ptx_4,pty_4=(jts[14])
ptx_4=int(ptx_4)
pty_4=int(pty_4)
ptx_5,pty_5=(jts[1])
ptx_5=int(ptx_5)
pty_5=int(pty_5)
ptx_6,pty_6=(jts[2])
ptx_6=int(ptx_6)
pty_6=int(pty_6)
ptx_7,pty_7=(jts[3])
ptx_7=int(ptx_7)
pty_7=int(pty_7)
ptx_8,pty_8=(jts[5])
ptx_8=int(ptx_8)
pty_8=int(pty_8)
ptx_9,pty_9=(jts[6])
ptx_9=int(ptx_9)
pty_9=int(pty_9)
ptx_10,pty_10=(jts[11])
ptx_10=int(ptx_10)
pty_10=int(pty_10)
ptx_11,pty_11=(jts[8])
ptx_11=int(ptx_11)
pty_11=int(pty_11)

#cv2.circle(depth,(ptx_0,pty_0),6,(255,0,0),4)
#cv2.circle(depth,(ptx_1,pty_1),6,(255,0,0),4)
#cv2.circle(depth,(ptx_2,pty_2),6,(255,0,0),4)
#cv2.circle(depth,(ptx_3,pty_3),6,(255,0,0),4)
#cv2.circle(depth,(ptx_4,pty_4),6,(255,0,0),4)
##cv2.circle(depth,(ptx_5,pty_5),6,(255,0,0),4)
#cv2.circle(depth,(ptx_6,pty_6),6,(255,0,0),4)
#cv2.circle(depth,(ptx_7,pty_7),6,(255,0,0),4)
#cv2.circle(depth,(ptx_8,pty_8),6,(255,0,0),4)
#cv2.circle(depth,(ptx_9,pty_9),6,(255,0,0),4)
#cv2.circle(depth,(ptx_10,pty_10),6,(255,0,0),4)
#cv2.circle(depth,(ptx_11,pty_11),6,(255,0,0),4)


cv2.line(depth,(ptx_4,pty_4),(ptx_2,pty_2),(255,0,0),4)
cv2.line(depth,(ptx_1,pty_1),(ptx_3,pty_3),(255,0,0),4)
cv2.line(depth,(ptx_0,pty_0),(ptx_5-12,pty_5-70),(255,0,0),4)
cv2.line(depth,(ptx_1,pty_1),(ptx_5-12,pty_5-70),(255,0,0),4)
cv2.line(depth,(ptx_4,pty_4),(ptx_5-12,pty_5-70),(255,0,0),4)
cv2.line(depth,(ptx_0,pty_0),(ptx_6,pty_6),(255,0,0),4)
cv2.line(depth,(ptx_0,pty_0),(ptx_8,pty_8),(255,0,0),4)
cv2.line(depth,(ptx_7,pty_7),(ptx_6,pty_6),(255,0,0),4)
cv2.line(depth,(ptx_9,pty_9),(ptx_8,pty_8),(255,0,0),4)
#cv2.line(depth,(ptx_10,pty_10),(ptx_8,pty_8),(255,0,0),4)
cv2.line(depth,(ptx_8,pty_8),(ptx_0,pty_0),(255,0,0),4)
cv2.line(depth,(ptx_11,pty_11),(ptx_0,pty_0),(255,0,0),4)
cv2.line(depth,(ptx_10,pty_10),(ptx_0,pty_0),(255,0,0),4)

cv2.imshow('re',depth)
cv2.imwrite('IR_Skeleton.png',depth)