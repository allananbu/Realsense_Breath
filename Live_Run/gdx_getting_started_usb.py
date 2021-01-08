''' 
The gdx functions are located in a gdx.py file inside a folder named "gdx". In order for 
the import to find the gdx folder, the folder needs to be in the same directory as this python program.

The gdx functions used in a typical program to collect data include:

gdx.open_usb() or gdx.open_ble()
gdx.select_sensors()
gdx.start()
gdx.read()
gdx.stop()
gdx.close() 

Below is a simple starter program that uses these functions to collect data from a Go Direct 
device (or devices) connected via USB. This example will provide you with prompts to select 
the sensors and the sampling period. Try a period of 1000 ms (1 sample/second). 

Tip: Skip the prompts to select the sensors and period by entering arguments in the functions.
Example 1, collect data from sensor 1 at a period of 1000ms using:
gdx.select_sensors([1]), gdx.start(1000)
Example 2, collect data from sensors 1, 2 and 3 at a period of 100ms using:
gdx.select_sensors([1,2,3]), gdx.start(100)

'''

# This code imports the gdx functions. 
from gdx import gdx
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd

gdx = gdx.gdx()

# This code uses the gdx functions to collect data from your Go Direct sensors. 
gdx.open_usb()
gdx.select_sensors([1])
gdx.start(period=100) 
# force=[]
temp=np.zeros([2,1])
peaks=[]
valley=[]
#time.sleep(6.9)
force=np.zeros([500,1])
for i in range(0,500):
    measurements = gdx.read()
    force[i]=measurements
    if i>1:
        diff_mean=np.sign(force[i]-force[i-1])
        temp[1]=diff_mean
        if temp[0]!=temp[1]:
            if temp[1]==-1:
                peaks.append(i-1)
            elif temp[1]==1:
                valley.append(i-1)
        temp[0]=temp[1]
    # plt.plot(force)
    # plt.pause(0.0001)
    if measurements == None: 
        break 
    print(measurements)

gdx.stop()
gdx.close()
time=np.arange(0,len(force)/10,0.1)

dict={'Time':time,'Force':force}
df=pd.DataFrame(dict)
df.to_csv('E:/Research/JRF_VideoBasedVitalSign_breathe_Paper2/Realsense/Realsense_Breath/Live_Runtest/test3.csv')
np.save('belt_data.npy',df)

plt.figure(1)
plt.plot(time,force)
plt.show()