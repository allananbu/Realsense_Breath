# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 03:31:21 2021

@author: Allan
"""

import multiprocessing
import subprocess
#import playback_bag
#import gdx_getting_started_usb

bots = ['gdx_getting_started_usb','playback_bag']
modules = map(__import__,bots)

def worker(file):
    #your subprocess code
    subprocess.Popen(['screen', './gdx_getting_started_usb.py'])
    subprocess.Popen(['screen', './playback_bag.py'])

if __name__ == '__main__':
    bots = ['gdx_getting_started_usb','playback_bag']

    for i in bots:
        p=multiprocessing.Process(target=worker(i))
        p.start()