#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 00:35:30 2019

@author: yangyutu123
"""

import cv2
import os
import re

#fourcc = cv2.cv.CV_FOURCC('M','S','V','C') #Microspoft Video 1

image_folder = 'mapHex'
video_name = 'video.avi'

images_raw = [img for img in os.listdir(image_folder) if img.endswith(".png")]
numbers = [int(re.search(r'\d+', string1).group()) for string1 in images_raw]
# get sorted index
index = sorted(range(len(numbers)), key=lambda k: numbers[k])
images = [images_raw[i] for i in index]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

# 10 fps
video = cv2.VideoWriter(video_name, 0, 10, (width,height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()

import ffmpeg
(
    ffmpeg
    .input('video.avi')
    .hflip()
    .output('output.wmv')
    .run()
)