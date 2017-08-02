#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 20:36:00 2017

@author: nishant
"""

import cv2
import argparse
parser=argparse.ArgumentParser()
parser.add_argument('-v','--video',help='path to video file') 
parser.add_argument('-o','--output',help='path to output video')
parser.add_argument('-b','--blocksize',help='size of the pixel block neighborhood to find threshold')
args=parser.parse_args()
inp=args.video
out=args.output
block=int(args.blocksize)
#from matplotlib import pyplot as plt 
cap=cv2.VideoCapture(inp)
w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps=int(cap.get(cv2.CAP_PROP_FPS))
fourcc=cv2.VideoWriter_fourcc(*'XVID')
#f=open('/home/nishant/kl.avi','rw')
out=cv2.VideoWriter(out,fourcc,fps,(w,h),False)
while(True):
       ret,frame=cap.read()
       if not ret:
              break
       im=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
       th3 = cv2.adaptiveThreshold(im,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,block,2)
       cv2.imshow('th3',th3)
#       print int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#       print int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
       out.write(th3)
       k =cv2.waitKey(1) & 0xff
       if k==32:
              break
#       videowriter=cv2.Videowriter
cap.release()
out.release()
cv2.destroyAllWindows()
       