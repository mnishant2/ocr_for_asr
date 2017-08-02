# -*- coding: utf-8 -*-
"""
Created on Wed Jun 07 15:55:09 2017

@author: Asus
"""

import numpy as np
import cv2 
import io,json
import os
import sys
import argparse
from wordprocessing import wordprocessing
check=True
parser=argparse.ArgumentParser()
parser.add_argument('-v','--video',help='path to video file')
parser.add_argument('-o','--output',help='path to output file')
parser.add_argument('-t','--text',help='address to the text file')
parser.add_argument("-n","--num", help="specify whether numbers needed",
                    action="store_true")
args=parser.parse_args()
txt=args.video
out=args.output
print txt
newtxt=args.text
if (args.num):
       check=False
final=wordprocessing(newtxt,check)

bstart=[0]
bstop=[]
bstart_frame=[0]
bstop_frame=[]
#cv2.VideoCapture.set(cv2.CAP_PROP_FPS,10)
cap=cv2.VideoCapture(txt)
#cap.set(cv2.CAP_PROP_FPS,10)
fgbg = cv2.createBackgroundSubtractorMOG2() 
print cap.grab()
while(True):
    if len(bstop_frame)==len(final):
        del(bstart_frame[-1])
        bstop_frame[-1]=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        break
    ret,frame=cap.read()
    if not ret:
        break
    k = cv2.waitKey(1) & 0xff
    if k == 32:
        while(True):
            key2 = cv2.waitKey(1) & 0xff
            cv2.imshow('im', frame)

            if key2 == 32:
                break
            elif key2 == 101:
                a=int(cap.get(cv2.CAP_PROP_POS_MSEC))
                bstop.append(a)
                c=int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                bstop_frame.append(c)
            elif key2== 115:
                a=int(cap.get(cv2.CAP_PROP_POS_MSEC))
                bstart.append(a)
                c=int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                bstart_frame.append(c)
    elif k == 113:
        break
#        a=int(cap.get(cv2.CAP_PROP_POS_MSEC))
#        bstart.append(a)
#    else:
#        a=int(cap.get(cv2.CAP_PROP_POS_FRAMES))
#        bstop.append(a)
#    ret,frame=cap.read()
#    frame1=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#    frame1 = cv2.GaussianBlur(frame, (5, 5), 0)
#    fgmask = fgbg.apply(frame1)
    cv2.imshow('im',frame)
#    cv2.imshow('im1',frame1)
#    cv2.imshow('im2',fgmask)
#    print int(cap.get(cv2.CAP_PROP_FPS))
#    sum=int(np.average(np.asarray(fgmask)))
#    if sum>8:
#        a=int(cap.get(cv2.CAP_PROP_POS_MSEC))
#        bstart.append(a)
    
    
#print bstart,bstop,bstart_frame,bstop_frame
#print final[0]
bstop_frame.extend([0]*(len(final)-len(bstop_frame)))
bstart_frame.extend([0]*(len(final)-len(bstart_frame)))

for i in range(len(final)):
    final[i].insert(0,bstop[i])
    final[i].insert(0,bstart[i])
    final[i].insert(0,bstop_frame[i])
    final[i].insert(0,bstart_frame[i])
print final
output=out+'.json'
json.dump(final,open(output,'wb'),indent=2,separators=(',',']'))

cap.release()
cv2.destroyAllWindows()  