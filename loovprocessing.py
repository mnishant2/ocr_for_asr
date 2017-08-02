#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 18:17:29 2017

@author: nishant
"""
import os,sys
import re
from itertools import chain
import string
import nltk
import subprocess
import json
import enchant
import argparse
import numpy as np
d=enchant.Dict('en_US')
parser=argparse.ArgumentParser()
parser.add_argument('-v','--video',help='path to video file')
parser.add_argument('-g','--ground_truth',help='path to groundtruth')
parser.add_argument('-l','--loov',help='path to build in LOOV directory ending with /')
parser.add_argument('-o','--output',help='output directory for loov output ending with /')
args=parser.parse_args()
gt=args.ground_truth
vid=args.video
LOOV=args.loov
out=args.output
#vid=str(raw_input('enter path to video'))
#gt=str(raw_il=nput('enter path to ground truth'))
final=json.load(open(gt,'rb'))
d = enchant.Dict("en_US")
acc=[]
lent=[]
rec=[]
lenocr=[]
ocrfinal=[[] for i in range(len(final))]
#commandline=[[] for i in range(len(final))]
for i in range(len(final)):
       ss=str(final[i][0])
       endpos=str(final[i][1]-final[i][0])
       op=out+str(i)
       commandline=[LOOV+'./LOOV','-v',vid,'-output',op,'-ss',0,'-endpos',0,'-lang','eng','-print_text']
       commandline[6]=ss
       commandline[8]=endpos  
       print commandline
       subprocess.Popen(commandline).wait()           
          
#       subprocess.check_output(commandline,shell=True,stderr=subprocess.STDOUT)
#       print A.returncode
#       subprocess.Popen(commandline)
#       print(A.communicate)
#       subprocess.Popen(commandline)
#for t in range(len(final)):   
       finocr=[]       
       result=open(out+str(i)+'.OCR','rw')
       line=result.readlines()
       for j in range(len(line)):
            k=line[j].split()
            if (k[0]=='start_box'):
                 n=k[8:]
                 ns=' '.join(n)
                 nst=str(ns)
                 nst=nst.translate(None,string.punctuation)
                 nst=nst.split()
                 for m in range(len(nst)):
                     if (d.check(str(nst[m]))):
                        finocr.append(nst[m])
                                 
       ocrfinal[i]=finocr            
       final[i]=set(final[i][4:])
       finocr=set(finocr)
       lenocr.append(len(finocr))
       lent.append(len(final[i]))
       match=len(set.intersection(final[i],finocr))
       accu=100*(float(match)/float(len(finocr)))
       recall=100*(float(match)/float(len(final[i])))
       rec.append(recall)
       acc.append(accu)
op=out+'list.json'
json.dump(ocrfinal,open(op,'wb'),indent=2,separators=(',',']'))
print np.sum(lent)
print np.sum(lenocr)
print rec
print np.average(rec)
print acc
print np.average(acc)
#       
##from nltk.corpus import words
##subprocess.call()
###    file=open(txt,'r')
##f = open('/home/nishant/l01.txt','r')
##line=f.readlines()
##print final[0]
###print line
##for i in range(len(line)):
##      k=line[i].split()
##      print k
##      if (k[0]=='start_box'):
             
             
            