# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 13:38:03 2017

@author:NM
"""

def wordprocessing(txt,check):
    import os,sys
    import re
    from itertools import chain
    import string
    print sys.argv
    def splice_list(n,slist,nlist):
        start='start'+str(n)
        stop='stop'+str(n)
        indstart=slist.index(start)
        indstop=slist.index(stop)
        nlist.append(slist[indstart+1:indstop])
        return nlist
    def replace_str_index(text,index=0,replacement='',l=1):
        if l==2:
            replacement=replacement.ljust(1)
        
        
        return '%s%s%s'%(text[:index],replacement,text[index+l:])
    
    #import nltmyst
    
    index=0
    file=open(txt,'r')
    myst=file.read()
    if check:
           myst=re.sub(r'\d+','',myst)
    myst=myst.replace(".","")
    myst=myst.replace(":","")
    myst=myst.replace("?","")
    myst=myst.replace("(","")
    myst=myst.replace(")","")
    myst=myst.replace(",","")
    myst=myst.replace("*","")
    myst=myst.replace("</p>","")
    myst=myst.replace("<p/>","")
    myst=myst.replace("<p>","")
    myst=myst.replace("<body>","")
    myst=myst.replace("</body>","")
    myst=myst.replace("</html>","")
    myst=myst.replace("\xef\xbb\xbf","")
    myst=myst.replace("\xc2\xbb","")
    myst=myst.replace("\xe2\x80\x99","'")
    myst=myst.replace("\xe2\x80\xa6","")
    myst=myst.replace("\xe2\x80\xa2","")
    myst=myst.replace("\xe2\x80\x9d","")
    myst=myst.replace("\xe2\x80\x9c","")
    myst=myst.replace("-"," ")
    num=myst.count("<div class=")
    #for i in range(1,num+1):
    ##    k='start'+str(i)
    k= str(1)
    #    print k
    myst=myst.replace('<div class="page">','start'+k+' ')
    myst=myst.replace('</div>','stop'+k+' ')
    
    z=[m.start() for m in re.finditer('start1', myst)]
    y=[m.start() for m in re.finditer('stop1', myst)]
    # print myst[z[29]+5]
    for j in range(0,len(z)):
       l=int(len(str(j)))
    #   print myst[z[j]] 
       myst= replace_str_index(myst,z[j]+5,str(j),l)
       myst= replace_str_index(myst,y[j]+4,str(j),l)
    
    #print num
    dict=myst.split()
    #mysting.replace(myst,)
    words=list(dict)
    #print words
    #print [word.mystip(mysting.punctuation) for word in words]
    for word in words:
        word.strip(string.punctuation)
    #print words  
    final=[[] for i in range(num)]
    for j in range(num):
        final[j]=splice_list(j,words,final[j])
    final=list(chain.from_iterable(final))    
    return final
if __name__ =='__main__':
    import argparse
    check=True
    parser=argparse.ArgumentParser()
    parser.add_argument('-t','--text',help='address to text file')
    parser.add_argument("-n","--num", help="specify whether numbers needed",
                    action="store_true")
    args=parser.parse_args()
    txt=args.text
    if (args.num):
	check=False
    final=wordprocessing(txt,check)
    print final