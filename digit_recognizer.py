#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 17:29:44 2019

@author: sarfaraz
"""
#import digit recognition using scikit learn
import IM 
import cv2
import numpy as np
import digits_ann_2 as LNN

import matplotlib.pyplot as pt
import pandas as pd 


def inside(r1, r2):
    x1,y1,w1,h1 = r1
    x2,y2,w2,h2 = r2
    if (x1 > x2) and (y1 > y2) and (x1+w1 < x2+w2) and (y1+h1 < y2 + h2):
        return True
    else:
        return False


def wrap_digit(rect):
   
    (x, y, w, h) = rect
    padding = 5
    hcenter = x + w/2
    vcenter = y + h/2
    if (h > w):
        w = h
        x = hcenter - (w/2)
    else:
        h = w
        y = vcenter - (h/2)
    return (int(x-padding), int(y-padding), int(w+padding), int(h+padding))
#ann, test_data =KNN.train(KNN.create_ANN(56), 20000, 20)
net3 =LNN.create_ANN(56)
ann3, test_data =LNN.train(net3, 2000,40)
font = cv2.FONT_HERSHEY_SIMPLEX


img = cv2.imread("final_digit.jpg", 1)
img = img.astype('uint8')
bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
bw = cv2.GaussianBlur(bw, (7,7), 0)
ret, thbw = cv2.threshold(bw, 80, 255, cv2.THRESH_BINARY_INV)
thbw = cv2.erode(thbw, np.ones((2,2), np.uint8), iterations = 2)
image, cntrs, hier = cv2.findContours(thbw.copy(), cv2.RETR_TREE,
cv2.CHAIN_APPROX_SIMPLE)


rectangles = []
for c in cntrs:
    r = x,y,w,h = cv2.boundingRect(c)
    a = cv2.contourArea(c)
    b = (img.shape[0]-3) * (img.shape[1] - 3)
    is_inside = False
    for q in rectangles:
        if inside(r, q):
            is_inside = True
            break
    if not is_inside:
        if not a == b:
            rectangles.append(r)
          
for r in rectangles:
    x,y,w,h = wrap_digit(r)
    cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 2)# drawing rectanlges around the digit i.e.,contours
    roi = thbw[y:y+h, x:x+w]
    IM.display(roi)
    print('inside this ')
    img2 = cv2.resize(roi.copy(),(28,28))
    IM.display(img2)
    digit_class = int(LNN.predict(ann3, roi.copy())[0])
    #digit_class = clf.predict(roi)
#    print(digit_class)
    font = cv2.FONT_HERSHEY_SIMPLEX 
    #s = 'number found!'
    cv2.putText(img, str(digit_class), (x, y-1), font, 2, (200,255,155), 2, cv2.LINE_AA)

    #cv2.putText(img, 'number found', (200, ), font, 100, (255, 0, 0))##error is here , got it.
    #k = cv2.putText(img,s , (20,20), font, 100, (255, 0, 0))##error is here , got it.
    #cv2.putText(img, ,(x, y), font, 1, (200,255,155), 2, cv2.LINE_AA)
    #IM.display(k)
    #IM.display(img)
        
IM.display(thbw)
IM.display(img)
