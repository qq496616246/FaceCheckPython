#!/usr/bin/python
# -*- coding:utf-8 -*-
# @Time   : 2018/4/16 0016 15:27
# @Author : scw
# @File   : get_faces_from_photo.py
# 从文件夹source_photos读取图片并保存其中的人像到training_material

import cv2
import dlib
import os
import sys
import random

input_path=os.path.dirname(os.getcwd())+'/source_photos'
output_path=os.path.dirname(os.getcwd())+'/training_material'

size=64

def count_dirs(path):
    count=0
    for dir in os.listdir(path):
        count+=1
    return count

# 改变图片亮度与对比度
def relight (img,light=1,bias=0):
    w=img.shape[1]
    h=img.shape[0]
    #image=[]
    for i in range(0,w):
        for j in range(0,h):
            for c in range(3):
                tmp=int(img[j,i,c]*light+bias)
                if tmp>255:
                    tmp=255
                elif tmp<0:
                    tmp=0
                img[j,i,c]=tmp
    return img

#使用dlib自带的frontal_face_detector作为特征提取器
detector=dlib.get_frontal_face_detector()

def get_faces_from_photos(photo_path,out_path,name):
    img=cv2.imread(photo_path)
    try:
        # 转为灰度图
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 使用detector进行人脸检测
        dets = detector(gray_img, 1)
        if not len(dets):
            print('Can`t get face.')
            pass
        for i, d in enumerate(dets):
             x1 = d.top() if d.top() > 0 else 0
             y1 = d.bottom() if d.bottom() > 0 else 0
             x2 = d.left() if d.left() > 0 else 0
             y2 = d.right() if d.right() > 0 else 0
             face = img[x1:y1, x2:y2]
                # 调整图片的尺寸与对比度
             face = cv2.resize(face, (size, size))
             face = relight(face, random.uniform(0.5, 1.5), random.randint(-50, 50))
             if not os.path.exists(out_path):
                os.makedirs(out_path)
             if i==0:
                cv2.imwrite(out_path + '/' +name, face)
                print('Being processed %s' % name)
             else:
                 cv2.imwrite(out_path + '/' + name+str(i)+'.jpg', face)
                 print('Being processed %s' % name+str(i))

    except:
        print('Error!')
        pass

for dir in os.listdir(input_path):
    dirs = output_path + '/' + dir
    for name in os.listdir(input_path+'/'+dir):
        photo_path=input_path+'/'+dir+'/'+name
        get_faces_from_photos(photo_path,dirs,name)