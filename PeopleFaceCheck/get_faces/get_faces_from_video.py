#!/usr/bin/python
# -*- coding:utf-8 -*-
# @Time   : 2018/4/16 0016 15:28
# @Author : scw
# @File   : get_faces_from_video.py

import cv2
import dlib
import os
import sys
import random

number=input("Input number of faces:")
output_dir=os.path.dirname(os.getcwd())+'/training_material'
input_dir=os.path.dirname(os.getcwd())+'/source_videos'

size=64

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def count_dirs(path):
    count=0
    for dir in os.listdir(path):
        count+=1
    return count

#改变图片亮度与对比度
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

index=1

def get_faces_from_videoes(ranges,path,out_path):
    camera=cv2.VideoCapture(path)#获取视频文件
    global index
    while True:
        try:
            if (index < ranges):
                print('Being processed picture %s' % index)
                if os.path.exists(out_path + '/' + str(index) + '.jpg'):
                    index+=1
                    print('picture %s is already exit'%index)
                    pass
                # 从摄像头读取照片
                success, img = camera.read()
                if not camera.read():
                    break
                # 转为灰度图
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # 使用detector进行人脸检测
                dets = detector(gray_img, 1)
                if not len(dets):
                    # print('Can`t get face.')
                    cv2.imshow('img', img)
                    key = cv2.waitKey(1) & 0xff
                    if key == 27:
                        sys.exit(0)
                    pass
                for i, d in enumerate(dets):
                    x1 = d.top() if d.top() > 0 else 0
                    y1 = d.bottom() if d.bottom() > 0 else 0
                    x2 = d.left() if d.left() > 0 else 0
                    y2 = d.right() if d.right() > 0 else 0
                    face = img[x1:y1, x2:y2]
                    # 调整图片的尺寸
                    face = cv2.resize(face, (size, size))
                    face = relight(face, random.uniform(0.5, 1.5), random.randint(-50, 50))
                    cv2.rectangle(img, (x2, x1), (y2, y1), (255, 0, 0), 3)
                    cv2.imshow('image', img)
                    if not os.path.exists(out_path):
                        os.makedirs(out_path)
                    cv2.imwrite(out_path + '/' + str(index) + '.jpg', face)
                    key = cv2.waitKey(1) & 0xff
                    if key == 27:
                        sys.exit(0)
                    index += 1
            else:
                break
        except:
            break

for dir in os.listdir(input_dir):
    dirs = output_dir + '/' + dir
    index=1
    for x in os.listdir(input_dir+'/'+dir):
        video_path=input_dir+'/'+dir+'/'+x
        get_faces_from_videoes(number,video_path,dirs)