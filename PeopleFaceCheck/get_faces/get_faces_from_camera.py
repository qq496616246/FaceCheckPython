#!/usr/bin/python
# -*- coding:utf-8 -*-
# @Time   : 2018/4/16 0016 15:27
# @Author : scw
# @File   : get_faces_from_camera.py
# 此代码实现持续获取摄像头图片信息(还可以用其他的方式，都写了代码)
import cv2
import dlib
import os
import random
# 获取名称
named_faces = input("Input named:")
# 图片数
photos_num = int(input('Input the number of photos'))
output_path = os.path.dirname(os.getcwd())+'/training_material/'+named_faces#输出文件夹

# 设置头像大小
size = 64

# 若无路径，建立路径
if not os.path.exists(output_path):
    os.makedirs(output_path)

# 改变图片亮度与对比度
def relight (img,light=1,bias=0):
    w=img.shape[1]
    h=img.shape[0]
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

# 使用dlib自带的frontal_face_detector作为特征提取器
detector = dlib.get_frontal_face_detector()

# 打开摄像头
camera = cv2.VideoCapture(0)
# 标记
index = 1
while True:
    if( int(index) <= photos_num ):
        print('Being processed picture %s' %index)
        if os.path.exists(output_path+'/'+str(index)+'.jpg'):
            print('photo exists')
            index+=1
            pass
        # 从摄像头读取照片
        success,img = camera.read()
        # 转为灰度图
        gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # 使用detector进行人脸检测
        dets = detector(gray_img,1)
        if not dets:
            print("Can't find faces")
            pass
        for i ,d in enumerate(dets):
            x1=d.top() if d.top()>0 else 0
            y1=d.bottom() if d.bottom()>0 else 0
            x2=d.left() if d.left()>0 else 0
            y2=d.right() if d.right()>0 else 0
            face=img[x1:y1,x2:y2]
            # 调整图片的对比度和亮度
            face=relight(face,random.uniform(0.5,1.5),random.randint(-50,50))
            face=cv2.resize(face,(size,size))
            cv2.imshow('Image',face)
            print("get %s photos"%index)
            cv2.imwrite(output_path+'/'+str(index)+'.jpg',face)
            index+=1
        # 取消注销显示图片
        key = cv2.waitKey(30) & 0xff
        if key == 27:
            break
    else:
        print("Finish!")
        break
