#!/usr/bin/python
# -*- coding:utf-8 -*-
# @Time   : 2018/4/16 0016 15:26
# @Author : scw
# @File   : get_faces_from_Internet.py

import re
import requests
import cv2
import dlib
import sys
import os

#定义索引
index=1

ranges=int(input("Input number:"))

#从网络上获取人像
def dowmloadPic(html, keyword,input_dir,output_dir):
    global index
    pic_url = re.findall('"objURL":"(.*?)",', html, re.S)
    print('找到关键词:' + keyword + '的图片，现在开始下载图片...')
    for each in pic_url:
        print('正在下载第' + str(index) + '张图片，图片地址:' + str(each))
        try:
            pic = requests.get(each, timeout=10)
        except requests.exceptions.ConnectionError:
            print('【错误】当前图片无法下载')
            continue

        dir = os.path.dirname(os.getcwd())+'/DownLoad_picture/' + keyword + '.jpg'#存放到临时文件夹
        fp = open(dir, 'wb')
        fp.write(pic.content)
        fp.close()
        try:
            # 从文件读取图片
            img = cv2.imread(dir)
            while(os.path.exists(output_dir + '/' + str(index) + '.jpg')):
                index+=1#避免重复
            if(index>ranges):
                break
            # 转为灰度图片
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # 使用detector进行人脸检测 dets为返回的结果
            dets = detector(gray_img, 1)
            if not len(dets):
                print("找不到人脸!")
                os.remove(dir)
                pass
            # 使用enumerate 函数遍历序列中的元素以及它们的下标
            # 下标i为人脸序号
            # left：人脸左边距离图片左边界的距离 ；right：人脸右边距离图片左边界的距离
            # top：人脸上边距离图片上边界的距离 ；bottom：人脸下边距离图片上边界的距离
            for i, d in enumerate(dets):
                x1 = d.top() if d.top() > 0 else 0
                y1 = d.bottom() if d.bottom() > 0 else 0
                x2 = d.left() if d.left() > 0 else 0
                y2 = d.right() if d.right() > 0 else 0
                face = img[x1:y1, x2:y2]
                # 调整图片的尺寸为64x64
                face = cv2.resize(face, (size, size))
                # 保存图片
                cv2.imwrite(output_dir + '/' + str(index) + '.jpg', face)
                index += 1
                os.remove(dir)
        except:
            pass
        if (index > ranges):
            break

input_dir = os.path.dirname(os.getcwd())+'/DownLoad_picture'

size = 64

if not os.path.exists(input_dir):
    os.makedirs(input_dir)


#使用dlib自带的frontal_face_detector作为我们的特征提取器
detector = dlib.get_frontal_face_detector()

if __name__ == '__main__':
    word = input("Input key word: ")
    picture_dir=os.path.dirname(os.getcwd())+'/training_material/'+str(word)
    if not os.path.exists(picture_dir):
        os.makedirs(picture_dir)
    for pn in range(0,1000,20):
        url = 'http://image.baidu.com/search/flip?tn=baiduimage&ie=utf-8&word='+word+'&pn='+str(pn)+'&gsm=3c&ct=&ic=0&lm=-1&width=0&height=0'
        if (index > ranges):
            break
        result = requests.get(url)
        dowmloadPic(result.text, word,input_dir,picture_dir)