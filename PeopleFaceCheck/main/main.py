#!/usr/bin/python
# -*- coding:utf-8 -*-
# @Time   : 2018/4/16 0016 15:29
# @Author : scw
# @File   : main.py
# 进行测试
import tensorflow as tf
import cv2
import os
import sys
import dlib
# 添加CNN生成代码的路径
sys.path.append(os.path.dirname(os.getcwd())+'/CNN_training')
import create_CNN_network as CNN

size = 64

# 生成索引字典
def named_dict(path):
	dirs=os.listdir(path)
	name_dict={}
	test={}
	for i in range(len(dirs)):
		test={i:dirs[i]}
		name_dict.update(test)
		test.clear()
	test={None:'no faces'}
	name_dict.update(test)
	return name_dict

name_dict = named_dict(CNN.data_path)

output = CNN.cnnlayer()
predict = tf.argmax(output, 1)

saver = tf.train.Saver()
sess = tf.Session()
saver.restore(sess, tf.train.latest_checkpoint(os.path.dirname(os.getcwd())+'/model'))

# 获取key
def name_key(image):
    res = sess.run(predict, feed_dict={CNN.x: [image / 255.0], CNN.keep_prob_5: 1.0, CNN.keep_prob_75: 1.0})[0]
    return res

detector = dlib.get_frontal_face_detector()

# 读取摄像头参数设为0(0为默认摄像头）
# 更换摄像头：更换参数
# 输入视频：参数改为视频路径
# 输入图像注释此行
camera = cv2.VideoCapture(0)

while True:
    # 若读取图像，注释此行
    _, img = camera.read()
    # 若读取图像，反注释此行，参数输入图像路径
    # img=cv2.imread(path)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dets = detector(gray_image, 1)
    if not len(dets):
        print('Can`t get face.')
        cv2.imshow('img', img)
        key = cv2.waitKey(30) & 0xff
        if key == 27:
            sys.exit(0)

    for i, d in enumerate(dets):
        x1 = d.top() if d.top() > 0 else 0
        y1 = d.bottom() if d.bottom() > 0 else 0
        x2 = d.left() if d.left() > 0 else 0
        y2 = d.right() if d.right() > 0 else 0
        face = img[x1:y1, x2:y2]
        # 调整图片的尺寸
        face = cv2.resize(face, (size, size))
        cv2.rectangle(img, (x2, x1), (y2, y1), (255, 0, 0), 3)
        cv2.putText(img,name_dict[name_key(face)],(x2,x1),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
        cv2.imshow('image', img)
        key = cv2.waitKey(30) & 0xff
        # 当按了esc键之后，进行退出识别
        if key == 27:
            sys.exit(0)
sess.close()