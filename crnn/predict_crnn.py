# !/usr/bin/python
# -*- coding: utf-8 -*-
"""predict crnn model docstrings.

CRNN模型预测文字文本框中的文字

    $python predict_crnn.py

Version: 0.1
"""

import time
import os
import cv2
import numpy as np
import keras.backend as K
from PIL import Image
from math import *

from .net.network import predict, crnn_network, char

image_root = './test/'
model_path = './model/crnn_weights-08.hdf5'

root_recs = './crnn/test/recs/'

def dumpRotateImage(img, rec):
    """
    根据坐标旋转图片[-90, 90]
    :param img:
    :param rec:
    :return:
    """
    # print("IMG (Xi,Yi) = ", xDim, yDim, rec[0], rec[1], rec[2], rec[3], rec[4], rec[5], rec[6], rec[7])
    xDim, yDim = img.shape[1], img.shape[0]

    # fixme 扩展文字白边 参数为经验值
    xlength = int((rec[4] - rec[0]) * 0.02)
    ylength = int((rec[5] - rec[1]) * 0.05)

    pt1 = (max(1, rec[0] - xlength), max(1, rec[1] - ylength))
    pt2 = (rec[6], rec[7])
    pt3 = (min(rec[4] + xlength, xDim - 2),
           min(yDim - 2, rec[5] + ylength))
    #pt4 = (rec[4], rec[5])

    degree = degrees(atan2(pt2[1] - pt1[1], pt2[0] - pt1[0]))  # fixme 图像倾斜角度

    height, width = img.shape[:2]
    heightNew = int(width * fabs(sin(radians(degree))) +
                    height * fabs(cos(radians(degree))))
    widthNew = int(height * fabs(sin(radians(degree))) +
                   width * fabs(cos(radians(degree))))

    matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)
    matRotation[0, 2] += (widthNew - width) / 2
    matRotation[1, 2] += (heightNew - height) / 2   # fixme 扩展宽高 否则会被裁剪
    imgRotation = cv2.warpAffine(
        img, matRotation, (widthNew, heightNew), borderValue=(255, 255, 255))

    pt1 = list(pt1)
    pt3 = list(pt3)

    # img_rot = Image.fromarray(imgRotation)
    # img_rot.save(root_quad + "xx_rot.jpg")

    [[pt1[0]], [pt1[1]]] = np.dot(matRotation,
                                  np.array([[pt1[0]], [pt1[1]], [1]]))
    [[pt3[0]], [pt3[1]]] = np.dot(matRotation,
                                  np.array([[pt3[0]], [pt3[1]], [1]]))
    ydim, xdim = imgRotation.shape[:2]

    # fixme 通过pixel抠图 返回array
    # imgOut = imgRotation[max(1, int(pt1[1])):min(ydim - 1, int(pt3[1])),
    #                      max(1, int(pt1[0])):min(xdim - 1, int(pt3[0]))]
    # height,width=imgOut.shape[:2]

    # print("NEW IMG (NXi, NYi) = ", ydim, xdim, (max(1, int(pt1[0])), max(1, int(pt1[1])), min(xdim - 1, int(pt3[0])) , min(ydim - 1, int(pt3[1]))))

    # fixme 扩展文字白边 参数为经验值
    xlen = int((pt2[0] - pt1[0]) * 0.03)
    ylen = int((pt3[1] - pt1[1]) * 0.12)

    pt1_N = []
    pt3_N = []
    pt1_N.append(max(1, int(pt1[0]) - xlen))
    pt1_N.append(max(1, int(pt1[1]) - ylen))
    pt3_N.append(min(xdim - 1, int(pt3[0]) + xlen))
    pt3_N.append(min(ydim - 1, int(pt3[1]) + ylen))

    imgRotation = np.uint8(imgRotation)
    img_rot = Image.fromarray(imgRotation)
    img_rec = img_rot.crop((pt1_N[0], pt1_N[1], pt3_N[0], pt3_N[1]))

    return img_rec

def predict_text(model, recs_all, recs_len, img_all, img_name=None):
    """

    :param model:
    :param recs_all:
    :param recs_len:
    :param img_all:
    :param img_name:
    :return:
    """
    texts_str = ''
    texts = []
    img_list = []
    width_list = []
    img_index = 0

    # fixme 当前是前面所有长度的和
    for i in range(len(recs_len)):
        if i > 0:
            recs_len[i] += recs_len[i - 1]

    for i in range(len(recs_all)):

        for j in range(len(recs_len)):
            if i < recs_len[j]:
                img_index = j
                break

        img_rec = dumpRotateImage(img_all[img_index], recs_all[i]).convert('L')

        scale = img_rec.size[1] * 1.0 / 32
        if not scale > 0:
            continue

        w = int(img_rec.size[0] / scale)

        # fixme 像素缩放后小于1pixel
        if not w > 0:
            continue

        img_rec = img_rec.resize((w, 32), Image.BILINEAR)
        width_list.append(w)

        # fixme 增强图像对比度 提高识别
        img_in = np.array(img_rec)
        img_out = np.zeros(img_in.shape, np.uint8)
        cv2.normalize(img_in, img_out, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)

        # fixme 黑白色彩反转 达到黑字白底的目的
        # todo 根据面积比较的反转
        # todo 可以尝试提取图片的前景色
        # black = 0
        # for m in range(32):
        #    for n in range(64 if w >= 64 else w):
        #        if img_out[m, n] < 100 :
        #            black += 1
        # if black > (32*(64 if w >= 64 else w)/2):
        #    img_out = 255 - img_out

        # todo 根据顶点的线条比较反转
        black = 0
        for m in range(32):
            if img_out[m, 0] < 100:
                black += 1
        for n in range(64 if w >= 64 else w):
            if img_out[0, n] < 100:
                black += 1
        if black > (32 + (64 if w >= 64 else w)) // 2:
            img_out = 255 - img_out

        # todo 获取黑色文字进行二值化(效果不佳)
        # for i in range(32):
        #     for j in range(w):
        #         if not (img_out[i, j] < 50):
        #             img_out[i, j] = 255
        #
        # ret, img_out = cv2.threshold(img_out, 180, 255, cv2.THRESH_BINARY)

        img_rec = img_out.astype(np.float32) / 255.0 - 0.5  # img_rec is array
        img_list.append(img_rec)

    width_max = max(width_list)
    X = np.zeros((len(width_list), 32, width_max, 1), dtype=np.float)

    for i in range(len(width_list)):
        img_pad = np.zeros((32, width_max - width_list[i]), np.float32) + 0.5
        img_rec = np.concatenate((img_list[i], img_pad), axis=1)
        X[i] = np.expand_dims(img_rec, axis=2)

        # fixme 保存裁剪后的图像
        if not img_name is None:
            img_out = (img_rec + 0.5) * 255
            img_sa = Image.fromarray(img_out.astype(np.int32))
            img_sa.convert('L').save(root_recs + img_name + '_%d_.jpg' % i)

    y_pred = model.predict(X)

    out = K.get_value(K.ctc_decode(y_pred, input_length=np.ones(y_pred.shape[0]) * y_pred.shape[1])[0][0])

    for i in range(len(out)):
        out_s = u''.join([char[x] for x in out[i] if x != -1])
        # texts_str += (out_s)
        texts.append(out_s)

    # return texts_str
    return texts

if __name__ == '__main__':

    model, basemodel = crnn_network()

    if os.path.exists(model_path):
        basemodel.load_weights(model_path)

    files = sorted(os.listdir(image_root))
    for file in files:
        t = time.time()
        image_path = os.path.join(image_root, file)
        print("&")
        print("=============================================")
        print("ocr image is %s" % image_path)
        out = predict(image_path, basemodel)
        print("------------------------------------")
        print("It takes time : {}s".format(time.time() - t))
        print("result ：%s" % out)
        print("------------------------------------")
        print("&")


