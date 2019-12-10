# !/usr/bin/python
# -*- coding: utf-8 -*-
"""predict east model docstrings.

EAST模型预测图像中文字文本框

    $python predict_east.py

Version: 0.1
"""

import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import cv2
import copy
from PIL import Image, ImageDraw
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

from .data.label import point_inside_of_quad
from .net.network import East
from .data.preprocess import resize_image
from .net.nms import nms
from .net import cfg

root_image = './east/test/image/'
root_temp = './east/test/temp/'
root_predict = './east/test/predict/'
root_quad = './east/test/quad/'

def deal_image_red(im, h, w):
    img = copy.deepcopy(im)
    # img = im.copy()
    # red
    for i in range(h):
        for j in range(w):
            if img[i,j,0] > 170:
                if img[i,j,2] > 90 or img[i,j,1] > 90:
                    img[i,j,0] = 0

    img_cov = 255 - img[:,:,0]
    ret, ret_bin = cv2.threshold(img_cov, 180, 255, cv2.THRESH_BINARY)

    # dilate 开闭操作
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    img_dil = cv2.dilate(ret_bin, element)

    # erode
    # element = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    # img_ero = cv2.erode(img_dil, element)

    # 2 dilate
    # element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # img_dil = cv2.dilate(img_ero, element)

    img = cv2.cvtColor(img_dil, cv2.COLOR_GRAY2RGB)

    return img

def deal_image_black(im, h, w):
    img = copy.deepcopy(im)
    # img = im.copy()
    # black
    for i in range(h):
        for j in range(w):
            if not (img[i, j, 0] < 90 and img[i, j, 1] < 90 and img[i, j, 2] < 90):
                img[i, j, 0] = 255
                img[i, j, 1] = 255
                img[i, j, 2] = 255

    img_cov = img[:, :, 0]
    ret, ret_bin = cv2.threshold(img_cov, 180, 255, cv2.THRESH_BINARY)

    # dilate 开闭操作
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    img_dil = cv2.dilate(ret_bin, element)

    # erode
    # element = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    # img_ero = cv2.erode(img_dil, element)

    # 2 dilate
    # element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # img_dil = cv2.dilate(img_ero, element)

    img = cv2.cvtColor(img_dil, cv2.COLOR_GRAY2RGB)

    return img

def deal_image_wht(im, h, w):
    img = copy.deepcopy(im)
    # img = im.copy()
    # black
    for i in range(h):
        for j in range(w):
            if not (img[i, j, 0] > 190 and img[i, j, 1] > 190 and img[i, j, 2] > 190):
                img[i, j, 0] = 0
                img[i, j, 1] = 0
                img[i, j, 2] = 0

    img_cov = 255 - img[:, :, 0]
    ret, ret_bin = cv2.threshold(img_cov, 180, 255, cv2.THRESH_BINARY)

    # dilate 开闭操作
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    img_dil = cv2.dilate(ret_bin, element)

    # erode
    # element = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    # img_ero = cv2.erode(img_dil, element)

    # 2 dilate
    # element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # img_dil = cv2.dilate(img_ero, element)

    img = cv2.cvtColor(img_dil, cv2.COLOR_GRAY2RGB)

    return img

def sigmoid(x):
    """`y = 1 / (1 + exp(-x))`"""
    return 1 / (1 + np.exp(-x))

def cut_text_line(geo, scale_ratio_w, scale_ratio_h, im_array, img_name, s):
    geo /= [scale_ratio_w, scale_ratio_h]
    p_min = np.amin(geo, axis=0)
    p_max = np.amax(geo, axis=0)
    min_xy = p_min.astype(int)
    max_xy = p_max.astype(int) + 2
    sub_im_arr = im_array[min_xy[1]:max_xy[1], min_xy[0]:max_xy[0], :].copy()
    for m in range(min_xy[1], max_xy[1]):
        for n in range(min_xy[0], max_xy[0]):
            if not point_inside_of_quad(n, m, geo, p_min, p_max):
                sub_im_arr[m - min_xy[1], n - min_xy[0], :] = 255
    sub_im = image.array_to_img(sub_im_arr, scale=False)
    sub_im.save(img_name + '_subim%d.jpg' % s)

def predict_quad(model, img, pixel_threshold=cfg.pixel_threshold, quiet=False, img_name=None):
    """

    :param model:
    :param img:
    :param pixel_threshold:
    :param quiet:
    :param img_name:
    :return:
    """

    d_wight, d_height = resize_image(img, cfg.max_predict_img_size)
    # fixme BILINEAR 双线性算法复杂度高 性能消耗大 但是图像损坏较小 影响后面识别效果
    img = img.resize((d_wight, d_height), Image.BILINEAR).convert('RGB')
    img = image.img_to_array(img)

    # fixme 文字叠加时 使用了简单的颜色进行分离 不需要可省略
    img_red = deal_image_red(img, d_height, d_wight)
    img_blk = deal_image_black(img, d_height, d_wight)
    img_wht = deal_image_wht(img, d_height, d_wight)

    # fixme 预处理后保存图像
    num_img = 4
    img_all = np.zeros((num_img, d_height, d_wight, 3))
    img_all[0] = img
    img_all[1] = img_red
    img_all[2] = img_blk
    img_all[3] = img_wht

    img_ori = preprocess_input(img, mode='tf')  # suit tf tensor
    img_red = preprocess_input(img_red, mode='tf')  # suit tf tensor
    img_blk = preprocess_input(img_blk, mode='tf')  # suit tf tensor
    img_wht = preprocess_input(img_wht, mode='tf')

    # todo 图像预处理分成多张image批量处理
    x = np.zeros((num_img, d_height, d_wight, 3))
    x[0] = img_ori
    x[1] = img_red
    x[2] = img_blk
    x[3] = img_wht

    # (sample, h, w, channels)
    y_pred = model.predict(x)

    text_recs_all = []
    text_recs_len = []
    for n in range(num_img):
        # (sample, rows, cols, 7_points_pred)
        y = y_pred[n]
        y[:, :, :3] = sigmoid(y[:, :, :3])
        cond = np.greater_equal(y[:, :, 0], pixel_threshold)
        activation_pixels = np.where(cond)  # fixme 返回元祖tuple类型 a[0]保存了纵坐标 a[1]保存横坐标
        quad_scores, quad_after_nms = nms(y, activation_pixels)

        text_recs = []
        x[n] = np.uint8(x[n])
        with image.array_to_img(img_all[n]) as im:     # Image.fromarray(x[n]) error ?
            im_array = x[n]

            # fixme 注意：拿去CRNN识别的是缩放后的图像
            scale_ratio_w = 1
            scale_ratio_h = 1

            quad_im = im.copy()
            draw = ImageDraw.Draw(im)
            for i, j in zip(activation_pixels[0], activation_pixels[1]):
                px = (j + 0.5) * cfg.pixel_size
                py = (i + 0.5) * cfg.pixel_size
                line_width, line_color = 1, 'blue'
                if y[i, j, 1] >= cfg.side_vertex_pixel_threshold:
                    if y[i, j, 2] < cfg.trunc_threshold:
                        line_width, line_color = 2, 'yellow'
                    elif y[i, j, 2] >= 1 - cfg.trunc_threshold:
                        line_width, line_color = 2, 'green'
                draw.line([(px - 0.5 * cfg.pixel_size, py - 0.5 * cfg.pixel_size),
                           (px + 0.5 * cfg.pixel_size, py - 0.5 * cfg.pixel_size),
                           (px + 0.5 * cfg.pixel_size, py + 0.5 * cfg.pixel_size),
                           (px - 0.5 * cfg.pixel_size, py + 0.5 * cfg.pixel_size),
                           (px - 0.5 * cfg.pixel_size, py - 0.5 * cfg.pixel_size)],
                          width=line_width, fill=line_color)

            if not img_name is None:
                im.save(root_temp + img_name + '_%d_.jpg' % n)

            quad_draw = ImageDraw.Draw(quad_im)
            for score, geo, s in zip(quad_scores, quad_after_nms,
                                     range(len(quad_scores))):
                if np.amin(score) > 0:
                    quad_draw.line([tuple(geo[0]),
                                    tuple(geo[1]),
                                    tuple(geo[2]),
                                    tuple(geo[3]),
                                    tuple(geo[0])], width=2, fill='blue')

                    if cfg.predict_cut_text_line:
                        cut_text_line(geo, scale_ratio_w, scale_ratio_h, im_array,
                                      img_name, s)

                    rescaled_geo = geo / [scale_ratio_w, scale_ratio_h]
                    text_rec = np.reshape(rescaled_geo, (8,)).tolist()
                    text_recs.append(text_rec)
                elif not quiet:
                    print('quad invalid with vertex num less then 4.')

            if not img_name is None:
                quad_im.save(root_predict + img_name + '_%d_.jpg' % n)

        for t in range(len(text_recs)):
            text_recs_all.append(text_recs[t])

        text_recs_len.append(len(text_recs))

    return text_recs_all, text_recs_len, img_all

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-p',
                        default='demo/z08.jpg',
                        help='image path')
    parser.add_argument('--threshold', '-t',
                        default=cfg.pixel_threshold,
                        help='pixel activation threshold')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    # img_path___ = args.path
    threshold = float(args.threshold)
    #print(img_path, threshold)

    east = East()
    east_detect = east.east_network()
    east_detect.load_weights(cfg.saved_model_weights_file_path)

    east_detect.summary()

    for files in os.listdir(root_image):
        img_path = os.path.join(root_image, files)
        print("image path %s" % img_path)
        predict_quad(east_detect, img_path, threshold)
