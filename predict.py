# !/usr/bin/python
# -*- coding: utf-8 -*-
"""predict text from images docstrings.

OCR模型(east+crnn)识别图片中的文字, Input:images, Output:text dictionary

    $python predict.py

Version: 0.1
"""

import os
from east.net.network import East
from east.predict_east import predict_quad
from crnn.net.network import crnn_network
from crnn.predict_crnn import predict_text
from keras.preprocessing import image

east_model_weights_file = "./east/model/east_model_weights.h5"
crnn_model_weights_file = "./crnn/model/crnn_model_weights.hdf5"

root_image = "./east/test/image/"

if __name__ == '__main__':

    # todo east model predict
    east = East()
    east_model = east.east_network()
    east_model.load_weights(east_model_weights_file)
    # east_model.summary()

    # todo crnn model predict
    model, crnn_model = crnn_network()
    crnn_model.load_weights(crnn_model_weights_file)

    for files in os.listdir(root_image):
        img_path = os.path.join(root_image, files)
        im_name = img_path.split('/')[-1][:-4]
        print("path : %s" % img_path)
        print("name : %s" % im_name)

        # fixme height 过长压缩导致无法看清字体
        # fixme 图像h/w比例大于阈值采用裁剪方式识别
        img = image.load_img(img_path).convert('RGB')
        height = img.height
        width = img.width
        scale = height / width

        if scale > 1.5 and height > 2560:
            # todo 重叠部分系数(coefficient) = width/10
            coe = 0.1
            height_s = width * (1 - coe)
            for i in range(int(height / height_s + 1)):
                height_y = i * height_s
                pt1 = (0, min(height_y, height - width))
                pt3 = (width , min(height_y + width, height))
                img_crop = img.crop((pt1[0], pt1[1], pt3[0], pt3[1]))

                im_crop_name = str(im_name) + '_%d' % i
                text_recs_all, text_recs_len, img_all = predict_quad(east_model, img_crop, img_name=im_crop_name)
                if len(text_recs_all) > 0:
                    texts = predict_text(crnn_model, text_recs_all, text_recs_len, img_all, img_name=im_crop_name)
                    for s in range(len(texts)):
                        print("result ：%s" % texts[s])

                    # print("result ：%s" % texts_str)
        else:
            text_recs_all, text_recs_len, img_all = predict_quad(east_model, img, img_name=im_name)
            if len(text_recs_all) > 0:
                texts = predict_text(crnn_model, text_recs_all, text_recs_len, img_all, img_name=im_name)
                for s in range(len(texts)):
                    print("result ：%s" % texts[s])

                # print("result ：%s" % texts_str)
