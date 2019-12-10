# !/usr/bin/python
# -*- coding: utf-8 -*-
"""crnn network.

   LSTM + CTC

Version: 0.1
"""

import os,sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parentdir)

from PIL import Image

import keras.backend as K
import numpy as np

from keras.layers import Flatten, BatchNormalization, Permute, TimeDistributed, Dense, Bidirectional, GRU, LSTM
from keras.layers import Input, Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Lambda
from keras.models import Model
from keras.optimizers import SGD, Adam, Adadelta, RMSprop

img_h = 32
char_file = 'E:\datas\char_std_5990.txt'

char = ''
with open(char_file, encoding='utf-8') as f:
    for ch in f.readlines():
        ch = ch.strip('\r\n')
        char = char + ch

# caffe_ocr中把0作为blank，但是tf 的CTC  the last class is reserved to the blank label.
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/util/ctc/ctc_loss_calculator.h
char = char[1:] + ' '
n_class = len(char)

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def crnn_network(height=img_h, nclass=n_class):
    '''
    cnn + rnn + ctc model !!!
    :param height:
    :param nclass:
    :return:
    '''
    input = Input(shape=(height, None, 1), name='the_input')
    # CNN
    m = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', name='conv1')(input)
    m = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1')(m)
    m = Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same', name='conv2')(m)
    m = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool2')(m)
    m = Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same', name='conv3')(m)
    m = Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same', name='conv4')(m)

    m = ZeroPadding2D(padding=(0, 1))(m)
    m = MaxPooling2D(pool_size=(2, 2), strides=(2, 1), padding='valid', name='pool3')(m)

    m = Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same', name='conv5')(m)
    m = BatchNormalization(axis=3)(m)
    m = Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same', name='conv6')(m)
    m = BatchNormalization(axis=3)(m)
    m = ZeroPadding2D(padding=(0, 1))(m)
    m = MaxPooling2D(pool_size=(2, 2), strides=(2, 1), padding='valid', name='pool4')(m)
    m = Conv2D(512, kernel_size=(2, 2), activation='relu', padding='valid', name='conv7')(m)

    # m的输出维度为(h, w, c) -> (1, w/4, 512) 转换 (w, b, c) = (seq_len, batch, input_size)
    m = Permute((2, 1, 3), name='permute')(m)
    m = TimeDistributed(Flatten(), name='timedistrib')(m)

    # RNN
    m = Bidirectional(LSTM(256, return_sequences=True), name='blstm1')(m)
    m = Dense(256, name='blstm1_out', activation='linear')(m)
    m = Bidirectional(LSTM(256, return_sequences=True), name='blstm2')(m)
    y_pred = Dense(nclass, name='blstm2_out', activation='softmax')(m)

    basemodel = Model(inputs=input, outputs=y_pred)
    #basemodel.summary()

    labels = Input(name='the_labels', shape=[None], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

    model = Model(inputs=[input, labels, input_length, label_length], outputs=[loss_out])

    # sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
    # model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd, metrics=['accuracy'])
    # model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer="adadelta", metrics=['accuracy'])
    # model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adam', metrics=['accuracy'])
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='rmsprop', metrics=['accuracy'])

    # model.summary()

    return model, basemodel

def predict(img_path, model):
    """
    输入图片，输出keras模型的识别结果
    :param img_path:
    :return:
    """
    img = Image.open(img_path)
    img = img.convert('L')

    scale = img.size[1] * 1.0 / 32
    w = int(img.size[0] / scale)
    img = img.resize((w, 32), Image.BILINEAR)
    img = np.array(img).astype(np.float32) / 255.0 - 0.5
    X = img.reshape((32, w, 1))
    X = np.array([X])

    y_pred = model.predict(X)

    y_pred = y_pred[:, :, :]

    out = K.get_value(K.ctc_decode(y_pred, input_length=np.ones(y_pred.shape[0]) * y_pred.shape[1], )[0][0])[:, :]
    out_s = u''.join([char[x] for x in out[0]])

    return out_s
