# -*- coding: utf-8 -*-
# !/usr/bin/python
'''



'''

import time
import os

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

os.environ['OMP_NUM_THREADS'] = '16'

import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard, EarlyStopping

from PIL import Image

from .network import crnn_network


img_h = 32
img_w = 280
batch_size = 64
train_label_file = 'E:/datas/train.txt'
test_label_file = 'E:/datas/test.txt'
root_image_path = 'E:/datas/images/'

save_weigths = 'F:\ocr_detection\project_OCR\crnn\model\crnn_weights-{epoch:02d}.hdf5'
save_tensorboard = 'F:\ocr_detection\project_OCR\crnn\model\log'

pre_train_weigths = 'pre-train_weights_good.hdf5'

class random_uniform_num():
    """
    均匀随机，确保每轮每个只出现一次
    """
    def __init__(self, total):
        self.total = total
        self.range = [i for i in range(total)]
        np.random.shuffle(self.range)
        self.index = 0

    def get(self, batch_size):
        r_n = []
        if (self.index + batch_size > self.total):
            r_n_1 = self.range[self.index:self.total]
            np.random.shuffle(self.range)
            self.index = (self.index + batch_size) - self.total
            r_n_2 = self.range[0:self.index]
            r_n.extend(r_n_1)
            r_n.extend(r_n_2)
        else:
            r_n = self.range[self.index:self.index + batch_size]
            self.index = self.index + batch_size
        return r_n

def read_file(filename):
    res = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for i in lines:
            res.append(i.strip('\r\n'))
    dic = {}
    for i in res:
        p = i.split(' ')
        dic[p[0]] = p[1:]
    return dic

def gen(train_file, batch_size=64, max_label_length=10, img_size=(32, 280)):
    """

    :param train_file:
    :param batch_size:
    :param label_length:
    :param img_size:
    :return:
    """
    image_dic = read_file(train_file)
    image_file = [i for i, j in image_dic.items()]

    x = np.zeros((batch_size, img_size[0], img_size[1], 1), dtype=np.float)
    labels = np.ones([batch_size, max_label_length]) * 10000

    input_length = np.zeros([batch_size, 1])
    label_length = np.zeros([batch_size, 1])

    r_n = random_uniform_num(len(image_file))
    print('图片总量', len(image_file))

    image_file = np.array(image_file)

    while 1:
        shuffle_image = image_file[r_n.get(batch_size)]

        t = time.time()
        #print("time is : {}s".format(t))
        
        for i, j in enumerate(shuffle_image):
            img_path = root_image_path + j
            img1 = Image.open(img_path).convert('L')
            img = np.array(img1, 'f') / 255.0 - 0.5

            x[i] = np.expand_dims(img, axis=2)
            # print('imag:shape',img.shape)

            image_lable = image_dic[j]
            label_length[i] = len(image_lable)

            if (len(image_lable) <= 0):
                print("len<0", j)

            input_length[i] = img_size[1] // 4 + 1
            labels[i, :len(image_lable)] = [int(i) - 1 for i in image_lable]

            #if i==0 or i==1:
                #print("path: %s" % j)
                #print("shape: ", img.shape)
                #new_img = Image.fromarray(img * 255.0)
                #new_img = new_img.convert("RGB")
                #new_img_name = 'bbbbbbbbbbb_%s' % j
                #new_img.save(new_img_name)
                #print("label: ", labels[i]) # (sample, label_len)
                #print("input_length: ", input_length[i]) # (sample, 1)
                #print("label_length: ", label_length[i]) # (sample, 1)

        #print("It takes time : {}s".format(time.time() - t))

        inputs = {'the_input': x,
                  'the_labels': labels,
                  'input_length': input_length,
                  'label_length': label_length,
                  }
        outputs = {'ctc': np.zeros([batch_size])}
        yield (inputs, outputs)

def get_session(gpu_fraction=1.0):
    '''
    Assume that you have 6GB of GPU memory and want to allocate ~2GB
    '''

    num_threads = int(os.environ.get('OMP_NUM_THREADS'))
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

if __name__ == '__main__':
    #K.set_session(get_session(gpu_fraction=1.0))
    #tf.Session()
    model, basemodel = crnn_network()

    if os.path.exists(pre_train_weigths):
        basemodel.load_weights(pre_train_weigths)

    print('-----------begin_fit------------')

    cc1 = gen(train_label_file, batch_size)
    cc2 = gen(test_label_file, batch_size)

    checkpointer = ModelCheckpoint(
        filepath=save_weigths,
        monitor='val_loss',
        verbose=0,
        save_weights_only=False,
        save_best_only=True)

    rlu = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.1,
        patience=1,
        verbose=0,
        mode='auto',
        epsilon=0.0001,
        cooldown=0,
        min_lr=0)

    earlystop = EarlyStopping(patience=10)

    tensorboard = TensorBoard(save_tensorboard,
                              write_graph=True)

    # model.fit_generator(
    #     cc1,
    #     steps_per_epoch=3279606 // batch_size,
    #     epochs=4,
    #     validation_data=cc2,
    #     callbacks=[earlystop, checkpointer, tensorboard],
    #     validation_steps=364400 // batch_size)

    # model.save()
    # model.save_weights()

    yaml_string = basemodel.to_yaml()
    with open('crnn_model.yaml','w') as f:
        f.write(yaml_string)