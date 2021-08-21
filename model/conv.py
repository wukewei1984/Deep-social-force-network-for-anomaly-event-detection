#coding:utf-8

import cv2
import pylab
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
g2=tf.Graph()

def conv_2d_process(img):
    fil = np.array([[ -1,-1, 0],                        #这个是设置的滤波，也就是卷积核
                    [ -1, 0, 1],
                    [  0, 1, 1]])
    res = cv2.filter2D(img,-1,fil)                      #使用opencv的卷积函数

    # plt.imshow(res)                                     #显示卷积后的图片
    # plt.imsave("res.jpg",res)
    # plt.show()
    return res


#卷积，将图片大小变为原来的一半
def fun_conv_2d(img_data):
    with tf.device('/gpu:1'):
        with g2.as_default():
            temp_data = img_data
            img_shape = img_data.shape

            # 读取图片，矩阵化，转换为张量
            img_data = tf.convert_to_tensor(img_data, dtype=tf.float32)

            # 将张量转化为4维
            img_data = tf.reshape(img_data, shape=[1, img_shape[0], img_shape[1], 3])

            # 权重（也叫filter、过滤器）
            weights = tf.Variable(tf.random_normal(shape=[2, 2, 3, 3], dtype=tf.float32))

            # 卷积
            conv = tf.nn.conv2d(img_data, weights, strides=[1, 2, 2, 1], padding='SAME')

            img_data = tf.reshape(conv, shape=[1, img_shape[0]/2, img_shape[1]/2, 3])
            img_data = tf.nn.relu(img_data)

            #反卷积
            deconv = tf.nn.conv2d_transpose(conv,weights,output_shape=[1, img_shape[0], img_shape[1], 3], strides=[1,2,2,1],padding="SAME")

            #将４维转为３维
            new_img_data = tf.reshape(deconv, shape=[img_shape[0], img_shape[1], 3])
            with tf.Session(graph=g2) as sess2:
                sess2.run(tf.global_variables_initializer())
                res = sess2.run(new_img_data)
                plt.imshow(res)
                plt.show()

            c = res.shape
    return res




def fun_deconv_2d(img_data):
    return img_data