#coding:utf-8

# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Trains and Evaluates the MNIST network using a feed dictionary."""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import time
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import input_data
import c3d_model
import numpy as np
import matplotlib.pyplot as plt

# Basic model parameters as external flags.
flags = tf.app.flags
gpu_num = 2
flags.DEFINE_integer('batch_size', 10 , 'Batch size.')
FLAGS = flags.FLAGS

def placeholder_inputs(batch_size):
  """Generate placeholder variables to represent the input tensors.
  These placeholders are used as inputs by the rest of the model building
  code and will be fed from the downloaded data in the .run() loop, below.
  Args:
    batch_size: The batch size will be baked into both placeholders.
  Returns:
    images_placeholder: Images placeholder.
    labels_placeholder: Labels placeholder.
  """
  # Note that the shapes of the placeholders match the shapes of the full
  # image and label tensors, except the first dimension is now batch_size
  # rather than the full size of the train or test data sets.
  images_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                         c3d_model.NUM_FRAMES_PER_CLIP,
                                                         c3d_model.CROP_SIZE,
                                                         c3d_model.CROP_SIZE,
                                                         c3d_model.CHANNELS))
  labels_placeholder = tf.placeholder(tf.int64, shape=(batch_size))
  return images_placeholder, labels_placeholder

def _variable_on_cpu(name, shape, initializer):
  #with tf.device('/cpu:%d' % cpu_id):
  with tf.device('/cpu:0'):
    var = tf.get_variable(name, shape, initializer=initializer)
  return var

def _variable_with_weight_decay(name, shape, stddev, wd):
  var = _variable_on_cpu(name, shape, tf.truncated_normal_initializer(stddev=stddev))
  if wd is not None:
    weight_decay = tf.nn.l2_loss(var) * wd
    tf.add_to_collection('losses', weight_decay)
  return var

def run_test():
  model_name = "./models/c3d_ucf_model-2999"
  test_list_file = 'test.list'
  num_test_videos = len(list(open(test_list_file,'r')))
  print("Number of test videos={}".format(num_test_videos))

  # Get the sets of images and labels for training, validation, and
  images_placeholder, labels_placeholder = placeholder_inputs(FLAGS.batch_size * gpu_num)
  with tf.variable_scope('var_name') as var_scope:
    weights = {
            'wc1': _variable_with_weight_decay('wc1', [3, 3, 3, 3, 64], 0.04, 0.00),
            'wc2': _variable_with_weight_decay('wc2', [3, 3, 3, 64, 128], 0.04, 0.00),
            'wc3a': _variable_with_weight_decay('wc3a', [3, 3, 3, 128, 256], 0.04, 0.00),
            'wc3b': _variable_with_weight_decay('wc3b', [3, 3, 3, 256, 256], 0.04, 0.00),
            'wc4a': _variable_with_weight_decay('wc4a', [3, 3, 3, 256, 512], 0.04, 0.00),
            'wc4b': _variable_with_weight_decay('wc4b', [3, 3, 3, 512, 512], 0.04, 0.00),
            'wc5a': _variable_with_weight_decay('wc5a', [3, 3, 3, 512, 512], 0.04, 0.00),
            'wc5b': _variable_with_weight_decay('wc5b', [3, 3, 3, 512, 512], 0.04, 0.00),
            'wd1': _variable_with_weight_decay('wd1', [8192, 4096], 0.04, 0.001),
            'wd2': _variable_with_weight_decay('wd2', [4096, 4096], 0.04, 0.002),
            'out': _variable_with_weight_decay('wout', [4096, c3d_model.NUM_CLASSES], 0.04, 0.005)
            }
    biases = {
            'bc1': _variable_with_weight_decay('bc1', [64], 0.04, 0.0),
            'bc2': _variable_with_weight_decay('bc2', [128], 0.04, 0.0),
            'bc3a': _variable_with_weight_decay('bc3a', [256], 0.04, 0.0),
            'bc3b': _variable_with_weight_decay('bc3b', [256], 0.04, 0.0),
            'bc4a': _variable_with_weight_decay('bc4a', [512], 0.04, 0.0),
            'bc4b': _variable_with_weight_decay('bc4b', [512], 0.04, 0.0),
            'bc5a': _variable_with_weight_decay('bc5a', [512], 0.04, 0.0),
            'bc5b': _variable_with_weight_decay('bc5b', [512], 0.04, 0.0),
            'bd1': _variable_with_weight_decay('bd1', [4096], 0.04, 0.0),
            'bd2': _variable_with_weight_decay('bd2', [4096], 0.04, 0.0),
            'out': _variable_with_weight_decay('bout', [c3d_model.NUM_CLASSES], 0.04, 0.0),
            }
  logits = []
  for gpu_index in range(0, gpu_num):
    with tf.device('/gpu:%d' % gpu_index):

        # 增加部分－－－begin
        d = images_placeholder[gpu_index * FLAGS.batch_size:(gpu_index + 1) * FLAGS.batch_size, :, :, :, :]
        # e = images_placeholder
        # f = e.shape
        # g = d[0].shape
        # res_conv = []
        # # # 卷积
        # # conv_filter = tf.Variable(tf.random_normal([1, 3, 3, 3]))
        # for i in range(10):
        #       img_conv = tf.nn.conv2d(d[i], tf.random_normal([1, 3, 3, 3]), strides=[1, 1, 1, 1], padding='SAME')
        #       img_deconv = tf.nn.conv2d_transpose(img_conv,tf.random_normal([1, 3, 3, 3]),g,strides=[1,1,1,1], padding='SAME')
        #       res_conv.append(img_deconv)
        # #
        # img_data = tf.convert_to_tensor(res_conv, dtype=tf.float32)
        # # 增加部分－－－end
        #
        # img_data = img_data + d
        # img_data = tf.concat([img_data, d], 1)

        logit = c3d_model.inference_c3d(d, 0.5, FLAGS.batch_size, weights, biases)
        logits.append(logit)
  logits = tf.concat(logits,0)
  norm_score = tf.nn.softmax(logits)
  saver = tf.train.Saver()
  sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
  init = tf.global_variables_initializer()
  sess.run(init)
  # Create a saver for writing training checkpoints.
  saver.restore(sess, model_name)
  # And then after everything is built, start the training loop.
  bufsize = 1
  write_file = open("predict_ret.txt", "w+", bufsize)
  next_start_pos = 0

  right_num=0

  all_steps = int((num_test_videos - 1) / (FLAGS.batch_size * gpu_num) + 1)
  total_steps=0
  for step in xrange(all_steps):
    # Fill a feed dictionary with the actual set of images and labels
    # for this particular training step.
    start_time = time.time()
    test_images, test_labels, next_start_pos, _, valid_len = \
            input_data.read_clip_and_label(
                    test_list_file,
                    FLAGS.batch_size * gpu_num,
                    start_pos=next_start_pos
                    )
    predict_score = norm_score.eval(
            session=sess,
            feed_dict={images_placeholder: test_images}
            )
    total_steps += valid_len
    for i in range(0, valid_len):
      true_label = test_labels[i],
      top1_predicted_label = np.argmax(predict_score[i])
      top2_predicted_label = np.argmin(predict_score[i])
      # Write results: true label, class prob for true label, predicted label, class prob for predicted label
      # label_1= 0
      # label_2= 1

      if true_label[0]==top1_predicted_label:
          right_num += 1

      write_file.write('{}, {}, {}, {}\n'.format(
              true_label[0],
              predict_score[i][true_label],
              top1_predicted_label,
              predict_score[i][top1_predicted_label]))

              # label_1,
              # predict_score[i][label_1],
              # label_2,
              # predict_score[i][label_2]))
  write_file.close()
  print(right_num)
  print("all steps:")
  print(total_steps)
  print('test accuracy = ', float(right_num)/total_steps)
  print("done")

def draw_line():
    filename = './predict_ret.txt'
    file = open(filename)
    lines = file.readlines()

    rows = len(lines)  # 文件行数
    datamat = np.zeros((rows, 4))  # 初始化矩阵
    row = 0
    for line in lines:
        line = line.strip().split(',')  # strip()默认移除字符串首尾空格或换行符
        datamat[row, :] = line[:]
        row += 1

    real_cate = datamat
    anomaly_score = real_cate[:,1]
    gt = np.zeros(len(anomaly_score))
    for i in range(2160, 3901):
        gt[i]=1
    for i in range(4860, 6601):
        gt[i]=1


    plt.xlabel("frame number")  # xlabel、ylabel：分别设置X、Y轴的标题文字。
    plt.ylabel("anomaly score")
    plt.title("Shooting047")  # title：设置子图的标题。

    plt.plot(anomaly_score)
    plt.plot(gt)
    plt.savefig('anomaly score --- Explosion008.png', dpi=120, bbox_inches='tight')  # dpi 代表像素
    plt.show()

def main(_):
    run_test()
    # draw_line()

if __name__ == '__main__':
  tf.app.run()
