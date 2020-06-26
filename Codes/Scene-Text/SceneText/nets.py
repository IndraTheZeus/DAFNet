#!/usr/bin/python
#
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Network definitions for multiplane image (MPI) prediction networks.
"""
from __future__ import division
import tensorflow as tf
import tensorflow.compat.v2.nn as slim

def binning_net(inputs,dc_image, num_outputs, ngf=64, vscope='net', reuse_weights=False):
  """Network definition for multiplane image (MPI) inference.

  Args:
    inputs: stack of input images [batch, height, width, input_channels]
    num_outputs: number of output channels
    ngf: number of features for the first conv layer
    vscope: variable scope
    reuse_weights: whether to reuse weights (for weight sharing)
  Returns:
    pred: network output at the same spatial resolution as the inputs.
  """
  in_shape = inputs.get_shape().as_list()
  assert in_shape[1] == 256
  assert in_shape[2] == 256
  assert in_shape[3] == 198
  with tf.compat.v1.variable_scope(vscope, reuse=reuse_weights):
     
   # with tf.compat.v2.nn as slim:
      w1_1 = tf.compat.v1.get_variable('w1_1',shape=(3,3,in_shape[3],ngf),initializer=tf.contrib.layers.xavier_initializer(),trainable=True)
      b1_1 = tf.compat.v1.get_variable('b1_1',shape=(ngf),initializer=tf.zeros_initializer(),trainable=True)
      cnv1_1 = slim.relu(slim.bias_add(slim.conv2d(inputs,w1_1, strides=1,padding="SAME"),b1_1))
      cnv1_1 = slim.dropout(cnv1_1,rate=0.0)
      w1_2 = tf.compat.v1.get_variable('w1_2',shape=(3,3,ngf,ngf*2),initializer=tf.contrib.layers.xavier_initializer(),trainable=True)
      b1_2 = tf.compat.v1.get_variable('b1_2',shape=(ngf*2),initializer=tf.zeros_initializer(),trainable=True)
      cnv1_2 = slim.relu(slim.bias_add(slim.conv2d(cnv1_1,w1_2, strides=2,padding="SAME"),b1_2))  ##Stride orig was 2
      cnv1_2 = slim.dropout(cnv1_2,rate=0.0) 
      
      w2_1 = tf.compat.v1.get_variable('w2_1',shape=(3,3,ngf*2,ngf*2),initializer=tf.contrib.layers.xavier_initializer(),trainable=True)
      b2_1 = tf.compat.v1.get_variable('b2_1',shape=(ngf*2),initializer=tf.zeros_initializer(),trainable=True)
      cnv2_1 = slim.relu(tf.nn.bias_add(slim.conv2d(cnv1_2,w2_1, strides=1,padding="SAME"),b2_1))
      cnv2_1 = slim.dropout(cnv2_1,rate=0.0)
      w2_2 = tf.compat.v1.get_variable('w2_2',shape=(3,3,ngf*2,ngf*4),initializer=tf.contrib.layers.xavier_initializer(),trainable=True)
      b2_2 = tf.compat.v1.get_variable('b2_2',shape=(ngf*4),initializer=tf.zeros_initializer(),trainable=True)
      cnv2_2 = slim.relu(slim.bias_add(slim.conv2d(cnv2_1,w2_2, strides=2,padding="SAME"),b2_2))  ##Stride orig was 2
      cnv2_2 = slim.dropout(cnv2_2,rate=0.0)

      w3_1 = tf.compat.v1.get_variable('w3_1',shape=(3,3,ngf*4,ngf*4),initializer=tf.contrib.layers.xavier_initializer(),trainable=True)
      b3_1 = tf.compat.v1.get_variable('b3_1',shape=(ngf*4),initializer=tf.zeros_initializer(),trainable=True)
      cnv3_1 = slim.relu(slim.bias_add(slim.conv2d(cnv2_2,w3_1, strides=1,padding="SAME"),b3_1))
      cnv3_1 = slim.dropout(cnv3_1,rate=0.0)
      w3_2 = tf.compat.v1.get_variable('w3_2',shape=(7,7,ngf*4,ngf*4),initializer=tf.contrib.layers.xavier_initializer(),trainable=True)
      b3_2 = tf.compat.v1.get_variable('b3_2',shape=(ngf*4),initializer=tf.zeros_initializer(),trainable=True)
      cnv3_2 = slim.relu(slim.bias_add(slim.conv2d(cnv3_1,w3_2, strides=1,padding="SAME"),b3_2))
      cnv3_2 = slim.dropout(cnv3_2,rate=0.0)
      w3_3 = tf.compat.v1.get_variable('w3_3',shape=(3,3,ngf*4,ngf*8),initializer=tf.contrib.layers.xavier_initializer(),trainable=True)
      b3_3 = tf.compat.v1.get_variable('b3_3',shape=(ngf*8),initializer=tf.zeros_initializer(),trainable=True)
      cnv3_3 = slim.relu(slim.bias_add(slim.conv2d(cnv3_2,w3_3, strides=2,padding="SAME"),b3_3))  ##Stride was 2
      cnv3_3 = slim.dropout(cnv3_3,rate=0.0)

      w4_1 = tf.compat.v1.get_variable('w4_1',shape=(3,3,ngf*8,ngf*8),initializer=tf.contrib.layers.xavier_initializer(),trainable=True)
      b4_1 = tf.compat.v1.get_variable('b4_1',shape=(ngf*8),initializer=tf.zeros_initializer(),trainable=True)
      cnv4_1 = slim.relu(slim.bias_add(slim.conv2d(cnv3_3,w4_1, strides=1,padding="SAME",dilations=2),b4_1))
      cnv4_1 = slim.dropout(cnv4_1,rate=0.0)
      w4_2 = tf.compat.v1.get_variable('w4_2',shape=(7,7,ngf*8,ngf*8),initializer=tf.contrib.layers.xavier_initializer(),trainable=True)
      b4_2 = tf.compat.v1.get_variable('b4_2',shape=(ngf*8),initializer=tf.zeros_initializer(),trainable=True)
      cnv4_2 = slim.relu(slim.bias_add(slim.conv2d(cnv4_1,w4_2, strides=1, padding="SAME",dilations=2),b4_2))
      cnv4_2 = slim.dropout(cnv4_2,rate=0.0)
      w4_3 = tf.compat.v1.get_variable('w4_3',shape=(3,3,ngf*8,ngf*8),initializer=tf.contrib.layers.xavier_initializer(),trainable=True)
      b4_3 = tf.compat.v1.get_variable('b4_3',shape=(ngf*8),initializer=tf.zeros_initializer(),trainable=True)
      cnv4_3 = slim.relu(slim.bias_add(slim.conv2d(cnv4_2,w4_3, strides=1, padding="SAME",dilations=2),b4_3))
      cnv4_3 = slim.dropout(cnv4_3,rate=0.0)

      # Adding skips
      skip = tf.concat([cnv4_3, cnv3_3], axis=3)
      w6_1 = tf.compat.v1.get_variable('w6_1',shape=(3,3,ngf*4,ngf*16),initializer=tf.contrib.layers.xavier_initializer(),trainable=True)
      b6_1 = tf.compat.v1.get_variable('b6_1',shape=(ngf*4),initializer=tf.zeros_initializer(),trainable=True)
      cnv6_1 = slim.relu(slim.bias_add(slim.conv2d_transpose(skip,w6_1, strides=2,padding="SAME",output_shape=(in_shape[0],64,64,ngf*4)),b6_1)) ##Stride was 2
      cnv6_1 = slim.dropout(cnv6_1,rate=0.0)
      w6_2 = tf.compat.v1.get_variable('w6_2',shape=(7,7,ngf*4,ngf*4),initializer=tf.contrib.layers.xavier_initializer(),trainable=True)
      b6_2 = tf.compat.v1.get_variable('b6_2',shape=(ngf*4),initializer=tf.zeros_initializer(),trainable=True)
      cnv6_2 = slim.relu(slim.bias_add(slim.conv2d(cnv6_1,w6_2, strides=1,padding="SAME"),b6_2))
      cnv6_2 = slim.dropout(cnv6_2,rate=0.0)
      w6_3 = tf.compat.v1.get_variable('w6_3',shape=(3,3,ngf*4,ngf*4),initializer=tf.contrib.layers.xavier_initializer(),trainable=True)
      b6_3 = tf.compat.v1.get_variable('b6_3',shape=(ngf*4),initializer=tf.zeros_initializer(),trainable=True)
      cnv6_3 = slim.relu(slim.bias_add(slim.conv2d(cnv6_2,w6_3, strides=1,padding="SAME"),b6_3))
      cnv6_3 = slim.dropout(cnv6_3,rate=0.0)

      skip = tf.concat([cnv6_3, cnv2_2], axis=3)
      w7_1 = tf.compat.v1.get_variable('w7_1',shape=(3,3,ngf*2,ngf*8),initializer=tf.contrib.layers.xavier_initializer(),trainable=True)
      b7_1 = tf.compat.v1.get_variable('b7_1',shape=(ngf*2),initializer=tf.zeros_initializer(),trainable=True)
      cnv7_1 = slim.relu(slim.bias_add(slim.conv2d_transpose(skip,w7_1, strides=2,padding="SAME",output_shape=(in_shape[0],128,128,ngf*2)),b7_1)) ##Stride was 2
      cnv7_1 = slim.dropout(cnv7_1,rate=0.0)
      w7_2 = tf.compat.v1.get_variable('w7_2',shape=(3,3,ngf*2,ngf*2),initializer=tf.contrib.layers.xavier_initializer(),trainable=True)
      b7_2 = tf.compat.v1.get_variable('b7_2',shape=(ngf*2),initializer=tf.zeros_initializer(),trainable=True)
      cnv7_2 = slim.relu(slim.bias_add(slim.conv2d(cnv7_1,w7_2, strides=1,padding="SAME"),b7_2))
      cnv7_2 = slim.dropout(cnv7_2,rate=0.0)

      skip = tf.concat([cnv7_2, cnv1_2], axis=3)
      w8_1 = tf.compat.v1.get_variable('w8_1',shape=(3,3,ngf,ngf*4),initializer=tf.contrib.layers.xavier_initializer(),trainable=True)
      b8_1 = tf.compat.v1.get_variable('b8_1',shape=(ngf),initializer=tf.zeros_initializer(),trainable=True)
      cnv8_1 = slim.relu(slim.bias_add(slim.conv2d_transpose(skip,w8_1, strides=2,padding="SAME",output_shape=(in_shape[0],256,256,ngf)),b8_1)) ##Stride was 2
      cnv8_1 = slim.dropout(cnv8_1,rate=0.0)
      w8_2 = tf.compat.v1.get_variable('w8_2',shape=(3,3,ngf,ngf),initializer=tf.contrib.layers.xavier_initializer(),trainable=True)
      b8_2 = tf.compat.v1.get_variable('b8_2',shape=(ngf),initializer=tf.zeros_initializer(),trainable=True)
      cnv8_2 = slim.relu(slim.bias_add(slim.conv2d(cnv8_1,w8_2, strides=1,padding="SAME"),b8_2))
      cnv8_2 = slim.dropout(cnv8_2,rate=0.0)

      feat = cnv8_2
      wf = tf.compat.v1.get_variable('wF',shape=(7,7,ngf,num_outputs),initializer=tf.contrib.layers.xavier_initializer(),trainable=True)
      bf = tf.compat.v1.get_variable('bf',shape=(num_outputs),initializer=tf.zeros_initializer(),trainable=True)
      pred = slim.tanh(slim.bias_add(slim.conv2d(feat,wf,strides=1,padding="SAME"),bf))
      #pred = slim.bias_add(slim.conv2d(feat,wf,strides=1,padding="SAME"),bf)
      #FC Classification
      #pred = tf.compat.v1.layers.dense(cnv8_2, 1, activation=slim.relu, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer(),bias_initializer=tf.zeros_initializer(), kernel_regularizer=None,
                  #bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
                  #bias_constraint=None, trainable=True, name=None, reuse=None)
      
      #pred1 = tf.compat.v1.layers.dense(pred, 1, activation=slim.tanh, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer(),bias_initializer=tf.zeros_initializer(), kernel_regularizer=None,
                  #bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
                  #bias_constraint=None, trainable=True, name=None, reuse=None)
      
      
      return pred
      #wf1 = tf.compat.v1.get_variable('wf1',shape=(3,3,num_outputs,1),initializer=tf.contrib.layers.xavier_initializer(),trainable=True)
      #bf1 = tf.compat.v1.get_variable('bf1',shape=(1),initializer=tf.zeros_initializer(),trainable=True)
      #pred2 = slim.tanh(slim.bias_add(slim.conv2d(
      #    pred,wf1,
      #    strides=1,
      #    padding="SAME"),bf1))
      #flat_pred = tf.reshape(pred2,[in_shape[0],-1])
      #flat_pred = tf.concat([flat_pred,tf.ones([in_shape[0],1],dtype=tf.float32)],axis=1)
      #wfc = tf.compat.v1.get_variable('wfc',shape=(in_shape[1]*in_shape[2],in_shape[1]*in_shape[2]),initializer=tf.contrib.layers.xavier_initializer(),trainable=True)
      #bfc = tf.compat.v1.get_variable('bfc',shape=(1,in_shape[1]*in_shape[2]),initializer=tf.zeros_initializer(),trainable=True)
      #FC = tf.concat([wfc,bfc],axis=0)
      #out_FC = slim.tanh(tf.matmul(flat_pred,FC))
      #out_pred = tf.reshape(out_FC,[in_shape[0],in_shape[1],in_shape[2]])
      
