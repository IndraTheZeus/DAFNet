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
"""Functions for learning multiplane images (MPIs).
"""

from __future__ import division
#import os
import time
import tensorflow as tf
#import tensorflow_addons as tfa
#from third_party.vgg import build_vgg19
from SceneText.FilteringResNet import binningResNet
#from SceneText.nets import binning_net
tf.compat.v1.disable_eager_execution()

#import math
#import numpy as np
#import sys
#from tensorflow import flags
FLAGS = tf.compat.v1.app.flags.FLAGS

def compute_f_scores(real,fake):
      labeled_real = tf.image.connected_components(real)
      labeled_fake = tf.image.connected_components(fake)
      noRealComps = tf.math.reduce_max(labeled_real)
      noFakeComps = tf.math.reduce_max(labeled_fake)
      acc_precision = tf.constant(0)
      acc_recall = tf.constant(0)
      gt_count = tf.constant(1)
      cond = lambda gt_count,acc_precision:gt_count<=noRealComps
      body = lambda gt_count,acc_precision:[gt_count+1,tf.divide(tf.reduce_sum(tf.where(tf.equal(real,gt_count),fake,tf.zeros_like(real))),tf.reduce_sum(tf.cast(tf.equal(real,gt_count),dtype=tf.float32)))]
      gt_count,acc_precision = tf.while_loop(cond, body, [gt_count,acc_precision])
      avg_precision = tf.divide(acc_precision,noRealComps)
      det_count = tf.constant(1)
      cond = lambda det_count,acc_recall:det_count<=noFakeComps
      body = lambda det_count,acc_recall:[det_count+1,tf.divide(tf.reduce_sum(tf.where(tf.equal(fake,det_count),real,tf.zeros_like(real))),tf.reduce_sum(tf.cast(tf.equal(fake,det_count),dtype=tf.float32)))]
      det_count,acc_recall = tf.while_loop(cond, body, [det_count,acc_recall])
      avg_recall = tf.divide(acc_recall,noFakeComps)
      return (2*acc_recall*acc_precision)/(acc_recall+acc_precision),avg_precision,avg_recall
      
class BIN_MODEL(object):
  """Class definition for MPI learning module.
  """

  def __init__(self):
    pass

  def get_binarized_image(self,binned_images,dc_image=None):
    """.
    Args:

    Returns:
      
    """
    with tf.name_scope('prediction'):
        bin_pred = binningResNet(binned_images, None,1)
        #bin_pred = binning_net(binned_images,None, 3)
    return bin_pred


      
  def build_test_graph(self,inputs, which_loss='IOU'):
    with tf.name_scope('input_data'):
      gt_image = self.preprocess_image(inputs.gt_image)
      input_image = self.preprocess_image(inputs.scene_image)
      #dc_image = inputs.dc_image
      gt_image = tf.cast(gt_image>0.25,dtype=tf.float32)
      #gt_image = tf.concat([gt_image,gt_image,gt_image],axis=-1)
      image_name = inputs.image_name
    with tf.name_scope('inference'):
      binned_images = tf.cast(inputs.binned_images,dtype=tf.float32)
      pred= self.get_binarized_image(binned_images)
      #pred= self.get_binarized_image(input_image)
      pred = tf.divide(tf.add(pred,1.0),2.0)

    with tf.name_scope('loss'):
      if which_loss == 'FScore':
        def compute_error(real, fake):
          acc_f,avg_prec,avg_rec = compute_f_scores(real,fake)
          avg_f = (2*avg_prec*avg_rec)/(avg_prec+avg_rec)
          loss = -acc_f
          return loss,avg_prec,avg_rec,avg_f
        total_loss,precision,recall,f_score = compute_error(gt_image,pred)
        print_op = tf.print("Evaluate: ", precision, recall,f_score, output_stream=tf.compat.v1.logging.info)
        with tf.control_dependencies([print_op]):
          total_loss = tf.Print(total_loss, [total_loss], message="Loss = ")
      if which_loss == 'IOU':
        def compute_error(real, fake):
          intersection = tf.reduce_sum(tf.where(tf.equal(real,1.0),fake,tf.zeros_like(real)))
          union = tf.reduce_sum(tf.where(tf.equal(real,1.0),real,fake))
          return tf.subtract(1.0,tf.divide(intersection,union))
        total_loss= compute_error(gt_image,pred)
        total_loss = tf.Print(total_loss, [total_loss], message="Loss = ")
      if which_loss == 'CrossEntropy':
        def compute_error(real, fake):
          logh = tf.math.negative(tf.reduce_sum(tf.where(tf.equal(real,1.0),tf.math.log(0.00001+fake),tf.zeros_like(fake))))
          log1_h = tf.math.negative(tf.reduce_sum(tf.where(tf.equal(real,1.0),tf.zeros_like(fake),tf.math.log(1.00001 - fake))))
          return tf.divide(tf.math.add(logh,log1_h),tf.cast(tf.size(fake),dtype=tf.float32))
        total_loss= compute_error(gt_image,pred)
        total_loss = tf.Print(total_loss, [total_loss], message="Loss = ")
    with tf.name_scope('write'):
       pred_image = self.deprocess_image(tf.reshape(pred,[FLAGS.image_height,FLAGS.image_width,-1]))
       image_string = tf.image.encode_png(pred_image, compression=0)
       folder = tf.constant("ICDAR_TEST_RESULTS\\")
       extension = tf.constant(".png")
       write_op = tf.io.write_file(tf.strings.join([folder,image_name[0],extension]), image_string, name=None)
    # Summaries
    tf.compat.v1.summary.scalar('TEST_total_loss', total_loss)
    # Output image
    tf.compat.v1.summary.image('TEST_output_image', self.deprocess_image(pred))
    tf.compat.v1.summary.image('TEST_input_image', input_image)
    tf.compat.v1.summary.image('TEST_gt_image', self.deprocess_image(gt_image))
    return write_op
      
  def test(self,write_op,checkpoint_dir,num_test_imgs):
    parameter_count = tf.reduce_sum(
        [tf.reduce_prod(tf.shape(v)) for v in tf.compat.v1.trainable_variables()])
    global_step=tf.compat.v1.train.get_or_create_global_step()
    incr_global_step = tf.compat.v1.assign(global_step, global_step + 1)
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sv = tf.compat.v1.train.MonitoredTrainingSession(
        checkpoint_dir=checkpoint_dir, summary_dir=checkpoint_dir,save_summaries_steps=1,save_checkpoint_steps=None,save_checkpoint_secs=None,config=config)   
    with sv as sess:
      tf.compat.v1.logging.info('Trainable variables: ')
      for var in tf.trainable_variables():
        tf.compat.v1.logging.info(var.name)
      tf.compat.v1.logging.info('parameter_count = %d' % sess.run(parameter_count))
      for step in range(1, num_test_imgs):
        fetches = {
            'pred': write_op,
            'global_step': global_step,
            'incr_global_step': incr_global_step,
        }
        results = sess.run(fetches)
       

  def build_train_graph(self,
                        inputs,bins,
                        which_loss='IOU',
                        learning_rate=0.0002,
                        beta1=0.9,writer=None):
    """Construct the training computation graph.

    Args:
      inputs: dictionary of tensors (see 'input_data' below) needed for training
      min_depth: minimum depth for the plane sweep volume (PSV) and MPI planes
      max_depth: maximum depth for the PSV and MPI planes
      num_psv_planes: number of PSV planes for network input
      num_mpi_planes: number of MPI planes to infer
      which_color_pred: how to predict the color at each MPI plane
      which_loss: which loss function to use (vgg or pixel)
      learning_rate: learning rate
      beta1: hyperparameter for ADAM
      vgg_model_file: path to VGG weights (required when VGG loss is used)
    Returns:
      A train_op to be used for training.
    """

    with tf.name_scope('input_data'):
      gt_image = self.preprocess_image(inputs.gt_image)
      input_image = self.preprocess_image(inputs.scene_image)
      orig_image = self.deprocess_image(inputs.OrigImage)
      #dc_image = inputs.dc_image
      #gt_image = tf.cast(tf.cast(gt_image,dtype=tf.bool),dtype=tf.float32)
      gt_image = tf.cast(gt_image>0.25,dtype=tf.float32)
      #gt_image = tf.concat([gt_image,gt_image,gt_image],axis=-1)

    with tf.name_scope('inference'):
      binned_images = tf.cast(inputs.binned_images,dtype=tf.float32)
      assert len(binned_images.get_shape().as_list()) == 4
      assert binned_images.get_shape().as_list()[3] == 194
      #for i in range(0,binned_images.get_shape().as_list()[3],3):
        #tf.compat.v1.summary.image('binned_images'+str(i),self.deprocess_image(binned_images[:,:,:,i:i+3]))
      pred= self.get_binarized_image(binned_images)
      #pred= self.get_binarized_image(input_image)
      pred = tf.divide(tf.add(pred,1.0),2.0)
      #pred = tf.where(dc_image,tf.cast(tf.zeros_like(pred),dtype=tf.float32),pred)
      #sub_pred = tf.divide(tf.add(sub_pred,1.0),2.0)
      #pred = tf.expand_dims(pred,-1)

    with tf.name_scope('loss'):
      if which_loss == 'FScore':
        def compute_error(real, fake):
          acc_f,avg_prec,avg_rec = compute_f_scores(real,fake)
          avg_f = (2*avg_prec*avg_rec)/(avg_prec+avg_rec)
          loss = -acc_f
          return loss,avg_prec,avg_rec,avg_f
        total_loss,precision,recall,f_score = compute_error(gt_image,pred)
        print_op = tf.print("Evaluate: ", precision, recall,f_score, output_stream=tf.compat.v1.logging.info)
        with tf.control_dependencies([print_op]):
          total_loss = tf.Print(total_loss, [total_loss], message="Loss = ")
      if which_loss == 'IOU':
        def compute_error(real, fake):
          intersection = tf.reduce_sum(tf.where(tf.equal(real,1.0),fake,tf.zeros_like(real)))
          union = tf.reduce_sum(tf.where(tf.equal(real,1.0),real,fake))
          return tf.subtract(1.0,tf.divide(intersection,union))
        total_loss= compute_error(gt_image,pred)
        total_loss = tf.compat.v1.Print(total_loss, [total_loss], message="Loss = ")
      if which_loss == 'IOU_IW':
        def compute_error(real, fake):
          intersection = tf.reduce_sum(tf.where(tf.equal(real[0],1.0),fake[0],tf.zeros_like(real[0])))
          union = tf.reduce_sum(tf.where(tf.equal(real[0],1.0),real[0],fake[0]))
          tl = tf.subtract(1.0,tf.divide(intersection,union))
          for b in range(1,FLAGS.batch_size):
            intersection = tf.reduce_sum(tf.where(tf.equal(real[b],1.0),fake[b],tf.zeros_like(real[b])))
            union = tf.reduce_sum(tf.where(tf.equal(real[b],1.0),real[b],fake[b]))
            tl = tf.subtract(1.0,tf.divide(intersection,union)) + tl
          return tl
        total_loss= compute_error(gt_image,pred)
        total_loss = tf.Print(total_loss, [total_loss], message="Loss = ")
      if which_loss == 'CrossEntropy':
        def compute_error(real, fake):
          return tf.nn.sigmoid_cross_entropy_with_logits(labels=real, logits=real,name='total_loss')
        total_loss= compute_error(gt_image,pred)
        #total_loss = tf.Print(total_loss, [total_loss], message="Loss = ")
 
    with tf.name_scope('train_op'):
        train_vars = [var for var in tf.compat.v1.trainable_variables()]
        optim = tf.compat.v1.train.AdamOptimizer(learning_rate, beta1)
        update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
        #grads_and_vars = optim.compute_gradients(total_loss, var_list=train_vars)
        #train_op = optim.apply_gradients(grads_and_vars)
        with tf.control_dependencies(update_ops):
           train_op = optim.minimize(total_loss,var_list=train_vars)

    
    # Summaries
    tf.compat.v1.summary.scalar('total_loss', total_loss)
    # Output image
    tf.compat.v1.summary.image('output_image', self.deprocess_image(pred))
    tf.compat.v1.summary.image('original_image', orig_image)
    #tf.compat.v1.summary.image('dont_care_image', self.deprocess_image(tf.cast(dc_image,dtype=tf.float32)))
    tf.compat.v1.summary.image('input_image', input_image)
    tf.compat.v1.summary.image('gt_image', self.deprocess_image(gt_image))

    return train_op

  def train(self, train_op, checkpoint_dir, continue_train, summary_freq,
            save_latest_freq, max_steps,writer=None):
    """Runs the training procedure.

    Args:
      train_op: op for training the network
      checkpoint_dir: where to save the checkpoints and summaries
      continue_train: whether to restore training from previous checkpoint
      summary_freq: summary frequency
      save_latest_freq: Frequency of model saving (overwrites old one)
      max_steps: maximum training steps
    """
    parameter_count = tf.reduce_sum(
        [tf.reduce_prod(tf.shape(v)) for v in tf.compat.v1.trainable_variables()])
    #global_step = tf.Variable(1, name='global_step', trainable=False)
    #incr_global_step = tf.compat.v1.assign(global_step, global_step + 1)
    global_step=tf.compat.v1.train.get_or_create_global_step()
    incr_global_step = tf.compat.v1.assign(global_step, global_step + 1)
    saver = tf.compat.v1.train.Saver(
        [var for var in tf.compat.v1.model_variables()] + [global_step], max_to_keep=10)
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sv = tf.compat.v1.train.MonitoredTrainingSession(
        checkpoint_dir=checkpoint_dir, summary_dir=checkpoint_dir,save_summaries_steps=summary_freq,save_checkpoint_steps=save_latest_freq,config=config)
   
    with sv as sess:
      tf.compat.v1.logging.info('Trainable variables: ')
      for var in tf.compat.v1.trainable_variables():
        tf.compat.v1.logging.info(var.name)
      tf.compat.v1.logging.info('parameter_count = %d' % sess.run(parameter_count))
      #if continue_train:
      #  checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
       # if checkpoint is not None:
      #    tf.compat.v1.logging.info('Resume training from previous checkpoint')
       #   saver.restore(sess, checkpoint)
      for step in range(1, max_steps):
        start_time = time.time()
        fetches = {
            'train': train_op,
            'global_step': global_step,
            'incr_global_step': incr_global_step,
        }
        #if step % summary_freq == 0:
            #tf.compat.v1.logging.info('Writing Summaries....')
            #writer.flush()

        results = sess.run(fetches)
        gs = results['global_step']
       # if step % summary_freq == 0
       #    tf.compat.v1.logging.info('[Step %.8d] time: %4.4f/it' % (gs, time.time() - start_time))

        #if step % save_latest_freq == 0:
          #tf.compat.v1.logging.info(' [*] Saving checkpoint to %s...' % checkpoint_dir)
          #saver.save(sess, os.path.join(checkpoint_dir, 'model.latest'))



  def preprocess_image(self, image):
    """Preprocess the image for CNN input.

    Args:
      image: the input image in either float [0, 1] or uint8 [0, 255]
    Returns:
      A new image converted to float with range [0, 1]
    """
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image

  def deprocess_image(self, image):
    """Undo the preprocessing.

    Args:
      image: the input image in float with range [0, 1]
    Returns:
      A new image converted to uint8 [0, 255]
    """
    return tf.image.convert_image_dtype(image, dtype=tf.uint8)


