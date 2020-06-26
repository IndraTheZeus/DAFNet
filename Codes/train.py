# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 17:35:52 2019

@author: indra
"""
from __future__ import division
import tensorflow as tf

from SceneText.data_loader import DataLoader

from SceneText.bin_model import BIN_MODEL

import array as arr
flags = tf.compat.v1.app.flags
flags.DEFINE_string('checkpoint_dir', 'checkpoints',
                    'Location to save the models.')
flags.DEFINE_boolean('training', True,
                     'Is Training and Gradient Descent Active.')
flags.DEFINE_integer('num_test_files', 301, 'Number of Files to be tested when training is False')

flags.DEFINE_string('gt_dir', '/content/drive/My Drive/Deep Learning/Datasets/Kaist/pixelGT_train.rar (Unzipped Files)/pixelGT_train',
                    'Folder containing GroundTruth data')

flags.DEFINE_string('gt_ext', 'bmp',
                    'Extension to ground truth')


flags.DEFINE_string('image_dir', '/content/drive/My Drive/Deep Learning/Datasets/Kaist/Images.rar (Unzipped Files)/Images',
                    'Path to training image directories.')

flags.DEFINE_string('image_ext', 'jpg',
                    'Extension to scene image')

flags.DEFINE_integer('num_bin_imgs', 201, 'Number of Bin Images')
flags.DEFINE_integer('max_distance', 255, 'Maximum pixel value possible')

flags.DEFINE_string('experiment_name', 'Kaist_IOU_RES30_ABinary2', 'Name for the experiment to run.')   #Testing Model3,ResModel2

flags.DEFINE_string('which_loss', 'IOU', 'Which loss to use to compare '
    'rendered and ground truth images. '
    'Can be "IOU" or "FScore" or "CrossEntropy" or "IOU_IW".')

flags.DEFINE_float('learning_rate', 0.0001, 'Learning rate')

flags.DEFINE_float('beta1', 0.9, 'beta1 hyperparameter for Adam optimizer.')

flags.DEFINE_integer('summary_freq', 30, 'Logging frequency.')

flags.DEFINE_integer(
    'save_latest_freq', 900, 'Frequency with which to save the model '
    '(overwrites previous model).')

flags.DEFINE_boolean('continue_train', False,
                     'Continue training from previous checkpoint.')

flags.DEFINE_integer('random_seed', 8964, 'Random seed.')

flags.DEFINE_integer('epochs', -1,
                     'Epochs of training data, or -1 to continue indefinitely.')

flags.DEFINE_integer('image_height', 256, 'Image height in pixels.')
flags.DEFINE_integer('image_width', 256, 'Image width in pixels.')



flags.DEFINE_float('min_scale', 1.0,
                   'Minimum scale for data augmentation.')
flags.DEFINE_float('max_scale', 1.5,
                   'Maximum scale for data augmentation.')

flags.DEFINE_integer('batch_size', 5, 'The size of a sample batch.')

flags.DEFINE_integer('buffer_size', 2000, 'The size of buffer for shuffling')
flags.DEFINE_integer('max_steps', 10000000, 'Maximum number of training steps.')


FLAGS = flags.FLAGS


def main(_):
  
  if not(FLAGS.training):
      assert FLAGS.batch_size == 1
  else:
    assert FLAGS.batch_size > 1
      
  bins = range(24,93,4)
    
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
#  tf.compat.v1.set_random_seed(FLAGS.random_seed)
  FLAGS.checkpoint_dir += '/%s/' % FLAGS.experiment_name
  if not tf.io.gfile.isdir(FLAGS.checkpoint_dir):
    tf.compat.v1.gfile.MakeDirs(FLAGS.checkpoint_dir)
    
  writer = tf.compat.v2.summary.create_file_writer(FLAGS.checkpoint_dir)
  tf.compat.v2.summary.experimental.set_step(1)
    # Set up data loader
  data_loader = DataLoader(FLAGS.gt_dir, FLAGS.image_dir,bins, True)
  train_batch = data_loader.sample_batch()
  
  model = BIN_MODEL()
  if FLAGS.training:
    train_op= model.build_train_graph(train_batch,bins,FLAGS.which_loss,FLAGS.learning_rate, FLAGS.beta1,writer)
    model.train(train_op, FLAGS.checkpoint_dir, FLAGS.continue_train,
              FLAGS.summary_freq, FLAGS.save_latest_freq, FLAGS.max_steps,writer)
  else:
     test_op = model.build_test_graph(train_batch,FLAGS.which_loss)
     model.test(test_op,FLAGS.checkpoint_dir,FLAGS.num_test_files)


if __name__ == '__main__':
  tf.compat.v1.app.run()
