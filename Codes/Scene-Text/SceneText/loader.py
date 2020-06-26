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
"""Loads video sequence data using the tf Dataset API.

For easiest setup, use create_from_flags, then you won't have to specify any
additional options.
"""
from __future__ import division
import tensorflow as tf
FLAGS = tf.compat.v1.app.flags.FLAGS
from SceneText import datasets
import random



#FLAGS = flags.FLAGS

# The lambdas in this file are only long because of clear argument names.
# pylint: disable=g-long-lambda


class Loader(object):
  """Process video sequences into a dataset for use in training or testing."""

  def __init__(
      self,
      # What to load
      scenes,
      # Whether data to be loaded is for use in training (vs. test). This
      # affects whether data is sampled pseudorandomly (in the case of training
      # data) or deteministically (in the case of validation/test data), and
      # whether data augmentation is applied (training data only).
      bins,
      max_distance=255,
      training=True,
      # Output dimensions
      image_height=None,
      image_width=None,
      
      epochs = -1, 
      # Batching
      batch_size=8,
      map_function=None,
      # Augmentation
      min_scale=1.0,
      max_scale=1.0,

      # Tuning for efficiency
      parallelism=10,
      parallel_image_reads=50,
      prefetch=1,  
      ):
    """
    """
    
    def prepare_for_training(scenes):
      """Steps applied to training dataset only."""
      # Random shuffling, random subsequences and random reversal for training.
      # Also we make it repeat indefinitely.
      shuffled = scenes.shuffle(FLAGS.buffer_size).repeat(epochs)
      return shuffled



    def load_image_data(scenes):
      return scenes.map(datasets.load_image_data(image_height, image_width,bins,max_distance,
                                   parallel_image_reads),num_parallel_calls=parallelism)
     
      
    def set_batched_shape(scene):
      return scene.set_batched_shape(batch_size)
      
    def batch_and_prefetch(dataset):
      return (dataset.padded_batch(batch_size, tf.compat.v1.data.get_output_shapes(dataset),drop_remainder=True)
              .map(set_batched_shape).prefetch(prefetch))
      
      
    #if training:
    scenes = prepare_for_training(scenes)

     # Load images
    scenes = load_image_data(scenes)
   
    scenes = batch_and_prefetch(scenes)
    
    if FLAGS.training and (FLAGS.min_scale != 1.0 or FLAGS.max_scale != 1.0):
       scenes = scenes.map(lambda scene: scene.random_scale_and_crop(FLAGS.min_scale, FLAGS.max_scale, image_height, image_width))



    # Custom processing
    if map_function:
      scenes = scenes.map(
          map_function, num_parallel_calls=parallelism)


    # Things we expose to the calling code
    self.scenes = scenes


# Create a dataset configured with the flags specified at the top of this file.
def create_from_flags(ground_truth_dir,
                      image_dir,
                      bins,
                      training=True):
  """Convenience function to return a Loader configured by flags."""
  #assert tf.io.gfile.isdir(image_dir)  # Ensure the provided path is valid.
  assert len(tf.io.gfile.listdir(image_dir)) > 0  # Ensure that some data exists.
  parallelism = 10
  ground_truth_glob = ground_truth_dir + '/*.' + FLAGS.gt_ext
  assert len(tf.io.gfile.glob(ground_truth_glob)) > 0
  gt_files = tf.data.Dataset.list_files(ground_truth_glob, False)
  #Added
  
  
  #Ended
  scenes = gt_files.map(
      datasets.get_scene_data, num_parallel_calls=parallelism)

  return Loader(
      scenes,
      bins,
      max_distance =FLAGS.max_distance,
      training = training,
      # Output dimensions
      image_height=FLAGS.image_height,
      image_width=FLAGS.image_width,
      epochs = FLAGS.epochs,
      # Batching
      batch_size=FLAGS.batch_size,
      # Augmentation
      min_scale=1.0,
      max_scale=1.0,
      # Tuning for efficiency
      parallelism=parallelism,
      parallel_image_reads=50,
      prefetch=1)
