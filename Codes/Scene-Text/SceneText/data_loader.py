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
"""Class definition of the data loader.
"""
from __future__ import division

#from tensorflow import flags

from SceneText import loader
import tensorflow as tf
FLAGS = tf.compat.v1.app.flags.FLAGS

class DataLoader(object):
  """Loader for video sequence data."""

  def __init__(self,
               ground_truth_dir,
               image_dir,
               bins,
               training=True,
              ):

    self.batch_size = FLAGS.batch_size
    self.image_height = FLAGS.image_height
    self.image_width = FLAGS.image_width

    self.datasets = loader.create_from_flags(
        ground_truth_dir =ground_truth_dir,
        image_dir=image_dir,
        bins=bins,
        training=training)


  def sample_batch(self):
    """Samples a batch of examples for training / testing.

    Returns:
      A batch of examples.
    """
    example = self.datasets.scenes
    iterator = tf.compat.v1.data.make_one_shot_iterator(example)
    return iterator.get_next()


