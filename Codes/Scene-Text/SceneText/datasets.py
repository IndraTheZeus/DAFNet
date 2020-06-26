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
"""Loads video sequence data using the tf Dataset API."""
import collections
import tensorflow.compat.v1 as tf
#from tensorflow import flags
FLAGS = tf.compat.v1.app.flags.FLAGS

class Scene(
    collections.namedtuple('Scene',
                           ['scene_image','sc_filename', 'gt_image','gt_filename','image_name','dc_image','binned_images', 'OrigImage', 'Language', 'Text'])):
  """
  """
  def set_batched_shape(self, batch_size):
    """
    Args:
      

    Returns:
      
    """

    def batch_one(tensor, dims):
      """Set shape for one tensor."""
      shape = tensor.get_shape().as_list()
      assert len(shape) == dims
      if shape[0] is None:
        shape[0] = batch_size
      else:
        assert shape[0] == batch_size
     
      tensor.set_shape(shape)

    batch_one(self.scene_image, 4)
    batch_one(self.binned_images, 4)
    batch_one(self.sc_filename, 1)
    batch_one(self.gt_image, 4)
    #batch_one(self.dc_image, 4)
    batch_one(self.gt_filename, 1)
    #batch_one(self.OrigImage, 4)
    batch_one(self.Language, 2)
    batch_one(self.Text, 2)
    return self


  def length(self):
    """  """
    return tf.shape(self.BoundingBox)[0]

  def random_scale_and_crop(self, min_scale, max_scale, height, width):
    """Randomly scale and crop sequence, for data augmentation.

    Args:
      min_scale: (float) minimum scale factor
      max_scale: (float) maximum scale factor
      height: (int) height of output images
      width: (int) width of output images

    Returns:
      A version of this sequence in which all images have been scaled in x and y
      by factors randomly chosen from min_scale to max_scale, and randomly
      cropped to give output images of the requested dimensions. Scaling and
      cropping are done consistently for all images in the sequence, and
      intrinsics are adjusted accordingly.
    """
    if min_scale == 1.0 and max_scale == 1.0:
      scaled_image = self.scene_image
      scaled_binned = self.binned_images
    else:
      input_size = tf.compat.v1.to_float(tf.shape(self.scene_image)[-3:-1])
      scale_factor = tf.random_uniform([2], min_scale, max_scale)
      scaled_image = tf.image.resize_area(self.scene_image, tf.to_int32(input_size * scale_factor))
      scaled_binned = tf.image.resize_area(self.binned_images, tf.to_int32(input_size * scale_factor))
      scaled_gt = tf.image.resize_area(self.gt_image, tf.to_int32(input_size * scale_factor)) 

    # Choose crop offset
    scaled_size = tf.shape(scaled_image)[-3:-1]
    offset_limit = scaled_size - [height, width] + 1
    offset_y = tf.random_uniform([], 0, offset_limit[0], dtype=tf.int32)
    offset_x = tf.random_uniform([], 0, offset_limit[1], dtype=tf.int32)

    sc_image,binned_images,gt_image = crop_image(scaled_image,scaled_binned,scaled_gt,offset_y, offset_x, height, width)
    return Scene(sc_image, self.sc_filename, gt_image,self.gt_filename,self.image_name,self.dc_image,binned_images ,self.scene_image,self.Language,self.Text)


def crop_image(
    scene_image, binned_images,gt_image, offset_y, offset_x, height, width):
  """Crop images and adjust instrinsics accordingly.

  Args:

  Returns:

  """
  # Convert to pixels, offset, and normalise to cropped size.
  cropped_images = tf.image.crop_to_bounding_box(scene_image, offset_y, offset_x, height, width)
  cropped_binned = tf.image.crop_to_bounding_box(binned_images, offset_y, offset_x, height, width)
  cropped_gt =  tf.image.crop_to_bounding_box(gt_image, offset_y, offset_x, height, width)
  return cropped_images, cropped_binned,cropped_gt



def Overlap2LevelBin(input_image,bins,max_distance=255,height=256,width=256):  
     binned_images = None
     input_image = tf.image.convert_image_dtype(input_image, dtype=tf.uint8)
     float_image = tf.image.convert_image_dtype(input_image, dtype=tf.float16)
     #ent=0
     for size in bins:
         k = size//2
         for main_lower in range(0,max_distance,k):
             #main_lower = tf.Print(main_lower, [main_lower], message="Main_lower = ")
             main_upper = main_lower+size-1+k      
             #main_upper = tf.Print(main_upper, [main_upper], message="Main_upper = ")
             gt_lower = input_image>=main_lower
             lt_higher = input_image<=main_upper
             bin_image = tf.logical_and(gt_lower,lt_higher)
             #mapped_image = tf.where(bin_image,float_image,tf.zeros_like(float_image))
             mapped_image = tf.cast(bin_image,dtype=tf.float16)
             resized_mapped = tf.squeeze(tf.compat.v1.image.resize_area(tf.expand_dims(mapped_image, axis=0), [height, width]),axis=0)
             resized_mapped.set_shape([height, width,3])
             assert len(resized_mapped.get_shape().as_list()) == 3
             assert resized_mapped.get_shape().as_list()[2] == 3 
             #tf.compat.v1.summary.image('binned_images'+str(ent),deprocess_image(tf.subtract(tf.multiply(tf.cast(bin_image,dtype=tf.float32),2.0),1.0)))
             #ent = ent+1
             if binned_images == None:
                 binned_images = resized_mapped
             else:
                 binned_images = tf.concat([binned_images,resized_mapped],axis=-1)
     resized_image = tf.squeeze(tf.compat.v1.image.resize_area(tf.expand_dims(float_image, axis=0), [height, width]),axis=0)
     resized_image.set_shape([height, width,3]) 
     #binned_images.set_shape([height,width,201])
     assert len(binned_images.get_shape().as_list()) == 3
     assert binned_images.get_shape().as_list()[2] == 198
     return tf.cast(binned_images>0.25,dtype=tf.float16)
     #return tf.concat([binned_images,resized_image],axis=-1)

def Overlap2LevelBin_2(input_image,bins,max_distance=255,height=256,width=256):
     binned_images = None
     input_image = tf.image.convert_image_dtype(input_image, dtype=tf.uint8)
     #ent=0
     redChannel = input_image[:,:,0]
     greenChannel = input_image[:,:,1]
     blueChannel = input_image[:,:,2]
     counter = 0
     for size in bins:
         k = size//2
         if counter%3 == 0:
             currChannel = redChannel
         elif counter%3 == 1:
             currChannel = greenChannel
         else:
             currChannel = blueChannel
         counter = counter + 1
         for main_lower in range(0,max_distance,k):
             #main_lower = tf.Print(main_lower, [main_lower], message="Main_lower = ")
             main_upper = main_lower+size-1+k      
             #main_upper = tf.Print(main_upper, [main_upper], message="Main_upper = ")
             gt_lower = currChannel>=main_lower
             lt_higher = currChannel<=main_upper
             bin_image = tf.logical_and(gt_lower,lt_higher)
             #mapped_image = tf.where(bin_image,float_image,tf.zeros_like(float_image))
             mapped_image = tf.cast(bin_image,dtype=tf.float16)
             resized_mapped = tf.expand_dims(mapped_image,axis=-1)
             #resized_mapped = tf.squeeze(tf.compat.v1.image.resize_area(tf.expand_dims(mapped_image, axis=0), [height, width]),axis=0)
             #resized_mapped.set_shape([height, width,1])
             #assert len(resized_mapped.get_shape().as_list()) == 3
             #assert resized_mapped.get_shape().as_list()[2] == 1
             #tf.compat.v1.summary.image('binned_images'+str(ent),deprocess_image(tf.subtract(tf.multiply(tf.cast(bin_image,dtype=tf.float32),2.0),1.0)))
             #ent = ent+1
             if binned_images == None:
                 binned_images = resized_mapped
             else:
                 binned_images = tf.concat([binned_images,resized_mapped],axis=-1)
     #resized_image = tf.squeeze(tf.compat.v1.image.resize_area(tf.expand_dims(float_image, axis=0), [height, width]),axis=0)
     #resized_image.set_shape([height, width,3]) 
     #binned_images.set_shape([height,width,201])
     assert len(binned_images.get_shape().as_list()) == 3
     assert binned_images.get_shape().as_list()[2] == 194
     return tf.cast(binned_images,dtype=tf.float16)
     #return tf.concat([binned_images,resized_image],axis=-1)


def get_scene_data(gt_file):
    splits = tf.compat.v1.string_split([gt_file],sep="/")
    gt_filename = splits.values[-1]
    gt_splits = tf.compat.v1.string_split([gt_filename],sep=".")
    image_name = gt_splits.values[0]
    scene_file = FLAGS.image_dir+"/"+image_name+"."+FLAGS.image_ext
    return Scene(tf.constant(0),scene_file,tf.constant(0),gt_file,image_name,tf.constant(0),tf.constant(0),tf.constant([[0,0,0,0]]),tf.constant(["English"]),tf.constant(["NO TEXT"]))



def deprocess_image(image):
    """Undo the preprocessing.

    Args:
      image: the input image in float with range [-1, 1]
    Returns:
      A new image converted to uint8 [0, 255]
    """
    image = (image + 1.) / 2.
    return tf.image.convert_image_dtype(image, dtype=tf.uint8)


def load_image_data(height, width,bins,max_distance,parallel_image_reads):
  """Returns a mapper function for loading image data.

  Args:
    base_path: (string) The base directory for images
    height: (int) Images will be resized to this height
    width: (int) Images will be resized to this width
    parallel_image_reads: (int) How many images to read in parallel

  Returns:
    A function mapping ViewSequence to ViewSequence, suitable for
    use with map(). The returned ViewSequence will be identical to the
    input one, except that sequence.images have been filled in.
  """
  def load_single_image(filename,channels=3):
    """Load and size a single image from a given filename."""
    contents = tf.io.read_file(filename)
    image = tf.image.convert_image_dtype(
        tf.image.decode_image(contents), tf.float32)
    # Unfortunately resize_area expects batched images, so add a dimension,
    # resize, and then remove it again.
    #resized = tf.squeeze(tf.compat.v1.image.resize_area(tf.expand_dims(image, axis=0), [height, width]),axis=0)
    #resized.set_shape([height, width,channels])  # RGB images have 3 channels.
    #return tf.concat([resized,resized,resized],axis=2)
    return image


  def mapper(scene):
    sc_image = load_single_image(scene.sc_filename)
    gt_image = load_single_image(scene.gt_filename,channels=1)
    #dc_image = tf.logical_and(tf.greater(gt_image,0.23),tf.logical_not(tf.greater_equal(gt_image,0.25)))
    dc_image = tf.constant([0])
    #gt_image = tf.where(dc_image,tf.zeros_like(gt_image),gt_image)
    
    #num_channels = binned_images.get_shape().as_list()[2]
    #resized_binned = tf.squeeze(tf.compat.v1.image.resize_area(tf.expand_dims(binned_images, axis=0), [height, width]),axis=0)
    #resized_binned.set_shape([height, width,FLAGS.num_bin_imgs])
    resized_input = tf.squeeze(tf.compat.v1.image.resize_area(tf.expand_dims(sc_image, axis=0), [height, width]),axis=0)
    resized_input.set_shape([height, width,3])
    resized_gt = tf.squeeze(tf.compat.v1.image.resize_area(tf.expand_dims(gt_image, axis=0), [height, width]),axis=0)
    resized_gt.set_shape([height, width,1])   #GroundTruth has 1 channel
    binned_images = Overlap2LevelBin_2(resized_input,bins,max_distance,height,width)
    return Scene(resized_input,scene.sc_filename,resized_gt,scene.gt_filename,scene.image_name,dc_image,binned_images, tf.constant([[0,0,0,0]]),
                        tf.constant(["ENGLISH"]), tf.constant(["NO TEXT"]))

  return mapper


