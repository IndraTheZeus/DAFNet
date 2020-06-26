from __future__ import division
import tensorflow as tf
import tensorflow.compat.v2.nn as slim
#from tensorflow import flags
FLAGS = tf.compat.v1.app.flags.FLAGS

def binningResNet(binned_images,orig_image, num_outputs, ngf=64, vscope='net', reuse_weights=False):
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
  in_shape = binned_images.get_shape().as_list()
  assert in_shape == [FLAGS.batch_size,FLAGS.image_height,FLAGS.image_width,194]
  with tf.compat.v1.variable_scope(vscope, reuse=reuse_weights):
     
   # with tf.compat.v2.nn as slim:
      def conv2d(inp,name,filter_shape,out_layers,stride=1):
          w = tf.compat.v1.get_variable('w_'+name,shape=(filter_shape[0],filter_shape[1],inp.get_shape().as_list()[3],out_layers),initializer=tf.compat.v1.keras.initializers.glorot_uniform(),trainable=True)
          b = tf.compat.v1.get_variable('b_'+name,shape=(out_layers),initializer=tf.zeros_initializer(),trainable=True)
          cnv = slim.bias_add(slim.conv2d(inp,w, strides=stride,padding="SAME"),b)
          return cnv
     
      def conv2d_transpose(inp,name,filter_shape,out_layers,stride=1):
          w = tf.compat.v1.get_variable('w_'+name,shape=(filter_shape[0],filter_shape[1],out_layers,inp.get_shape().as_list()[3]),initializer=tf.compat.v1.keras.initializers.glorot_uniform(),trainable=True)
          b = tf.compat.v1.get_variable('b_'+name,shape=(out_layers),initializer=tf.zeros_initializer(),trainable=True)
          cnv = slim.bias_add(slim.conv2d_transpose(inp,w, strides=stride,padding="SAME",output_shape=(inp.get_shape().as_list()[0],inp.get_shape().as_list()[1]*2,inp.get_shape().as_list()[2]*2,out_layers)),b)
          return cnv         

      def res_2_transpose(inp,block_name,filter_matrix,out_channel_list,stride_list=[1,1,1],dropout_list=[0.0,0.0])    :
          cnv1 = slim.relu(batch_normalization(conv2d_transpose(inp,block_name+'_cnv1',filter_matrix[0],out_channel_list[0],stride=stride_list[0]),block_name+'_bn1'))
          cnv2 = batch_normalization(conv2d(cnv1,block_name+'_cnv2',filter_matrix[1],out_channel_list[1],stride=stride_list[1]),block_name+'_bn2')
          cnv_r = conv2d_transpose(inp,block_name+'_cnvr',[1,1],out_channel_list[1],stride=stride_list[2])
          return slim.relu(cnv_r+cnv2)
          
      def res_2_block(inp,block_name,filter_matrix,out_channel_list,stride_list=[1,1,1],dropout_list=[0.0,0.0]):
          w1 = tf.compat.v1.get_variable('w1_'+block_name,shape=(filter_matrix[0],filter_matrix[0],inp.get_shape().as_list()[3],out_channel_list[0]),initializer=tf.compat.v1.keras.initializers.glorot_uniform(),trainable=True)
          b1 = tf.compat.v1.get_variable('b1_'+block_name,shape=(out_channel_list[0]),initializer=tf.zeros_initializer(),trainable=True)
          cnv1 = slim.bias_add(slim.conv2d(inp,w1, strides=stride_list[0],padding="SAME"),b1)
          bn_cnv1 = slim.relu(batch_normalization(cnv1,block_name+'_bn1'))
          w2 = tf.compat.v1.get_variable('w2_'+block_name,shape=(filter_matrix[1],filter_matrix[1],bn_cnv1.get_shape().as_list()[3],out_channel_list[1]),initializer=tf.compat.v1.keras.initializers.glorot_uniform(),trainable=True)
          b2 = tf.compat.v1.get_variable('b2_'+block_name,shape=(out_channel_list[1]),initializer=tf.zeros_initializer(),trainable=True)
          cnv2 = slim.bias_add(slim.conv2d(bn_cnv1,w2, strides=stride_list[1],padding="SAME"),b2)
          bn_cnv2 = batch_normalization(cnv2,block_name+'_bn2')
          if(inp.get_shape().as_list()[3] != out_channel_list[1]):
             wr = tf.compat.v1.get_variable('wr_'+block_name,shape=(1,1,inp.get_shape().as_list()[3],out_channel_list[1]),initializer=tf.compat.v1.keras.initializers.glorot_uniform(),trainable=True)
             br = tf.compat.v1.get_variable('br_'+block_name,shape=(out_channel_list[1]),initializer=tf.zeros_initializer(),trainable=True)
             cnv_r = slim.bias_add(slim.conv2d(inp,wr, strides=stride_list[2],padding="SAME"),br)
          else:
             cnv_r = inp
          out_cnv = cnv_r+bn_cnv2
          return slim.relu(out_cnv)
      
      def batch_normalization(inp,bn_name,momment=0.9):
          batchNm = tf.compat.v1.layers.batch_normalization(inp,momentum = momment,training=FLAGS.training,name=bn_name,reuse=tf.compat.v1.AUTO_REUSE,renorm=False)
          return batchNm
      
    
      
    
      conv1 = slim.relu(batch_normalization(conv2d(binned_images,'layer1',[7,7],128),'_bn1'))
      conv2 = slim.relu(batch_normalization(conv2d(conv1,'layer2',[7,7],64),'_bn2'))
      #res3 = res_2_block(conv2,'layer3',[3,3],[64,64],[1,1,1])
      res4 = res_2_block(conv2,'layer4',[3,3],[64,64],[1,1,1])
      res5 = res_2_block(res4,'layer5',[3,3],[128,128],[2,1,2])
      res6 = res_2_block(res5,'layer6',[3,3],[128,128],[1,1,1])
      res7 = res_2_block(res6,'layer7',[3,3],[256,256],[2,1,2])
      res8 = res_2_block(res7,'layer8',[3,3],[256,256],[1,1,1])
      res8_1 = res_2_block(res8,'layer8_1',[3,3],[512,512],[2,1,2])
      res8_2 = res_2_block(res8_1,'layer8_2',[3,3],[512,512],[1,1,1])
      res8_3 = res_2_transpose(res8_2,'layer8_3',[[3,3],[3,3]],[256,256],[2,1,2])
      res8_4 = res_2_block(res8_3,'layer8_4',[3,3],[256,256],[1,1,1]) 
      res9 = res_2_transpose(res8_4,'layer9',[[3,3],[3,3]],[128,128],[2,1,2])
      res10 = res_2_block(res9,'layer10',[3,3],[128,128],[1,1,1])
      res11 = res_2_transpose(res10,'layer11',[[3,3],[3,3]],[64,64],[2,1,2])
      res12 = res_2_block(res11,'layer12',[3,3],[64,64],[1,1,1])
      conv13 = slim.relu(batch_normalization(conv2d(res12,'layer13',[7,7],32),'_bn3'))
      conv14 = slim.relu(batch_normalization(conv2d(conv13,'layer14',[7,7],16),'_bn4'))
      conv15 = slim.relu(batch_normalization(conv2d(conv14,'layer15',[7,7],8),'_bn5'))
      conv16 = slim.relu(batch_normalization(conv2d(conv15,'layer16',[7,7],4),'_bn6'))
      conv17 = slim.relu(batch_normalization(conv2d(conv16,'layer17',[7,7],2),'_bn7'))
      conv18 = slim.tanh(conv2d(conv17,'layer18',[7,7],1))         

      return conv18     
