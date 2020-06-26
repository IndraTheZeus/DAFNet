# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 02:28:49 2019

@author: indra
"""

sess = tf.compat.v1.Session()
with sess.as_default():
    tensor = tf.range(10)
    print_op = tf.print("tensors:", tensor, {2: tensor * 2},
                        output_stream=sys.stdout)
    with tf.control_dependencies([print_op]):
      tripled_tensor = tensor * 3
    sess.run(tripled_tensor)