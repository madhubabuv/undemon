from __future__ import division
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers import utils,fully_connected
import numpy as np



def resize_like(inputs, ref):
    iH, iW = inputs.get_shape()[1], inputs.get_shape()[2]
    rH, rW = ref.get_shape()[1], ref.get_shape()[2]
    if iH == rH and iW == rW:
        return inputs
    return tf.image.resize_nearest_neighbor(inputs, [rH.value, rW.value])

def pose_net(tgt_image, src_image_stack, is_training=True,reuse_variables=False):

    inputs = tf.concat([tgt_image, src_image_stack], axis=3)
    H = inputs.get_shape()[1].value
    W = inputs.get_shape()[2].value
    num_source = int(src_image_stack.get_shape()[3].value//3)
    with tf.variable_scope('pose_exp_net',reuse=reuse_variables) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose,fully_connected],
                            normalizer_fn=None,
                            weights_regularizer=slim.l2_regularizer(0.05),
                            activation_fn=tf.nn.relu,
                            outputs_collections=end_points_collection):
            # cnv1 to cnv5b are shared between pose and explainability prediction
            cnv1  = slim.conv2d(inputs,16,  [7, 7], stride=2, scope='cnv1')
            cnv2  = slim.conv2d(cnv1, 32,  [5, 5], stride=2, scope='cnv2')
            cnv3  = slim.conv2d(cnv2, 64,  [3, 3], stride=2, scope='cnv3')
            cnv4  = slim.conv2d(cnv3, 128, [3, 3], stride=2, scope='cnv4')
            cnv5  = slim.conv2d(cnv4, 256, [3, 3], stride=2, scope='cnv5')
            # Pose specific layers
            with tf.variable_scope('pose'):
                cnv6  = slim.conv2d(cnv5, 256, [3, 3], stride=2, scope='cnv6')
                cnv7  = slim.conv2d(cnv6, 512, [3, 3], stride=2, scope='cnv7')
                
                b,h,w,c=cnv7.get_shape().as_list()

                flat_1=tf.reshape(cnv7,(b,h*w*c))

                dense_1=fully_connected(flat_1,512,activation_fn=tf.nn.relu)

                pose_pred_xyz=fully_connected(dense_1,3*num_source,activation_fn=None)

                pose_pred_quat=fully_connected(dense_1,3*num_source,activation_fn=None)
               
                pose_avg=tf.concat((pose_pred_xyz,pose_pred_quat),1)
                # Empirically we found that scaling by a small constant 
                # facilitates training.
                pose_avg=0.1*pose_avg
                pose_final = tf.reshape(pose_avg, [-1, num_source, 6])

            end_points = utils.convert_collection_to_dict(end_points_collection)

            return pose_final, end_points
