from __future__ import division
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers import utils
import numpy as np

# Range of disparity/inverse depth values
DISP_SCALING = 0.3
MIN_DISP = 0.00001


def resize_like(inputs, ref):
    iH, iW = inputs.get_shape()[1], inputs.get_shape()[2]
    rH, rW = ref.get_shape()[1], ref.get_shape()[2]
    if iH == rH and iW == rW:
        return inputs
    return tf.image.resize_nearest_neighbor(inputs, [rH.value, rW.value])


def disp_net(tgt_image, is_training=True):
    H = tgt_image.get_shape()[1].value
    W = tgt_image.get_shape()[2].value
    with tf.variable_scope('depth_net') as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            normalizer_fn=None,
                            weights_regularizer=slim.l2_regularizer(0.005),
                            activation_fn=tf.nn.relu,
                            outputs_collections=end_points_collection):
            cnv1 = slim.conv2d(tgt_image, 32, [7, 7], stride=2, scope='cnv1')
            cnv1b = slim.conv2d(cnv1, 32, [7, 7], stride=1, scope='cnv1b')
            cnv2 = slim.conv2d(cnv1b, 64, [5, 5], stride=2, scope='cnv2')
            cnv2b = slim.conv2d(cnv2, 64, [5, 5], stride=1, scope='cnv2b')
            cnv3 = slim.conv2d(cnv2b, 128, [3, 3], stride=2, scope='cnv3')
            cnv3b = slim.conv2d(cnv3, 128, [3, 3], stride=1, scope='cnv3b')
            cnv4 = slim.conv2d(cnv3b, 256, [3, 3], stride=2, scope='cnv4')
            cnv4b = slim.conv2d(cnv4, 256, [3, 3], stride=1, scope='cnv4b')
            cnv5 = slim.conv2d(cnv4b, 512, [3, 3], stride=2, scope='cnv5')
            cnv5b = slim.conv2d(cnv5, 512, [3, 3], stride=1, scope='cnv5b')
            cnv6 = slim.conv2d(cnv5b, 512, [3, 3], stride=2, scope='cnv6')
            cnv6b = slim.conv2d(cnv6, 512, [3, 3], stride=1, scope='cnv6b')
            cnv7 = slim.conv2d(cnv6b, 512, [3, 3], stride=2, scope='cnv7')
            cnv7b = slim.conv2d(cnv7, 512, [3, 3], stride=1, scope='cnv7b')

            upcnv7 = slim.conv2d_transpose(cnv7b, 512, [3, 3], stride=2, scope='upcnv7')
            # There might be dimension mismatch due to uneven down/up-sampling
            upcnv7 = resize_like(upcnv7, cnv6b)
            i7_in = tf.concat([upcnv7, cnv6b], axis=3)
            icnv7 = slim.conv2d(i7_in, 512, [3, 3], stride=1, scope='icnv7')

            upcnv6 = slim.conv2d_transpose(icnv7, 512, [3, 3], stride=2, scope='upcnv6')
            upcnv6 = resize_like(upcnv6, cnv5b)
            i6_in = tf.concat([upcnv6, cnv5b], axis=3)
            icnv6 = slim.conv2d(i6_in, 512, [3, 3], stride=1, scope='icnv6')

            upcnv5 = slim.conv2d_transpose(icnv6, 256, [3, 3], stride=2, scope='upcnv5')
            upcnv5 = resize_like(upcnv5, cnv4b)
            i5_in = tf.concat([upcnv5, cnv4b], axis=3)
            icnv5 = slim.conv2d(i5_in, 256, [3, 3], stride=1, scope='icnv5')

            upcnv4 = slim.conv2d_transpose(icnv5, 128, [3, 3], stride=2, scope='upcnv4')
            i4_in = tf.concat([upcnv4, cnv3b], axis=3)
            icnv4 = slim.conv2d(i4_in, 128, [3, 3], stride=1, scope='icnv4')
            disp4 = DISP_SCALING * slim.conv2d(icnv4, 2, [3, 3], stride=1,
                                               activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp4') + MIN_DISP
            disp4_up = tf.image.resize_bilinear(disp4, [np.int(H / 4), np.int(W / 4)])

            upcnv3 = slim.conv2d_transpose(icnv4, 64, [3, 3], stride=2, scope='upcnv3')
            i3_in = tf.concat([upcnv3, cnv2b, disp4_up], axis=3)
            icnv3 = slim.conv2d(i3_in, 64, [3, 3], stride=1, scope='icnv3')
            disp3 = DISP_SCALING * slim.conv2d(icnv3, 2, [3, 3], stride=1,
                                               activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp3') + MIN_DISP
            disp3_up = tf.image.resize_bilinear(disp3, [np.int(H / 2), np.int(W / 2)])

            upcnv2 = slim.conv2d_transpose(icnv3, 32, [3, 3], stride=2, scope='upcnv2')
            i2_in = tf.concat([upcnv2, cnv1b, disp3_up], axis=3)
            icnv2 = slim.conv2d(i2_in, 32, [3, 3], stride=1, scope='icnv2')
            disp2 = DISP_SCALING * slim.conv2d(icnv2, 2, [3, 3], stride=1,
                                               activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp2') + MIN_DISP
            disp2_up = tf.image.resize_bilinear(disp2, [H, W])

            upcnv1 = slim.conv2d_transpose(icnv2, 16, [3, 3], stride=2, scope='upcnv1')
            i1_in = tf.concat([upcnv1, disp2_up], axis=3)
            icnv1 = slim.conv2d(i1_in, 16, [3, 3], stride=1, scope='icnv1')
            disp1 = DISP_SCALING * slim.conv2d(icnv1, 2, [3, 3], stride=1,
                                               activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp1') + MIN_DISP

            end_points = utils.convert_collection_to_dict(end_points_collection)

            return [disp1, disp2, disp3, disp4], end_points
