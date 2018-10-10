from __future__ import division
from bilinear_sampler import bilinear_sampler_1d_h
import numpy as np

import tensorflow as tf

def preprocess_image(image):

    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image

def deprocess_image(image):

    return tf.image.convert_image_dtype(image, dtype=tf.uint8)

def create_mask(tensor, paddings):
    with tf.variable_scope('create_mask'):
        shape = tensor.get_shape().as_list()
        inner_width = shape[1] - (paddings[0][0] + paddings[0][1])
        inner_height = shape[2] - (paddings[1][0] + paddings[1][1])
        inner = tf.ones([inner_width, inner_height])

        mask2d = tf.pad(inner, paddings)
        mask3d = tf.tile(tf.expand_dims(mask2d, 0), [shape[0], 1, 1])
        mask4d = tf.expand_dims(mask3d, 3)
        return tf.stop_gradient(mask4d)


def create_border_mask(tensor, border_ratio=0.1):
    with tf.variable_scope('create_border_mask'):
        num_batch, height, width, _ = tensor.get_shape().as_list()

        min_dim = tf.cast(tf.minimum(height, width), 'float32')
        sz = tf.cast(tf.ceil(min_dim * border_ratio), 'int32')
        border_mask = create_mask(tensor, [[sz, sz], [sz, sz]])
        return tf.stop_gradient(border_mask)

def scale_pyramid(img, num_scales):

    if img == None:
        return None
    else:
        scaled_imgs = [img]
        _, h, w, _ = img.get_shape().as_list()
        for i in range(num_scales - 1):
            ratio = 2 ** (i + 1)
            nh = int(h / ratio)
            nw = int(w / ratio)
            scaled_imgs.append(tf.image.resize_area(img, [nh, nw]))
        return scaled_imgs

def generate_image_left( img, disp):
    return bilinear_sampler_1d_h(img, -disp)


def generate_image_right( img, disp):
    return bilinear_sampler_1d_h(img, disp)

def get_reference_explain_mask(downscaling,opt):
    tmp = np.array([0, 1])
    ref_exp_mask = np.tile(tmp,
                           (opt.batch_size,
                            int(opt.img_height / (2 ** downscaling)),
                            int(opt.img_width / (2 ** downscaling)),
                            1))
    ref_exp_mask = tf.constant(ref_exp_mask, dtype=tf.float32)
    return ref_exp_mask

