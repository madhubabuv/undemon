from __future__ import division
import tensorflow as tf
import tensorflow.contrib.slim as slim
from utils.bilinear_sampler import bilinear_sampler_1d_h
from utils.utils import create_border_mask
import numpy as np



def depth_losses(left,right,disp_left_est,disp_right_est,opt):

    image_loss = 0
    disp_gradient_loss = 0
    lr_loss = 0

    for s in range(opt.num_scales):
        curr_tgt_image = tf.image.resize_area(left,
                                              [int(opt.img_height / (2 ** s)), int(opt.img_width / (2 ** s))])


        right_curr_tgt_image = tf.image.resize_area(right,
                                                    [int(opt.img_height / (2 ** s)), int(opt.img_width / (2 ** s))])

        # Generating images with bilinear sampling
        with tf.variable_scope('images_from_disp'):

            left_est = generate_image_left(right_curr_tgt_image, disp_left_est[s])
            right_est = generate_image_right(curr_tgt_image, disp_right_est[s])

        with tf.variable_scope('left-right'):
            right_to_left_disp = generate_image_left(disp_right_est[s], disp_left_est[s])

            left_to_right_disp = generate_image_right(disp_left_est[s], disp_right_est[s])

        with tf.variable_scope('disparity_stereo_losses'):
            # L1
            l1_left = tf.abs(left_est - curr_tgt_image)
            l1_reconstruction_loss_left = charbonnier_loss(l1_left)

            l1_right = tf.abs(right_est - right_curr_tgt_image)
            l1_reconstruction_loss_right = charbonnier_loss(l1_right)

            # ssim

            ssim_left = SSIM(left_est, curr_tgt_image)
            ssim_right = SSIM(right_est, right_curr_tgt_image)


            ssim_loss_left = charbonnier_loss(ssim_left)
            ssim_loss_right = charbonnier_loss(ssim_right)

            image_loss_right = opt.alpha_image_loss * ssim_loss_right + (
                                                                    1 - opt.alpha_image_loss) * l1_reconstruction_loss_right
            image_loss_left = opt.alpha_image_loss * ssim_loss_left + (1 - opt.alpha_image_loss) * l1_reconstruction_loss_left
            image_loss += (image_loss_left + image_loss_right)

            # LR Consistency

            lr_left_loss = charbonnier_loss(tf.abs(right_to_left_disp - disp_left_est[s]))
            lr_right_loss = charbonnier_loss(tf.abs(left_to_right_disp - disp_right_est[s]))

            lr_loss += (lr_left_loss + lr_right_loss)

            #Smoothness Loss

            disp_left_smoothness = get_disparity_smoothness(disp_left_est[s], curr_tgt_image)
            disp_right_smoothness = get_disparity_smoothness(disp_right_est[s], right_curr_tgt_image)


            disp_left_loss = disp_left_smoothness / (2 ** s)
            disp_right_loss = disp_right_smoothness / (2 ** s)

            disp_gradient_loss += (disp_left_loss + disp_right_loss)

    return {

        'image_loss':image_loss,
        'gradient_loss':0.1*disp_gradient_loss,
        'lr_loss':lr_loss,
    }


def charbonnier_loss( x,mask=None,alpha=0.45, beta=1.0, epsilon=0.01):

    with tf.variable_scope('charbonnier_loss'):

        error = tf.pow(tf.square(x * beta) + tf.square(epsilon), alpha)

        if mask is not None:

            error=tf.multiply(mask,error)

        return tf.reduce_mean(error)


def gradient_x( img):
    gx = img[:, :, :-1, :] - img[:, :, 1:, :]
    return gx


def gradient_y( img):
    gy = img[:, :-1, :, :] - img[:, 1:, :, :]
    return gy


def get_disparity_smoothness( disp, img,full_scale=False):
    if full_scale:
        disp_mean=tf.reduce_mean(disp)
        disp_gradients_x = gradient_x(disp/disp_mean)
        disp_gradients_y = gradient_y(disp/disp_mean)
    else:
        disp_gradients_x = gradient_x(disp)
        disp_gradients_y = gradient_y(disp)

    image_gradients_x = gradient_x(img)
    image_gradients_y = gradient_y(img)

    weights_x = tf.exp(-tf.reduce_mean(tf.abs(image_gradients_x), 3, keep_dims=True))
    weights_y = tf.exp(-tf.reduce_mean(tf.abs(image_gradients_y), 3, keep_dims=True))

    smoothness_x = disp_gradients_x * weights_x
    smoothness_y = disp_gradients_y * weights_y

    return charbonnier_loss(tf.abs(smoothness_x)) + charbonnier_loss(tf.abs(smoothness_y))

def SSIM(x, y):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = slim.avg_pool2d(x, 3, 1, 'VALID')
    mu_y = slim.avg_pool2d(y, 3, 1, 'VALID')

    sigma_x = slim.avg_pool2d(x ** 2, 3, 1, 'VALID') - mu_x ** 2
    sigma_y = slim.avg_pool2d(y ** 2, 3, 1, 'VALID') - mu_y ** 2
    sigma_xy = slim.avg_pool2d(x * y, 3, 1, 'VALID') - mu_x * mu_y

    SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

    SSIM = SSIM_n / SSIM_d


    return tf.clip_by_value((1 - SSIM) / 2, 0, 1)

def generate_image_left( img, disp):
    return bilinear_sampler_1d_h(img, -disp)


def generate_image_right( img, disp):
    return bilinear_sampler_1d_h(img, disp)
