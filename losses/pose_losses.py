from __future__ import division
from utils.pose_utils import *
import tensorflow as tf
import tensorflow.contrib.slim as slim


def pose_losses(target_image,src_image_stack,disparity,pred_poses,intrinsics,opt):

    pixel_loss = 0

    pred_depth = [calc_depth(disparity[i],
                                  intrinsics[:, i, :, :],
                                  opt.img_width / (2 ** i)
                                  ) for i in range(len(disparity))]

    for s in range(opt.num_scales):

        curr_tgt_image = tf.image.resize_area(target_image,
                                              [int(opt.img_height / (2 ** s)),
                                               int(opt.img_width / (2 ** s))])

        curr_src_image_stack = tf.image.resize_area(src_image_stack,
                                                    [int(opt.img_height / (2 ** s)), int(opt.img_width / (2 ** s))])



        for i in range(opt.num_source):
            # Inverse warp the source image to the target image frame
            curr_proj_image = projective_inverse_warp(curr_src_image_stack[:, :, :, 3 * i:3 * (i + 1)],
                                                        pred_depth[s],
                                                        pred_poses[:, i, :],
                                                        intrinsics[:, s, :, :])

            curr_proj_error = tf.abs(curr_proj_image - curr_tgt_image)
            ssim_proj_image = SSIM(curr_proj_image, curr_tgt_image)
            # ssim_loss_temp_loss=tf.reduce_mean(ssim_proj_image)
            ssim_loss_temp_loss = charbonnier_loss(ssim_proj_image)


            pixel_loss += ((1 - opt.alpha_image_loss) * charbonnier_loss(curr_proj_error)
                           + opt.alpha_image_loss * ssim_loss_temp_loss)

    return {'image_loss':pixel_loss}


def SSIM( x, y):
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

def calc_depth( disp, cam_mat, img_width):

    batch, height, width, _ = disp.get_shape().as_list()

    base_length = 0.537

    f = cam_mat[:, 0, 0]

    pred_depth = tf.stack([(base_length * f[i]) / (img_width * disp[i])
                           for i in range(batch)])

    return pred_depth

def charbonnier_loss( x,mask=None,alpha=0.45, beta=1.0, epsilon=0.01):
    with tf.variable_scope('charbonnier_loss'):

        error = tf.pow(tf.square(x * beta) + tf.square(epsilon), alpha)

        if mask is not None:

            error=tf.multiply(mask,error)

        return tf.reduce_mean(error)
