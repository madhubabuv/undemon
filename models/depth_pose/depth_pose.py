from __future__ import division
from nets.depth_nets.DepthNet import disp_net
from nets.pose_nets.PoseNet import pose_net
from losses.depth_losses import depth_losses
from losses.pose_losses import pose_losses
from main import Net
import tensorflow as tf

class DepthPose(Net):

    def __init__(self,opt):

        self.args=opt

        super(DepthPose, self).__init__()

    def model(self,inputs,reuse_variables=None):

        target_image=inputs['left_target_image']

        source_images=inputs['left_source_images']

        predicted_disparity,disp_end_points=disp_net(target_image,
                                                     is_training=True,
                                                     reuse_variables=reuse_variables)

        predicted_pose,pose_end_points =pose_net(target_image,
                                 source_images,
                                 is_training=True,
                                 reuse_variables=reuse_variables)

        return {
            'disparity':predicted_disparity,
            'pose':predicted_pose,
        }

    def loss(self,inputs,predictions,reuse_variables=None):


        with tf.variable_scope('undemon_flow_loss',reuse=reuse_variables):

            left_target_image=inputs['left_target_image']

            left_source_images=inputs['left_source_images']

            right_target_image=inputs['right_target_image']

            intrinsics=inputs['intrinsics']

            pred_disp=predictions['disparity']

            pose=predictions['pose']

            disp_left_est = [tf.expand_dims(d[:, :, :, 0], 3) for d in pred_disp]

            disp_right_est = [tf.expand_dims(d[:, :, :, 1], 3) for d in pred_disp]

            self.args.num_scales = 4

            with tf.variable_scope('depth_losses'):

                losses_d=depth_losses(left_target_image,
                                      right_target_image,
                                      disp_left_est,disp_right_est,
                                      self.args)

                depth_loss=losses_d['image_loss']+\
                           losses_d['gradient_loss']+\
                           losses_d['lr_loss']

            with tf.variable_scope('pose_losses'):

                losses_p=pose_losses(left_target_image,
                                     left_source_images,
                                     disp_left_est,
                                     pose,
                                     intrinsics,
                                     self.args)

                pose_loss=losses_p['image_loss']


            total_loss=depth_loss+pose_loss

            #preserve the losses for the summary

            self.depth_appearance_loss=losses_d['image_loss']
            self.depth_gradient_loss=losses_d['gradient_loss']
            self.depth_lr_loss=losses_d['lr_loss']
            self.pose_appearance_loss=losses_p['image_loss']

        return total_loss

    def build_summaries(self,inputs,predictions):

        with tf.device('/cpu:0'):

            tf.summary.image("left_target", inputs['left_target_image'])
            tf.summary.image("right_target", inputs['right_target_image'])
            tf.summary.image("source_images_1", inputs['left_source_images'][:, :, :, 0:3])
            tf.summary.image("source_images_2", inputs['left_source_images'][:, :, :, 3:6])

            if 'disparity' in predictions and predictions['disparity'] is not None:

                tf.summary.scalar('depth_appearnace_loss',self.depth_appearance_loss)
                tf.summary.scalar('depth_gradient_loss',self.depth_gradient_loss)

                tf.summary.image('disparity', predictions['disparity'][0][:, :, :, 0:1])

            if 'pose' in predictions:

                tf.summary.scalar('pose_appearance_loss',self.pose_appearance_loss)

                tf.summary.histogram('x', predictions['pose'][:, :, 0])
                tf.summary.histogram('y', predictions['pose'][:, :, 1])
                tf.summary.histogram('z', predictions['pose'][:, :, 2])
                tf.summary.histogram('roll', predictions['pose'][:, :, 3])
                tf.summary.histogram('pitch', predictions['pose'][:, :, 4])
                tf.summary.histogram('Yaw', predictions['pose'][:, :, 5])
