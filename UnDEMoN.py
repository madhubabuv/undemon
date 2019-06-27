from __future__ import division
import os
import time
import math
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from undemon_data_loader import DataLoader
from nets import *
from utils import *
from bilinear_sampler import *
class UnDEMoN(object):
    def __init__(self):
        pass
    
    def build_train_graph(self):
        opt = self.opt
        loader = DataLoader(opt.dataset_dir,
                            opt.batch_size,
                            opt.mode,
                            opt.img_height,
                            opt.img_width,
                            opt.num_source,
                            opt.num_scales)
        with tf.name_scope("data_loading"):

            train_batch=loader.load_train_batch()
            left_tgt_image, left_src_image_stack, intrinsics = train_batch['left']
            right_tgt_image, right_src_image_stack, _ = train_batch['right']

            left_tgt_image = self.preprocess_image(left_tgt_image)
            left_src_image_stack = self.preprocess_image(left_src_image_stack)

            right_tgt_image = self.preprocess_image(right_tgt_image)
            #right_src_image_stack = self.preprocess_image(right_src_image_stack)

            tgt_image=left_tgt_image

            src_image_stack=left_src_image_stack

        with tf.name_scope("depth_prediction"):

            pred_disp, depth_net_endpoints = disp_net(left_tgt_image,
                                                      is_training=True)
            disp_left_est = [tf.expand_dims(d[:, :, :, 0], 3) for d in pred_disp]
            disp_right_est = [tf.expand_dims(d[:, :, :, 1], 3) for d in pred_disp]

            pred_depth = [self.calc_depth(disp_left_est[i],
                                          intrinsics[:, i, :, :],
                                          opt.img_width / (2 ** i)
                                          ) for i in range(4)]

        with tf.name_scope("pose_net"):
            pred_poses,pose_net_endpoints = \
                pose_net(tgt_image,
                             src_image_stack,
                             is_training=True)

        with tf.name_scope("compute_loss"):
            pixel_loss = 0
            exp_loss = 0
            smooth_loss = 0
            image_loss=0
            disp_gradient_loss=0
            lr_loss=0
            tgt_image_all = []
            src_image_stack_all = []
            proj_image_stack_all = []
            proj_error_stack_all = []
            exp_mask_stack_all = []
            for s in range(opt.num_scales):

                #left
                curr_tgt_image = tf.image.resize_area(tgt_image, 
                    [int(opt.img_height/(2**s)), int(opt.img_width/(2**s))])
                curr_src_image_stack = tf.image.resize_area(src_image_stack, 
                    [int(opt.img_height/(2**s)), int(opt.img_width/(2**s))])

                #right

                right_curr_tgt_image = tf.image.resize_area(right_tgt_image,
                    [int(opt.img_height / (2 ** s)), int(opt.img_width / (2 ** s))])

                # Generating images with bilinear sampling
                with tf.variable_scope('images_from_disp'):
                    left_est = self.generate_image_left(right_curr_tgt_image, disp_left_est[s])
                    right_est = self.generate_image_right(curr_tgt_image, disp_right_est[s])

                with tf.variable_scope('left-right'):
                    right_to_left_disp = self.generate_image_left(disp_right_est[s], disp_left_est[s])

                    left_to_right_disp = self.generate_image_right(disp_left_est[s], disp_right_est[s])


                with tf.variable_scope('disparity_stereo_losses'):

                    #L1
                    l1_left = tf.abs(left_est - curr_tgt_image)
                    l1_reconstruction_loss_left = self.charbonnier_loss(l1_left)

                    l1_right = tf.abs(right_est - right_curr_tgt_image)
                    l1_reconstruction_loss_right = self.charbonnier_loss(l1_right)

                    #ssim

                    ssim_left = self.SSIM(left_est,curr_tgt_image)
                    ssim_right = self.SSIM(right_est, right_curr_tgt_image)

                    ssim_loss_left = self.charbonnier_loss(ssim_left)
                    ssim_loss_right = self.charbonnier_loss(ssim_right)

                    #todo: have initialize alpha_image_loss
                    alpha_image_loss=0.85

                    image_loss_right =alpha_image_loss * ssim_loss_right + (1 - alpha_image_loss) *l1_reconstruction_loss_right
                    image_loss_left = alpha_image_loss * ssim_loss_left + (1 - alpha_image_loss) * l1_reconstruction_loss_left
                    image_loss+=(image_loss_left+image_loss_right)

                    #LR Consistency

                    lr_left_loss = self.charbonnier_loss(tf.abs(right_to_left_disp - disp_left_est[s]))
                    lr_right_loss = self.charbonnier_loss(tf.abs(left_to_right_disp - disp_right_est[s]))

                    lr_loss+=(lr_left_loss+lr_right_loss)

                    disp_left_smoothness = self.get_disparity_smoothness(disp_left_est[s], curr_tgt_image)
                    disp_right_smoothness = self.get_disparity_smoothness(disp_right_est[s],right_curr_tgt_image)

                    disp_left_loss = disp_left_smoothness/ (2**s)
                    disp_right_loss = disp_right_smoothness/ (2**s)
                    disp_gradient_loss+=(disp_left_loss + disp_right_loss)



                for i in range(opt.num_source):
                    # Inverse warp the source image to the target image frame
                    curr_proj_image = projective_inverse_warp(
                        curr_src_image_stack[:,:,:,3*i:3*(i+1)], 
                        tf.squeeze(pred_depth[s], axis=3), 
                        pred_poses[:,i,:], 
                        intrinsics[:,s,:,:])

                    curr_proj_error = tf.abs(curr_proj_image - curr_tgt_image)
                    ssim_proj_image =self.SSIM(curr_proj_image,curr_tgt_image)
                    ssim_loss_temp_loss=self.charbonnier_loss(ssim_proj_image)

                    pixel_loss += ((1-alpha_image_loss)*self.charbonnier_loss(curr_proj_error)+\
				   alpha_image_loss*ssim_loss_temp_loss)
                    # Prepare images for tensorboard summaries
                    if i == 0:
                        proj_image_stack = curr_proj_image
                        proj_error_stack = curr_proj_error
                       
                    else:
                        proj_image_stack = tf.concat([proj_image_stack, 
                                                      curr_proj_image], axis=3)
                        proj_error_stack = tf.concat([proj_error_stack, 
                                                      curr_proj_error], axis=3)
                tgt_image_all.append(curr_tgt_image)
                src_image_stack_all.append(curr_src_image_stack)
                proj_image_stack_all.append(proj_image_stack)
                proj_error_stack_all.append(proj_error_stack)
                
            total_loss = pixel_loss + smooth_loss + exp_loss +image_loss+(0.1*disp_gradient_loss)+lr_loss

        with tf.name_scope("train_op"):

            train_vars = [var for var in tf.trainable_variables()]

            self.global_step = tf.Variable(0, 
                                           name='global_step', 
                                           trainable=False)

            boundaries = [np.int32((3 / 5) * opt.max_steps), np.int32((4 / 5) * opt.max_steps)]
            values = [opt.learning_rate, opt.learning_rate / 2, opt.learning_rate / 4]
            learning_rate = tf.train.piecewise_constant(self.global_step, boundaries, values)

            optim = tf.train.AdamOptimizer(learning_rate, opt.beta1)
            self.train_op = slim.learning.create_train_op(total_loss, optim)

            self.incr_global_step = tf.assign(self.global_step, 
                                              self.global_step+1)

        # Collect tensors that are useful later (e.g. tf summary)
        self.pred_poses = pred_poses
        self.image_loss = image_loss
        self.pred_disp=disp_left_est
        self.lr_loss = lr_loss
        self.steps_per_epoch = loader.steps_per_epoch
        self.total_loss = total_loss
        self.pixel_loss = pixel_loss
        self.smooth_loss = disp_gradient_loss

        self.tgt_image_all = tgt_image_all
        self.src_image_stack_all = src_image_stack_all
        self.proj_image_stack_all = proj_image_stack_all
        self.proj_error_stack_all = proj_error_stack_all

    def charbonnier_loss(self,x, alpha=0.45, beta=1.0, epsilon=0.001):

        with tf.variable_scope('charbonnier_loss'):
            batch, height, width, channels = tf.unstack(tf.shape(x))
            #normalization = tf.cast(batch * height * width * channels, tf.float32)

            error = tf.pow(tf.square(x * beta) + tf.square(epsilon), alpha)

            return tf.reduce_mean(error)	

    def gradient_x(self, img):
        gx = img[:,:,:-1,:] - img[:,:,1:,:]
        return gx

    def gradient_y(self, img):
        gy = img[:,:-1,:,:] - img[:,1:,:,:]
        return gy
    def get_disparity_smoothness(self, disp, img):
        disp_gradients_x = self.gradient_x(disp)
        disp_gradients_y = self.gradient_y(disp)

        image_gradients_x =self.gradient_x(img)
        image_gradients_y = self.gradient_y(img)

        weights_x = tf.exp(-tf.reduce_mean(tf.abs(image_gradients_x), 3, keep_dims=True))
        weights_y = tf.exp(-tf.reduce_mean(tf.abs(image_gradients_y), 3, keep_dims=True))

        smoothness_x = disp_gradients_x * weights_x
        smoothness_y = disp_gradients_y * weights_y

        return self.charbonnier_loss(tf.abs(smoothness_x)) + self.charbonnier_loss(tf.abs(smoothness_y))

    def SSIM(self, x, y):
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

    def generate_image_left(self, img, disp):
        return bilinear_sampler_1d_h(img, -disp)

    def generate_image_right(self, img, disp):
        return bilinear_sampler_1d_h(img, disp)

    def calc_depth(self, disp, cam_mat, img_width):

        batch, height, width, _ = disp.get_shape().as_list()

        base_length = 0.54

        f = cam_mat[:, 0, 0]
        

        pred_depth = tf.stack([(base_length * f[i])/ (img_width * disp[i])
                               for i in range(batch)])

        return pred_depth


    def collect_summaries(self):
        opt = self.opt
        tf.summary.scalar("total_loss", self.total_loss)
        tf.summary.scalar("lr_loss", self.lr_loss)
        tf.summary.scalar("image_loss", self.image_loss)
        tf.summary.scalar("pixel_loss", self.pixel_loss)
        tf.summary.scalar("smooth_loss", self.smooth_loss)
        for s in range(opt.num_scales):
            tf.summary.image('scale%d_disparity_image' % s, self.pred_disp[s])
            tf.summary.image('scale%d_target_image' % s,self.tgt_image_all[s])
            for i in range(opt.num_source):
                tf.summary.image('scale%d_source_image_%d' % (s, i), self.src_image_stack_all[s][:, :, :, i*3:(i+1)*3])
                tf.summary.image('scale%d_projected_image_%d' % (s, i),self.proj_image_stack_all[s][:, :, :, i*3:(i+1)*3])
                tf.summary.image('scale%d_proj_error_%d' % (s, i),self.proj_error_stack_all[s][:,:,:,i*3:(i+1)*3])
        tf.summary.histogram("tx", self.pred_poses[:,:,0])
        tf.summary.histogram("ty", self.pred_poses[:,:,1])
        tf.summary.histogram("tz", self.pred_poses[:,:,2])
        tf.summary.histogram("qx", self.pred_poses[:,:,3])
        tf.summary.histogram("qy", self.pred_poses[:,:,4])
        tf.summary.histogram("qz", self.pred_poses[:,:,5])
        
    def train(self, opt):
        opt.num_source = opt.seq_length - 1
        # TODO: currently fixed to 4
        opt.num_scales = 4
        self.opt = opt
        self.build_train_graph()
        self.collect_summaries()

        with tf.name_scope("parameter_count"):
            parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) \
                                            for v in tf.trainable_variables()])

        self.saver = tf.train.Saver([var for var in tf.model_variables()] + \
                                    [self.global_step],
                                     max_to_keep=15)

        sv = tf.train.Supervisor(logdir=opt.checkpoint_dir, 
                                 save_summaries_secs=0, 
                                 saver=None)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with sv.managed_session(config=config) as sess:
            print('Trainable variables: ')
            for var in tf.trainable_variables():
                print(var.name)

            print("parameter_count =", sess.run(parameter_count))

            if opt.continue_train:
                if opt.init_checkpoint_file is None:
                    checkpoint = tf.train.latest_checkpoint(opt.checkpoint_dir)
                else:
                    checkpoint = opt.init_checkpoint_file
                print("Resume training from previous checkpoint: %s" % checkpoint)
                self.saver.restore(sess, checkpoint)

            start_time = time.time()
            for step in range(1, opt.max_steps):
                fetches = {
                    "train": self.train_op,
                    "global_step": self.global_step,
                    "incr_global_step": self.incr_global_step,
                    "loss":self.total_loss
               		}

                if step % opt.summary_freq == 0:
                    fetches["loss"] = self.total_loss
                    fetches["summary"] = sv.summary_op

                results = sess.run(fetches)
                print "loss:",results['loss']
                gs = results["global_step"]

                if step % opt.summary_freq == 0:

                    sv.summary_writer.add_summary(results["summary"], gs)
                    train_epoch = math.ceil(gs / self.steps_per_epoch)
                    train_step = gs - (train_epoch - 1) * self.steps_per_epoch

                    print("Epoch: [%2d] [%5d/%5d] time: %4.4f/it loss: %.3f" \
                            % (train_epoch, train_step, self.steps_per_epoch, \
                                (time.time() - start_time)/opt.summary_freq, 
                                results["loss"]))
                    start_time = time.time()

                if step % opt.save_latest_freq == 0:
                    self.save(sess, opt.checkpoint_dir, 'latest')

                if step % self.steps_per_epoch == 0:
                    self.save(sess, opt.checkpoint_dir, gs)

    def build_depth_test_graph(self):
        input_uint8 = tf.placeholder(tf.uint8, [self.batch_size, 
                    self.img_height, self.img_width, 3], name='raw_input')
        input_mc = self.preprocess_image(input_uint8)
        with tf.name_scope("depth_prediction"):
            pred_disp, depth_net_endpoints = disp_net(
                input_mc, is_training=False)
            pred_depth = [disp for disp in pred_disp]
        pred_depth = pred_depth[0]
        self.inputs = input_uint8
        self.pred_depth = pred_depth
        self.depth_epts = depth_net_endpoints

    def build_pose_test_graph(self):
        input_uint8 = tf.placeholder(tf.uint8, [self.batch_size, 
            self.img_height, self.img_width * self.seq_length, 3], 
            name='raw_input')
        input_mc = self.preprocess_image(input_uint8)
        loader = DataLoader(mode='test')
        tgt_image, src_image_stack = loader.batch_unpack_image_sequence(input_mc, self.img_height, self.img_width, self.num_source)

        with tf.name_scope("pose_prediction"):
            pred_poses,_ = pose_exp_net(tgt_image, src_image_stack,is_training=False)

            self.inputs = input_uint8
            self.pred_poses = pred_poses

    def preprocess_image(self, image):

        image = tf.image.convert_image_dtype(image, dtype=tf.float32)

        return image

    def deprocess_image(self, image):

        return tf.image.convert_image_dtype(image, dtype=tf.uint8)

    def setup_inference(self, 
                        img_height,
                        img_width,
                        mode,
                        seq_length=3,
                        batch_size=1):
        self.img_height = img_height
        self.img_width = img_width
        self.mode = mode
        self.batch_size = batch_size
        if self.mode == 'depth':
            self.build_depth_test_graph()
        if self.mode == 'pose':
            self.seq_length = seq_length
            self.num_source = seq_length - 1
            self.build_pose_test_graph()

    def inference(self, inputs, sess, mode='depth'):
        fetches = {}
        if mode == 'depth':
            fetches['depth'] = self.pred_depth
        if mode == 'pose':
            fetches['pose'] = self.pred_poses
        results = sess.run(fetches, feed_dict={self.inputs:inputs})
        return results

    def save(self, sess, checkpoint_dir, step):
        model_name = 'model'
        print(" [*] Saving checkpoint to %s..." % checkpoint_dir)
        if step == 'latest':
            self.saver.save(sess, 
                            os.path.join(checkpoint_dir, model_name + '.latest'))
        else:
            self.saver.save(sess, 
                            os.path.join(checkpoint_dir, model_name),
                            global_step=step)
