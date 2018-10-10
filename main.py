from __future__ import division
import abc
from enum import Enum
import argparse
import tensorflow as tf
slim = tf.contrib.slim
import math
import time
import os
import numpy as np
from utils.training_param import opt
from utils.average_gradients import average_gradients



class Mode(Enum):
    TRAIN = 1
    TEST = 2

class Net(object):

    _metaclass__ = abc.ABCMeta

    def __init__(self):

        self.steps_per_epoch=10000

    @abc.abstractmethod
    def load_data(self):

        "loads data"

        return

    @abc.abstractmethod
    def model(self,inputs,**kwargs):
        """
        Defines the model and returns a tuple of Tensors including predictions and losses.
        """
        return

    @abc.abstractmethod
    def loss(self,inputs,predictions,**kwargs):
        """
        Defines the model and returns a tuple losses.
        """
        return

    @abc.abstractmethod
    def build_summaries(self,inputs,outputs):

        """
        Builds all summaries
        :return:
        """


    def train(self):

        with tf.name_scope("train_op"):

            self.global_step = tf.Variable(0,
                                           name='global_step',
                                           trainable=False)


            boundaries = [np.int32((3 / 5) * opt.max_steps), np.int32((4 / 5) * opt.max_steps)]
            values = [opt.learning_rate, opt.learning_rate / 2, opt.learning_rate / 4]
            learning_rate = tf.train.piecewise_constant(self.global_step, boundaries, values)

            tf.summary.scalar('learning_rate',learning_rate)


            optim = tf.train.AdamOptimizer(learning_rate, opt.beta1)


        tower_grads  = []
        tower_losses = []
        reuse_variables = False
        with tf.variable_scope(tf.get_variable_scope()):

            inputs=self.load_data(opt.num_gpus)

            for gpu_id in range(opt.num_gpus):
                with tf.device('/gpu:%d' % gpu_id):

                    model_inputs={'left_target_image':inputs['left_target_split'][gpu_id],'right_target_image':inputs['right_target_split'][gpu_id],
                                  'left_source_images':inputs['left_source_image_split'][gpu_id]}

                    predictions=self.model(model_inputs,reuse_variables)

                    print "model Loaded"

                    loss_inputs={
                        'left_target_image': inputs['left_target_split'][gpu_id],
                        'left_source_images': inputs['left_source_image_split'][gpu_id],
                        'right_target_image':inputs['right_target_split'][gpu_id],
                        #'right_source_images':inputs['right_source_image_split'][gpu_id],
                        'intrinsics':inputs['intrinsics_split'][gpu_id]

                    }

                    loss=self.loss(loss_inputs,predictions,reuse_variables)

                    print "losses calculated"

                    if not reuse_variables:

                        self.build_summaries(loss_inputs,predictions)

                    reuse_variables=True

                    tower_losses.append(loss)

                    grads = optim.compute_gradients(loss)

                    tower_grads.append(grads)

            grads = average_gradients(tower_grads)

            print "optimizing"

            apply_gradient_op = optim.apply_gradients(grads, global_step=self.global_step)

            total_loss = tf.reduce_mean(tower_losses)

            tf.summary.scalar('total_loss',total_loss)

            self.incr_global_step = tf.assign(self.global_step,
                                              self.global_step+1)

        with tf.name_scope("parameter_count"):
            parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) \
                                             for v in tf.trainable_variables()])
        #self.saver = tf.train.Saver([var for var in tf.trainable_variables()] + \
        #                            [self.global_step],
        #                            max_to_keep=15)

        self.saver=tf.train.Saver(tf.trainable_variables())
        sv = tf.train.Supervisor(logdir=opt.checkpoint_dir,
                                 save_summaries_secs=0,
                                 saver=None)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement=True
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
            for step in range(1, opt.max_steps+1):
                fetches = {
                    "train": apply_gradient_op,
                    "global_step": self.global_step,
                    #'incr_global_step':self.incr_global_step,
                    "loss": total_loss
                }

                if step % opt.summary_freq == 0:
                    fetches["loss"] = total_loss
                    fetches["summary"] = sv.summary_op

                results = sess.run(fetches)
                print "loss:", results['loss']
                gs = results["global_step"]

                if step % opt.summary_freq == 0:
                    sv.summary_writer.add_summary(results["summary"], gs)
                    train_epoch = math.ceil(gs / self.steps_per_epoch)
                    train_step = gs - (train_epoch - 1) * self.steps_per_epoch
                    print("Epoch: [%2d] [%5d/%5d] time: %4.4f/it loss: %.3f" \
                          % (train_epoch, train_step, self.steps_per_epoch, \
                             (time.time() - start_time) / opt.summary_freq,
                             results["loss"]))
                    start_time = time.time()

                if step % self.steps_per_epoch == 0:
                    self.save(sess, opt.checkpoint_dir, gs)

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

