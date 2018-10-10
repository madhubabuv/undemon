from __future__ import division
from nets.depth_nets.DepthNet import disp_net
from utils.utils import *
import numpy as np
import tensorflow as tf
import PIL.Image as pil
import os

flags = tf.app.flags
flags.DEFINE_integer("batch_size", 4, "The size of of a sample batch")
flags.DEFINE_integer("img_height", 256, "Image height")
flags.DEFINE_integer("img_width", 512, "Image width")
flags.DEFINE_string("dataset_dir", None, "Dataset directory")
flags.DEFINE_string("output_dir", None, "Output directory")
flags.DEFINE_string("ckpt_file", None, "checkpoint file")
flags.DEFINE_string('test_file',None,'filename of test files')
FLAGS = flags.FLAGS


if __name__=="__main__":


    with open(FLAGS.test_file,'r') as f:
        test_files = f.readlines()
        test_files = [FLAGS.dataset_dir + t.split(' ')[0][:-1] for t in test_files]

    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)

    basename = os.path.basename(FLAGS.ckpt_file)
    output_file = FLAGS.output_dir + '/' + basename

    input_uint8 = tf.placeholder(tf.uint8, [FLAGS.batch_size,
                                            FLAGS.img_height,
                                            FLAGS.img_width, 3], name='raw_input')
    input_mc = preprocess_image(input_uint8)

    #with tf.variable_scope('undemon_flow'):
    #with tf.name_scope("depth_prediction"):
    pred_disp = disp_net(input_mc, is_training=False)
    pred_depth = [disp for disp in pred_disp[0]]
    saver = tf.train.Saver([var for var in tf.model_variables()])
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        saver.restore(sess, FLAGS.ckpt_file)
        pred_all = []
        for t in range(0, len(test_files), FLAGS.batch_size):
            if t % 100 == 0:
                print('processing %s: %d/%d' % (basename, t, len(test_files)))

            inputs = np.zeros((FLAGS.batch_size,
                               FLAGS.img_height,
                               FLAGS.img_width, 3),dtype=np.uint8)

            for b in range(FLAGS.batch_size):
                idx = t + b
                if idx >= len(test_files):
                    break
                fh = open(test_files[idx], 'r')
                raw_im = pil.open(fh)
                scaled_im = raw_im.resize((FLAGS.img_width, FLAGS.img_height), pil.ANTIALIAS)

                inputs[b] = np.array(scaled_im)

            pred = sess.run(pred_depth, feed_dict={input_uint8: inputs})

            for b in range(FLAGS.batch_size):
                idx = t + b
                if idx >= len(test_files):
                    break
                pred_all.append(pred[0][b,:,:,0])

        np.save(output_file, pred_all)