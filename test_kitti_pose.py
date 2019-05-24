from __future__ import division
import os
import math
import scipy.misc
import tensorflow as tf
import numpy as np
from glob import glob
from UnDEMoN import UnDEMoN
from kitti_eval.pose_evaluation_utils import dump_pose_seq_TUM

flags = tf.app.flags
flags.DEFINE_integer("batch_size", 1, "The size of of a sample batch")
flags.DEFINE_integer("img_height", 256, "Image height")
flags.DEFINE_integer("img_width", 512, "Image width")
flags.DEFINE_integer("seq_length", 5, "Sequence length for each example")
flags.DEFINE_integer("test_seq", 9, "Sequence id to test")
flags.DEFINE_string("dataset_dir", None, "Dataset directory")
flags.DEFINE_string("output_dir", None, "Output directory")
flags.DEFINE_string("ckpt_file", None, "checkpoint file")
FLAGS = flags.FLAGS

test_data={}

test_data[0]="2011_10_03_drive_0027 000000 004540"
test_data[1]="2011_10_03_drive_0042 000000 001100"
test_data[2]= "2011_10_03_drive_0034 000000 004660"
test_data[3]= "2011_09_26_drive_0067 000000 000800"
test_data[4]= "2011_09_30_drive_0016 000000 000270"
test_data[5]= "2011_09_30_drive_0018 000000 002760"
test_data[6]= "2011_09_30_drive_0020 000000 001100"
test_data[7]= "2011_09_30_drive_0027 000000 001100"
test_data[8]= "2011_09_30_drive_0028 001100 005170"
test_data[9]= "2011_09_30_drive_0033 000000 001590"
test_data[10]= "2011_09_30_drive_0034 000000 001200"

def load_image_sequence(dataset_dir, 
                        frames, 
                        tgt_idx,seq_name,	
                        seq_length, 
                        img_height, 
                        img_width):
    half_offset = int((seq_length - 1)/2)
    for o in range(-half_offset, half_offset+1):
        curr_idx = tgt_idx + o
        curr_drive, curr_frame_id = frames[curr_idx].split(' ')
        #img_file = os.path.join(
        #    dataset_dir, 'sequences', '%s/image_2/%s.png' % (curr_drive, curr_frame_id))
        img_file=os.path.join(dataset_dir+seq_name[:10]+"/"+seq_name+"_sync/image_02/data/",'%s.png'%(curr_frame_id))
        curr_img = scipy.misc.imread(img_file)
        curr_img = scipy.misc.imresize(curr_img, (img_height, img_width))
        if o == -half_offset:
            image_seq = curr_img
        else:
            image_seq = np.hstack((image_seq, curr_img))
    return image_seq

def is_valid_sample(frames, tgt_idx, seq_length):
    N = len(frames)
    tgt_drive, _ = frames[tgt_idx].split(' ')
    max_src_offset = int((seq_length - 1)/2)
    min_src_idx = tgt_idx - max_src_offset
    max_src_idx = tgt_idx + max_src_offset
    if min_src_idx < 0 or max_src_idx >= N:
        return False
    # TODO: unnecessary to check if the drives match
    min_src_drive, _ = frames[min_src_idx].split(' ')
    max_src_drive, _ = frames[max_src_idx].split(' ')
    if tgt_drive == min_src_drive and tgt_drive == max_src_drive:
        return True
    return False

def main():
    demon = UnDEMoN()
    demon.setup_inference(FLAGS.img_height,
                        FLAGS.img_width,
                        'pose',
                        FLAGS.seq_length)
    saver = tf.train.Saver([var for var in tf.trainable_variables()]) 
    if not os.path.isdir(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)
    #seq_dir = os.path.join(FLAGS.dataset_dir, 'sequences', '%.2d' % FLAGS.test_seq)
    seq_name=test_data[FLAGS.test_seq].split(' ')[0]
    seq_dir=FLAGS.dataset_dir+seq_name[:10]+"/"+seq_name+"_sync/"
    img_dir = os.path.join(seq_dir, 'image_02','data')
    N = len(glob(img_dir + '/*.png')[:4540])
    test_frames = ['%.2d %.10d' % (FLAGS.test_seq, n) for n in range(N)]
    with open(seq_dir+'image_02/time.txt', 'r') as f:
        times = f.readlines()
    times = np.array([(s[:-1]) for s in times])
    max_src_offset = (FLAGS.seq_length - 1)//2
    with tf.Session() as sess:
        saver.restore(sess, FLAGS.ckpt_file)
        print "Model Loaded"
        for tgt_idx in range(N):
            if not is_valid_sample(test_frames, tgt_idx, FLAGS.seq_length):
                continue
            if tgt_idx % 100 == 0:
                print('Progress: %d/%d' % (tgt_idx, N))
            # TODO: currently assuming batch_size = 1
            image_seq = load_image_sequence(FLAGS.dataset_dir, 
                                            test_frames, 
                                            tgt_idx,seq_name, 
                                            FLAGS.seq_length, 
                                            FLAGS.img_height, 
                                            FLAGS.img_width)
            pred = demon.inference(image_seq[None, :, :, :], sess, mode='pose')
            pred_poses = pred['pose'][0]
            #print pred_poses
            # Insert the target pose [0, 0, 0, 0, 0, 0] 
            pred_poses = np.insert(pred_poses, max_src_offset, np.zeros((1,6)), axis=0)
            curr_times = times[tgt_idx - max_src_offset:tgt_idx + max_src_offset + 1]
            out_file = FLAGS.output_dir + '%.6d.txt' % (tgt_idx - max_src_offset)
            dump_pose_seq_TUM(out_file, pred_poses, curr_times)

main()


