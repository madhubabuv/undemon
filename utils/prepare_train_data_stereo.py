from __future__ import division
import argparse
import scipy.misc
import numpy as np
from glob import glob
from joblib import Parallel, delayed
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir", type=str, required=True, help="where the dataset is stored")
parser.add_argument("--dataset_name", type=str, required=True, choices=["kitti_raw_eigen", "kitti_raw_stereo", "kitti_odom", "cityscapes"])
parser.add_argument("--dump_root", type=str, required=True, help="Where to dump the data")
parser.add_argument("--seq_length", type=int, required=True, help="Length of each training sequence")
parser.add_argument("--img_height", type=int, default=256, help="image height")
parser.add_argument("--img_width", type=int, default=512, help="image width")
parser.add_argument("--num_threads", type=int, default=7, help="number of threads to use")
args = parser.parse_args()

print args.dump_root

def concat_image_seq(seq):
    for i, im in enumerate(seq):
        if i == 0:
            res = im
        else:
            res = np.hstack((res, im))
    return res

def dump_example(n,args):
    print("frame_id%d"%n)
    if n % 200 == 0:
        print('Progress %d/%d....' % (n, data_loader.num_train))
    #print "target_id",n
    stereo_example = data_loader.get_train_example_with_idx(n)
    if stereo_example == False:
        return
    for example in stereo_example:
        image_seq = concat_image_seq(example['image_seq'])
        intrinsics = example['intrinsics']
        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        cx = intrinsics[0, 2]
        cy = intrinsics[1, 2]

        dump_dir = os.path.join(args.dump_root, example['folder_name'],example['camera_id'])

        if not os.path.isdir(dump_dir):
            os.makedirs(dump_dir)
        #try:
        #    os.makedirs(args.dump_root)
        #except OSError:
        #    if not os.path.isdir(args.dump_root):
        #        raise
        dump_img_file = dump_dir + '/%s.jpg' % example['file_name']
        scipy.misc.imsave(dump_img_file, image_seq.astype(np.uint8))
        dump_cam_file = dump_dir + '/%s_cam.txt' % example['file_name']
        with open(dump_cam_file, 'w') as f:
            f.write('%f,0.,%f,0.,%f,%f,0.,0.,1.' % (fx, cx, fy, cy))

def main():
    if not os.path.exists(args.dump_root):
        os.makedirs(args.dump_root)

    global data_loader
    if args.dataset_name == 'kitti_odom':
        from kitti.kitti_odom_loader import kitti_odom_loader
        data_loader = kitti_odom_loader(args.dataset_dir,
                                        img_height=args.img_height,
                                        img_width=args.img_width,
                                        seq_length=args.seq_length)

    if args.dataset_name == 'kitti_raw_eigen':
        from kitti.kitti_raw_loader_stereo import kitti_raw_loader
        data_loader = kitti_raw_loader(args.dataset_dir,
                                       split='eigen',
                                       img_height=args.img_height,
                                       img_width=args.img_width,
                                       seq_length=args.seq_length)

    if args.dataset_name == 'kitti_raw_stereo':
        from kitti.kitti_raw_loader_stereo import kitti_raw_loader
        data_loader = kitti_raw_loader(args.dataset_dir,
                                       split='stereo',
                                       img_height=args.img_height,
                                       img_width=args.img_width,
                                       seq_length=args.seq_length)        

    if args.dataset_name == 'cityscapes':
        from cityscapes.cityscapes_loader import cityscapes_loader
        data_loader = cityscapes_loader(args.dataset_dir,
                                        img_height=args.img_height,
                                        img_width=args.img_width,
                                        seq_length=args.seq_length)

    print(data_loader.num_train)

    Parallel(n_jobs=args.num_threads)(delayed(dump_example)(n,args) for n in range(data_loader.num_train-1))

    # Split into train/val
    np.random.seed(8964)
    subfolders = os.listdir(args.dump_root)
    with open(args.dump_root + 'train.txt', 'w') as tf:
        with open(args.dump_root + 'val.txt', 'w') as vf:
            for s in subfolders:
                if not os.path.isdir(args.dump_root + '/%s' % s):
                    continue
                os.chdir(args.dump_root)
                left_imfiles = glob(s+'/image_02/*.jpg')
                right_imfiles = glob(s+'/image_03/*.jpg')
                for left,right in zip(left_imfiles,right_imfiles):
                    #print left
                    if np.random.random()<0.1:
                        vf.write(left+' '+right+'\n')
                    else:
                        tf.write(left+' '+right+'\n')

main()

