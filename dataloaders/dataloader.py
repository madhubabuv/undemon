from __future__ import division
import os
import random
import tensorflow as tf

class DataLoader(object):
    def __init__(self, 
                 dataset_dir=None, 
                 batch_size=None,
                 mode=None, 
                 img_height=None, 
                 img_width=None,
                 num_scales=None,num_source=None,):
        self.mode=mode
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width = img_width
        self.num_source = num_source
        self.num_scales = num_scales


    def load_images(self,image_paths_queue):

        # Load images
        img_reader = tf.WholeFileReader()
        _, image_contents = img_reader.read(image_paths_queue)
        image_seq = tf.image.decode_jpeg(image_contents)
        tgt_image, src_image_stack =self.unpack_image_sequence(
                image_seq, self.img_height, self.img_width, self.num_source)

        return tgt_image,src_image_stack

    def load_intrinsics(self,cam_paths_queue):

        # Load camera intrinsics
        cam_reader = tf.TextLineReader()
        _, raw_cam_contents = cam_reader.read(cam_paths_queue)
        rec_def = []
        for i in range(9):
            rec_def.append([1.])
        raw_cam_vec = tf.decode_csv(raw_cam_contents,
                                    record_defaults=rec_def)
        raw_cam_vec = tf.stack(raw_cam_vec)
        intrinsics = tf.reshape(raw_cam_vec, [3, 3])

        return intrinsics

    def augment_image_pair(self, left_image, right_image):

        # randomly shift gamma
        random_gamma = tf.random_uniform([], 0.8, 1.2)
        left_image=tf.image.convert_image_dtype(left_image, tf.float32)
        right_image = tf.image.convert_image_dtype(right_image, tf.float32)

        left_image_aug  = left_image  ** random_gamma
        right_image_aug = right_image ** random_gamma

        # randomly shift brightness
        random_brightness = tf.random_uniform([], 0.5, 2.0)
        left_image_aug  =  left_image_aug * random_brightness
        right_image_aug = right_image_aug * random_brightness

        # randomly shift color
        random_colors = tf.random_uniform([(self.num_source+1)*3], 0.8, 1.2)
        white = tf.ones([tf.shape(left_image)[0], tf.shape(left_image)[1]])
        color_image = tf.stack([white * random_colors[i] for i in range((self.num_source+1)*3)], axis=2)

        left_image_aug  *= color_image
        right_image_aug *= color_image

        # saturate
        #left_image_aug  = tf.clip_by_value(left_image_aug,  0, 1)
        #right_image_aug = tf.clip_by_value(right_image_aug, 0, 1)

        left_image=tf.image.convert_image_dtype(left_image, tf.uint8)
        right_image = tf.image.convert_image_dtype(right_image, tf.uint8)


        return left_image, right_image

    def load_train_batch(self):

        """Load a batch of training instances.
        """
        seed = random.randint(0, 2**31 - 1)

        # Load the list of training files into queues
        file_list = self.format_file_list(self.dataset_dir, 'train')

        left_image_paths_queue = tf.train.string_input_producer(file_list['left'],shuffle=False)
        right_image_paths_queue = tf.train.string_input_producer(file_list['right'],shuffle=False)
        cam_paths_queue = tf.train.string_input_producer(file_list['cam'],shuffle=False)
        
        self.steps_per_epoch = int(len(file_list['left'])//self.batch_size)

        left_target_files,left_src_files=self.load_images(left_image_paths_queue)

        right_target_files,right_src_files=self.load_images(right_image_paths_queue)

        intrinsics=self.load_intrinsics(cam_paths_queue)

        if self.mode=='train':
            
            # data Augmentation
            left_image_all = tf.concat([left_target_files, left_src_files], 2)
            right_image_all = tf.concat([right_target_files, right_src_files], 2)

            # randomly flip images
            do_flip = tf.random_uniform([], 0, 1)
            left_image = tf.cond(do_flip > 0.5, lambda: tf.image.flip_left_right(right_image_all), lambda: left_image_all)
            right_image = tf.cond(do_flip > 0.5, lambda: tf.image.flip_left_right(left_image_all), lambda: right_image_all)

            # randomly augment images colors,gamma ratio
            do_augment = tf.random_uniform([], 0, 1)
            left_image, right_image = tf.cond(do_augment > 0.5,
                                              lambda: self.augment_image_pair(left_image, right_image),
                                              lambda: (left_image, right_image))

            #src_1_augment = tf.random_uniform([], 0, 1)


            #src_2_augment = tf.random_uniform([], 0, 1)


            #left_target_files,left_src_files=tf.cond(src_1_augment>0.5,lambda: (left_image[:,:,3:6],tf.concat((left_image[:,:,:3],left_image[:,:,6:]),2)),lambda:(left_image[:,:,:3], left_image[:,:,3:]))

            #right_target_files,right_src_files=tf.cond(src_1_augment>0.5,lambda:(right_image[:,:,3:6],tf.concat((right_image[:,:,:3],right_image[:,:,6:]),2)),lambda:(right_image[:,:,:3], right_image[:,:,3:]))


            left_target_files,left_src_files=left_image[:,:,:3], left_image[:,:,3:]

            right_target_files,right_src_files=right_image[:,:,:3], right_image[:,:,3:]
        

        min_after_dequeue = 2048

        capacity = min_after_dequeue + 4 * self.batch_size

        # Form training batches
        left_src_image_stack, left_tgt_image, left_intrinsics,\
        right_src_image_stack, right_tgt_image, right_intrinsics = \
            tf.train.shuffle_batch([left_src_files, left_target_files, intrinsics,
                                    right_src_files, right_target_files, intrinsics],
                                  self.batch_size, capacity, min_after_dequeue,4)
        
        left_intrinsics = self.get_multi_scale_intrinsics(
            left_intrinsics, self.num_scales)
        right_intrinsics = self.get_multi_scale_intrinsics(
            right_intrinsics, self.num_scales)
        training_batch={}

        training_batch['left']=(left_tgt_image,left_src_image_stack,left_intrinsics)
        training_batch['right']=(right_tgt_image,right_src_image_stack,right_intrinsics)
        return training_batch

    def unpack_image_sequence(self, image_seq, img_height, img_width, num_source):

        # Assuming the center image is the target frame
        tgt_start_idx = int(img_width * (num_source//2))
        tgt_image = tf.slice(image_seq, 
                             [0, tgt_start_idx, 0], 
                             [-1, img_width, -1])
        # Source frames before the target frame
        src_image_1 = tf.slice(image_seq, 
                               [0, 0, 0], 
                               [-1, int(img_width * (num_source//2)), -1])
        # Source frames after the target frame
        src_image_2 = tf.slice(image_seq, 
                               [0, int(tgt_start_idx + img_width), 0], 
                               [-1, int(img_width * (num_source//2)), -1])
        src_image_seq = tf.concat([src_image_1, src_image_2], axis=1)
        # Stack source frames along the color channels (i.e. [H, W, N*3])
        src_image_stack = tf.concat([tf.slice(src_image_seq, 
                                    [0, i*img_width, 0], 
                                    [-1, img_width, -1]) 
                                    for i in range(num_source)], axis=2)
        src_image_stack.set_shape([img_height, 
                                   img_width, 
                                   num_source * 3])
        tgt_image.set_shape([img_height, img_width, 3])


        return tgt_image, src_image_stack


    def format_file_list(self, data_root, split):
        with open(data_root + '/%s.txt' % split, 'r') as f:
            frames = f.readlines()


        left_subfolders = [x.split(' ')[0] for x in frames]
        right_subfolders = [x.split(' ')[1] for x in frames]

        left_path=[os.path.join(data_root,left_subfolders[i][:-4]) for i in range(len(left_subfolders))]
        left_image_file_list = [path+'.jpg' for path in left_path]
        right_path=[os.path.join(data_root,right_subfolders[i][:-5]) for i in range(len(left_subfolders))]
        right_image_file_list = [path+'.jpg' for path in right_path]

        cam_file_list = [path+ '_cam.txt' for path in left_path]

        all_list = {}
        all_list['left'] = left_image_file_list
        all_list['right'] = right_image_file_list
        all_list['cam'] = cam_file_list
        return all_list

    def make_intrinsics_matrix(self, fx, fy, cx, cy):
        # Assumes batch input
        batch_size = fx.get_shape().as_list()[0]
        zeros = tf.zeros_like(fx)
        r1 = tf.stack([fx, zeros, cx], axis=1)
        r2 = tf.stack([zeros, fy, cy], axis=1)
        r3 = tf.constant([0.,0.,1.], shape=[1, 3])
        r3 = tf.tile(r3, [batch_size, 1])
        intrinsics = tf.stack([r1, r2, r3], axis=1)

        return intrinsics

    def data_augmentation(self, im, intrinsics, out_h, out_w):
        # Random scaling
        def random_scaling(im, intrinsics):
            batch_size, in_h, in_w, _ = im.get_shape().as_list()
            scaling = tf.random_uniform([2], 1, 1.15)
            x_scaling = scaling[0]
            y_scaling = scaling[1]
            im=tf.image.convert_image_dtype(im, tf.float32)
            out_h = tf.cast(in_h * y_scaling, dtype=tf.int32)
            out_w = tf.cast(in_w * x_scaling, dtype=tf.int32)
            im = tf.image.resize_area(im, [out_h, out_w])
            fx = intrinsics[:,0,0] * x_scaling
            fy = intrinsics[:,1,1] * y_scaling
            cx = intrinsics[:,0,2] * x_scaling
            cy = intrinsics[:,1,2] * y_scaling
            intrinsics = self.make_intrinsics_matrix(fx, fy, cx, cy)
            im=tf.reshape(im,(batch_size,in_h,in_w,_))
            return im, intrinsics

        # Random cropping
        def random_cropping(im, intrinsics, out_h, out_w):
            # batch_size, in_h, in_w, _ = im.get_shape().as_list()
            batch_size, in_h, in_w, _ = tf.unstack(tf.shape(im))
            offset_y = tf.random_uniform([1], 0, in_h - out_h + 1, dtype=tf.int32)[0]
            offset_x = tf.random_uniform([1], 0, in_w - out_w + 1, dtype=tf.int32)[0]
            im=tf.image.convert_image_dtype(im, tf.float32)
            
            im = tf.image.crop_to_bounding_box(
                im, offset_y, offset_x, out_h, out_w)
            fx = intrinsics[:,0,0]
            fy = intrinsics[:,1,1]
            cx = intrinsics[:,0,2] - tf.cast(offset_x, dtype=tf.float32)
            cy = intrinsics[:,1,2] - tf.cast(offset_y, dtype=tf.float32)
            intrinsics = self.make_intrinsics_matrix(fx, fy, cx, cy)
            return im, intrinsics
        #im, intrinsics = random_scaling(im, intrinsics)
        #im, intrinsics = random_cropping(im, intrinsics, out_h, out_w)
        im = tf.cast(im, dtype=tf.uint8)
        return im, intrinsics
	




    def batch_unpack_image_sequence(self, image_seq, img_height, img_width, num_source):
        # Assuming the center image is the target frame
        tgt_start_idx = int(img_width * (num_source//2))
        tgt_image = tf.slice(image_seq, 
                             [0, 0, tgt_start_idx, 0], 
                             [-1, -1, img_width, -1])
        # Source frames before the target frame
        src_image_1 = tf.slice(image_seq, 
                               [0, 0, 0, 0], 
                               [-1, -1, int(img_width * (num_source//2)), -1])
        # Source frames after the target frame
        src_image_2 = tf.slice(image_seq, 
                               [0, 0, int(tgt_start_idx + img_width), 0], 
                               [-1, -1, int(img_width * (num_source//2)), -1])
        src_image_seq = tf.concat([src_image_1, src_image_2], axis=2)
        # Stack source frames along the color channels (i.e. [B, H, W, N*3])
        src_image_stack = tf.concat([tf.slice(src_image_seq, 
                                    [0, 0, i*img_width, 0], 
                                    [-1, -1, img_width, -1]) 
                                    for i in range(num_source)], axis=3)
        return tgt_image, src_image_stack

    def get_multi_scale_intrinsics(self, intrinsics, num_scales):
        intrinsics_mscale = []
        # Scale the intrinsics accordingly for each scale
        for s in range(num_scales):
            fx = intrinsics[:,0,0]/(2 ** s)
            fy = intrinsics[:,1,1]/(2 ** s)
            cx = intrinsics[:,0,2]/(2 ** s)
            cy = intrinsics[:,1,2]/(2 ** s)
            intrinsics_mscale.append(
                self.make_intrinsics_matrix(fx, fy, cx, cy))
        intrinsics_mscale = tf.stack(intrinsics_mscale, axis=1)

        return intrinsics_mscale
