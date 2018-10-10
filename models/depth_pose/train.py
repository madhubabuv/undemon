from __future__ import division
from depth_pose import DepthPose
from dataloaders.dataloader import DataLoader
from utils.utils import *
from utils.training_param import *


class UnDeMon(DepthPose):

    def __init__(self, opt):
        self.args=opt

        self.loader = DataLoader(opt.dataset_dir, opt.batch_size,
                                 opt.mode,opt.img_height,
                                 opt.img_width,opt.num_scales,opt.num_source)

        super(UnDeMon, self).__init__(opt)

    def load_data(self,num_gpus):

        with tf.name_scope("data_loading"):

            train_batch = self.loader.load_train_batch()
            left_tgt_image, left_src_image_stack, intrinsics = train_batch['left']
            right_tgt_image, right_src_image_stack, _ = train_batch['right']

            left_target_image = preprocess_image(left_tgt_image)
            left_source_image_stack = preprocess_image(left_src_image_stack)

            right_target_image = preprocess_image(right_tgt_image)


            #split the inputs as per the gpus

            left_target_image_split=tf.split(left_target_image,self.args.num_gpus)

            right_target_image_split=tf.split(right_target_image,self.args.num_gpus)

            left_source_image_stack_split=tf.split(left_source_image_stack,self.args.num_gpus)

            intrinsics_split=tf.split(intrinsics,self.args.num_gpus)


            return {
                'left_target_split':left_target_image_split,
                'right_target_split':right_target_image_split,
                'left_source_image_split':left_source_image_stack_split,
                'intrinsics_split':intrinsics_split
            }




if __name__=="__main__":

    model_obj=UnDeMon(opt)

    model_obj.train()


