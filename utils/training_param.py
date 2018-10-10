import argparse

parser = argparse.ArgumentParser(description='Parameters')

parser.add_argument('--mode',                      type=str,   help='train or test', default='train')
parser.add_argument('--continue_train',                      type=bool,   help='True or Flase', default=False)
parser.add_argument('--same_scale',                      type=bool,   help='True or Flase', default=False)
parser.add_argument('--model_name',                type=str,   help='model name', default='monodepth')
parser.add_argument('--img_height',              type=int,   help='input height', default=256)
parser.add_argument('--img_width',               type=int,   help='input width', default=512)
parser.add_argument('--beta1',               type=float,   help='input width', default=0.9)
parser.add_argument('--beta2',               type=float,   help='input width', default=0.99)
parser.add_argument('--dataset_dir',               type=str,  help="dataset directory",default=None)
parser.add_argument('--batch_size',                type=int,   help='batch size', default=4)
parser.add_argument('--max_steps',                type=int,   help='number of epochs', default=240000)
parser.add_argument('--learning_rate',             type=float, help='initial learning rate', default=1e-4)
parser.add_argument('--lr_loss_weight',            type=float, help='left-right consistency weight', default=1.0)
parser.add_argument('--alpha_image_loss',          type=float, help='weight between SSIM and L1 in the image loss', default=0.85)
parser.add_argument('--exp_reg_weight',          type=float, help='exp_mask_weight', default=0.1)
parser.add_argument('--census_weight',          type=float, help='exp_mask_weight', default=0.1)
parser.add_argument('--gradient_loss_weight', type=float, help='disparity smoothness weigth', default=0.1)
parser.add_argument('--do_stereo',                             help='if set, will train the stereo model', action='store_true')
parser.add_argument('--wrap_mode',                 type=str,   help='bilinear sampler wrap mode, edge or border', default='border')
parser.add_argument('--num_gpus',                  type=int,   help='number of GPUs to use for training', default=1)
parser.add_argument('--num_scales',               type=int,   help='number of threads to use for data loading', default=4)
parser.add_argument('--output_directory',          type=str,   help='output directory for test disparities, if empty outputs to checkpoint folder', default='checkpoints')
parser.add_argument('--checkpoint_dir',             type=str,   help='directory to save checkpoints and summaries', default='checkpoints')
parser.add_argument('--init_checkpoint_file',           type=str,   help='path to a specific checkpoint to load', default=None)
parser.add_argument('--summary_freq',                 type=int,         help='summary frequence',default=100)


parser.add_argument("--flow_warp_weight",      type=float ,     default=1.0,    help="Weight for warping by full flow")
parser.add_argument("--flow_smooth_weight",  type=float ,      default=   0.2,    help="Weight for flow smoothness")
parser.add_argument("--flow_consistency_weight",type=float ,   default=   0.2,   help= "Weight for bidirectional flow consistency")
parser.add_argument("--flow_consistency_alpha", type=float ,   default=   3.0,   help= "Alpha for flow consistency check")
parser.add_argument("--flow_consistency_beta",  type=float ,   default=  0.05,   help= "Beta for flow consistency check")

opt = parser.parse_args()

opt.num_source=2
