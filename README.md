# UnDEMoN
UnDEMoN: Unsupervised Depth and EgoMotion Network
Madhu Babu V, Kaushik Das, Anima Majumder and  Swagat Kumar
in IROS 2018 (oral)


## Prerequisites
This codebase was developed and tested with Tensorflow 1.0, CUDA 8.0 and Ubuntu 16.04.

## Preparing training data

We have defined an order that the data has to be preprocessed before training. 

For [KITTI](http://www.cvlibs.net/datasets/kitti/raw_data.php), first download the dataset using this [script](http://www.cvlibs.net/download.php?file=raw_data_downloader.zip) provided on the official website, and then run the `prepare_train_data_stereo.py` with the necessary arguments.


## Training
Once the data are formatted following the above instructions, you should be able to train the model by running the `train.py`.

You can then start a `tensorboard` session by

```bash
tensorboard --logdir=/path/to/tensorflow/log/files --port=8888
```
and visualize the training progress by opening [https://localhost:8888](https://localhost:8888) on your browser. If everything is set up properly, you should start seeing reasonable depth prediction. 

## Testing

Download the checkpoints from [here](https://drive.google.com/drive/folders/1bg55pMxO7fgLoyB-yb6pjtUNc6zD4pqT?usp=sharing) and use the `test_kitti_depth.py` and `test_kitt_pose.py` with necessary arguments. 

## Evaluation

We have used the depth evaluation scripts from [monodepth](https://github.com/mrharicot/monodepth) and the pose evalution scripts from the [SfMLearner](https://github.com/tinghuiz/SfMLearner/). 
We are very thankful the Monodepth and SfMLearner authors for their code bases.

## Reference
If you find our work useful in your research please consider citing our paper:
```
@inproceedings{babu2018undemon,
  title={UnDEMoN: Unsupervised Deep Network for Depth and Ego-Motion Estimation},
  author={Babu, V Madhu and Das, Kaushik and Majumdar, Anima and Kumar, Swagat},
  booktitle={2018 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages={1082--1088},
  year={2018},
  organization={IEEE}
}
```
## 👩‍⚖️ License

All rights are reserved and it is for non-commercial use only.
