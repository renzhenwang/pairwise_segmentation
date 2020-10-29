# Pairwise Segmentation
This code is an implementation of [pairwise semantic segmentation via conjugate fully convolutional network](https://link.springer.xilesou.top/chapter/10.1007/978-3-030-32226-7_18) and its journal version, which is implemented based on [pytorch-deeplab-xception](https://github.com/jfzhang95/pytorch-deeplab-xception/tree/previous) and experiments with the Liver Tumor Segmentation (LiTS) and Combined (CT-MR) Healthy Abdominal Organ Segmentation (CHAOS) datasets.

# Get Started
## 1. Data preprocessing
Download dataset from [LiTS](https://competitions.codalab.org/competitions/17094), and put 131 training data with segmentation masks under '../lits/training/'. Split the dataset to 5% proportions of training samples, and 20% validation samples:
```
python prepare_lits.py --data_path '../lits/training/' --pre_data_result '../lits/minor_lits/' --val_num 26 --semi_num 98
```
Note one can customize the path 'pre_data_result', but should keep the path name consistent in the file 'mypath.py'.

## 2. Training and validation
To tain the CFCN/C2FCN (replace main_cfcn.py with main_c2fcn.py) as follows:
```
python main_cfcn.py --dataset pairwise_lits --gpu-ids 0,1 --crop-size 512 --base-size 512 --loss-type pairwise_loss --batch-size 16
```
Once the model is trained, validate it as running:
```
python main_cfcn.py --dataset pairwise_lits --gpu-ids 0 --crop-size 512 --base-size 512 --loss-type pairwise_loss --only-val --save-predict --batch-size 1 --resume './run/pairwise_lits/pairwise_deeplab-resnet18/experiment_0/checkpoint.pth.tar'
```

## 3. Performance evaluation
Evaluate the segmentation performance with an average of Dice per volume score (Dice-per-case) and a global Dice score (Dice-global) 
```
python lits_output_stats.py --output-dir './run/pairwise_lits/pairwise_deeplab-resnet18/experiment_1/predict_mask/ --is_val --pred-dir './run/pairwise_lits/pairwise_deeplab-resnet18/experiment_1/result/' --data_root '../lits/training/'
```

# Citation
If the ropository is useful for your research, please consider citing:
```
@inproceedings{wang2019pairwise,
  title={Pairwise Semantic Segmentation via Conjugate Fully Convolutional Network},
  author={Wang, Renzhen and Cao, Shilei and Ma, Kai and Meng, Deyu and Zheng, Yefeng},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={157--165},
  year={2019},
  organization={Springer}
}
```

# Questions
Please contact 'wrzhen@stu.xjtu.edu.cn'.
