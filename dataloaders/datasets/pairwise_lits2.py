# -*- coding: utf-8 -*-

import os
import glob
import gzip
import pickle
import random
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from dataloaders import pairwise_transforms as tr

from mypath import Path


class PairwiseCompose(object):
    def __init__(self, tfs):
        self.transforms = tfs

    def __call__(self, img1, img2):
        for t in self.transforms:
            img1, img2 = t(img1, img2)
        return img1, img2


class LitsDataset(object):
    def __init__(self, data_root, margin, **kwargs):
        train_dir = os.path.join(data_root, 'train', '*')
        val_dir = os.path.join(data_root, 'val', '*')
        test_dir = os.path.join(data_root, 'test', '*')

        self.margin = margin

        self.train = []
        self.train2 = []
        self.train_info = glob.glob(os.path.join(train_dir, '*_info.pkl.gz'))
        
        for info in self.train_info:
            with gzip.open(info) as file:
                z_num = pickle.load(file)
                organ_range = pickle.load(file)

            for i in range(0, z_num):
                self.train.append(info.replace('info.pkl.gz', str(i)+'_clean.npy'))

            if organ_range[0] is not None:
                organ_min_range = max(organ_range[0]-margin, 0)
                organ_max_range = min(organ_range[1]+margin, z_num-1)
                for i in range(organ_min_range, organ_max_range + 1):
                    self.train2.append(info.replace('info.pkl.gz', str(i)+'_clean.npy'))
        
        self.val = []
        self.val2 = []
        self.val_info = glob.glob(os.path.join(val_dir, '*_info.pkl.gz'))
        for info in self.val_info:
            with gzip.open(info) as file:
                z_num = pickle.load(file)
                organ_range = pickle.load(file)

            for i in range(0, z_num):
                self.val.append(info.replace('info.pkl.gz', str(i)+'_clean.npy'))

            if organ_range[0] is not None:
                organ_min_range = max(organ_range[0] - margin, 0)
                organ_max_range = min(organ_range[1] + margin, z_num - 1)
                for i in range(organ_min_range, organ_max_range + 1):
                    self.val2.append(info.replace('info.pkl.gz', str(i)+'_clean.npy'))

        self.test = glob.glob(os.path.join(test_dir, '*.npy'))


class LitsDataloader(Dataset):
    NUM_CLASSES = 2
    
    def __init__(self, args,  data_root=Path.db_root_dir('lits'), data_phase='train', 
                 margin=5, homo_train=True, hete_train=True, homo_train_inter=[5,9,13], 
                 hete_train_iters=1, identy_train=False, identy_iters=1, transform=True):
        dataset = LitsDataset(data_root, margin)
        
        random.seed(args.seed)
        
        if data_phase not in ['train', 'val', 'test']:
            raise ValueError("data_phase must in ['train', 'val', 'test']")
        
        if data_phase == 'train':
            data_total_files = dataset.train
            data_patial_files = dataset.train2
            data1_files = []
            data2_files = []
            if hete_train and hete_train_iters:
                temp = data_total_files[:]
                for _ in range(hete_train_iters):
                    data1_files += temp
                data2_files = data1_files[:]
                random.shuffle(data2_files)
            if homo_train and homo_train_inter is not None:
                for i in range(len(homo_train_inter)):
                    temp = data_patial_files[:-homo_train_inter[i]]
                    data1_files += temp
                    temp = data_patial_files[homo_train_inter[i]:]
                    data2_files += temp
            if identy_train and identy_iters:
                temp = data_patial_files[:]
                for _ in range(identy_iters):
                    data1_files += temp
                    data2_files += temp

        elif data_phase == 'val':
            data1_files = dataset.val[:]
            data2_files = dataset.val[:]
        else:
            data1_files = dataset.test[:]
            data2_files = dataset.test[:]
            
        self.data1_files = data1_files[:]
        self.data2_files = data2_files[:]
        
        self.args = args
        self.data_phase = data_phase
        self.transform = transform


    def __len__(self):
        L = len(self.data1_files)
        return L
    
    @staticmethod
    def lum_trans(img):
        liver_win = [-200, 250]
        newimg = (img - liver_win[0]) / (liver_win[1] - liver_win[0])
        newimg[newimg < 0] = 0
        newimg[newimg > 1] = 1
        return newimg


    def __getitem__(self, index):
        ## sample1
        img1_path = self.data1_files[index]
        _img = np.load(img1_path)[1:4]
        _img = self.lum_trans(_img)
        
        target1_path = img1_path.replace('clean', 'label')
        _target = np.load(target1_path)
        
        _target = _target > 0
        _target = _target.astype('uint8')
        
        sample1 = {'image': _img, 'label': _target}
        
        if self.transform and self.data_phase=='train':
            _img = np.uint8(_img*255)
            _img = Image.fromarray(np.transpose(_img, axes=[1,2,0]))
            _target = Image.fromarray(_target)
            sample1 = {'image': _img, 'label': _target} 
            # sample1 = self.transform_tr(sample1)
        if self.transform and self.data_phase=='val':
            _img = np.uint8(_img*255)
            _img = Image.fromarray(np.transpose(_img, axes=[1,2,0]))
            _target = Image.fromarray(_target)
            sample1 = {'image': _img, 'label': _target} 
            # sample1 = self.transform_val(sample1)
        
        ## sample2
        img2_path = self.data2_files[index]
        _img = np.load(img2_path)[1:4]
        _img = self.lum_trans(_img)
        
        target2_path = img2_path.replace('clean', 'label')
        _target = np.load(target2_path)
        
        _target = _target > 0
        _target = _target.astype('uint8')
        
        sample2 = {'image': _img, 'label': _target}
        
        if self.transform and self.data_phase=='train':
            _img = np.uint8(_img*255)
            _img = Image.fromarray(np.transpose(_img, axes=[1,2,0]))
            _target = Image.fromarray(_target)
            sample2 = {'image': _img, 'label': _target} 
            # sample2 = self.transform_tr(sample2)
            sample1, sample2 = self.transform_tr(sample1, sample2)
        if self.transform and self.data_phase=='val':
            _img = np.uint8(_img*255)
            _img = Image.fromarray(np.transpose(_img, axes=[1,2,0]))
            _target = Image.fromarray(_target)
            sample2 = {'image': _img, 'label': _target} 
            # sample2 = self.transform_val(sample2)
            sample1, sample2 = self.transform_val(sample1, sample2)
        proxy_label = sample1['label'] + sample2['label']
        return sample1, sample2, proxy_label, index
    
    
    def transform_tr(self, sample1, sample2):
        composed_transforms = PairwiseCompose([
            tr.RandomZoom(base_size=self.args.base_size, crop_size=self.args.crop_size),
            # tr.RandomHorizontalFlip(),
            tr.Normalize(),
            tr.ToTensor()])
        return composed_transforms(sample1, sample2)


    def transform_val(self, sample1, sample2):

        composed_transforms = PairwiseCompose([
            # tr.FixScaleCrop(crop_size=self.args.crop_size),
            tr.Normalize(),
            tr.ToTensor()])
        return composed_transforms(sample1, sample2)


if __name__ == '__main__':
    from dataloaders.utils import decode_segmap
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.base_size = 512
    args.crop_size = 512
    args.seed = 1

    thor_train = LitsDataloader(args, data_phase='train',  transform=True)
    print(len(thor_train.data1_files))
    dataloader = DataLoader(thor_train, batch_size=15, shuffle=True, num_workers=0)

    for ii, (sample1, sample2, proxy_label, sample_indies) in enumerate(dataloader):
        for jj in range(sample1["image"].size()[0]):
            img = sample1['image'].numpy()
            gt = sample1['label'].numpy()
            tmp = np.array(gt[jj]).astype(np.uint8)
            segmap = decode_segmap(tmp, dataset='pairwise_lits')
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
            img_tmp *= 255.0
            img_tmp = img_tmp.astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(321)
            plt.imshow(img_tmp)
            plt.subplot(322)
            plt.imshow(segmap)
            
            img = sample2['image'].numpy()
            gt = sample2['label'].numpy()
            tmp = np.array(gt[jj]).astype(np.uint8)
            segmap = decode_segmap(tmp, dataset='pairwise_lits')
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
            img_tmp *= 255.0
            img_tmp = img_tmp.astype(np.uint8)
            plt.title('display')
            plt.subplot(323)
            plt.imshow(img_tmp)
            plt.subplot(324)
            plt.imshow(segmap)
            
            tmp = np.array(proxy_label[jj]).astype(np.uint8)
            segmap = decode_segmap(tmp, dataset='pairwise_lits')
            plt.subplot(325)
            plt.imshow(segmap)

        if ii == 1:
            break

    plt.show(block=True)