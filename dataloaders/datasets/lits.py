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
from dataloaders import medical_transforms as tr

from mypath import Path


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
    
    def __init__(self, args,  data_root=Path.db_root_dir('lits'), data_phase='train', margin=5, transform=True):
        dataset = LitsDataset(data_root, margin)
        
        if data_phase not in ['train', 'val', 'test']:
            raise ValueError("data_phase must in ['train', 'val', 'test']")
        
        if data_phase == 'train':
            data_files = dataset.train
        elif data_phase == 'val':
            data_files = dataset.val
        else:
            data_files = dataset.test
            
        self.data_files = data_files[:]
        random.shuffle(self.data_files)
        
        self.args = args
        self.data_phase = data_phase
        self.transform = transform


    def __len__(self):
        L = len(self.data_files)
        return L
    
    @staticmethod
    def lum_trans(img):
        liver_win = [-200, 250]
        newimg = (img - liver_win[0]) / (liver_win[1] - liver_win[0])
        newimg[newimg < 0] = 0
        newimg[newimg > 1] = 1
        return newimg


    def __getitem__(self, index):
        img_path = self.data_files[index]
        _img = np.load(img_path)[1:4]
        _img = self.lum_trans(_img)
        
        target_path = img_path.replace('clean', 'label')
        _target = np.load(target_path)
        
        _target = _target > 0
        _target = _target.astype('uint8')
        
        sample = {'image': _img, 'label': _target}
        
        if self.transform and self.data_phase=='train':
            _img = np.uint8(_img*255)
            _img = Image.fromarray(np.transpose(_img, axes=[1,2,0]))
            _target = Image.fromarray(_target)
            sample = {'image': _img, 'label': _target} 
            sample = self.transform_tr(sample)
        if self.transform and self.data_phase=='val':
            _img = np.uint8(_img*255)
            _img = Image.fromarray(np.transpose(_img, axes=[1,2,0]))
            _target = Image.fromarray(_target)
            sample = {'image': _img, 'label': _target} 
            sample = self.transform_val(sample)
        return sample, index
    
    
    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomZoom(base_size=self.args.base_size, crop_size=self.args.crop_size),
            # tr.RandomHorizontalFlip(),
            tr.Normalize(),
            tr.ToTensor()])
        return composed_transforms(sample)


    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            tr.FixScaleCrop(crop_size=self.args.crop_size),
            tr.Normalize(),
            tr.ToTensor()])
        return composed_transforms(sample)


if __name__ == '__main__':
    from dataloaders.utils import decode_segmap
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.base_size = 512
    args.crop_size = 512

    thor_train = LitsDataloader(args, data_phase='train',  transform=True)

    dataloader = DataLoader(thor_train, batch_size=15, shuffle=True, num_workers=0)

    for ii, (sample, sample_indies) in enumerate(dataloader):
        for jj in range(sample["image"].size()[0]):
            img = sample['image'].numpy()
            gt = sample['label'].numpy()
            tmp = np.array(gt[jj]).astype(np.uint8)
            segmap = decode_segmap(tmp, dataset='lits')
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
            img_tmp *= 255.0
            img_tmp = img_tmp.astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(211)
            plt.imshow(img_tmp)
            plt.subplot(212)
            plt.imshow(segmap)

        if ii == 1:
            break

    plt.show(block=True)