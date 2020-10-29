# -*- coding: utf-8 -*-

import sys
import os
import errno
import glob

import torch
import nibabel as nib
import numpy as np
from PIL import Image
from functools import partial
from multiprocessing import Pool
import numpy.linalg as nl
import pydicom
from scipy.ndimage import zoom
import scipy.special as scipy_special
from scipy.spatial.distance import pdist, cdist, squareform
from scipy.ndimage.interpolation import map_coordinates
from scipy import ndimage
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt


def mkdir_if_missing(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def sigmoid(output):
    return scipy_special.expit(output)


def save_checkpoint(state, fpath='checkpoint.pth.tar'):
    mkdir_if_missing(os.path.dirname(fpath))
    torch.save(state, fpath)


class Logger(object):
    """
    Write console output to external text file.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


class AverageMeter(object):
    """Computes and stores the average and current value.

       Code imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val):
        count = val.size
        v = val.sum()

        self.count += count
        self.sum += v

        self.avg = self.sum / self.count


def read_image(img_path):
    nii = nib.load(img_path)

    # convert to numpy
    data = nii.get_data()
    return data, nii.affine, nii.header


def save_nii(prediction, save_name, affine, header):
    new_img = nib.Nifti1Image(prediction.astype(np.int16), affine, header)
    nib.save(new_img, save_name)
    print("Image saved at: " + str(save_name))
    

def lum_trans(img):
    liver_win = [-200, 250]
    newimg = (img - liver_win[0]) / (liver_win[1] - liver_win[0])
    newimg[newimg < 0] = 0
    newimg[newimg > 1] = 1
    return newimg * 255


def img_merge(target, pred, img, save_path):
    to_image = Image.new('RGB', (img.shape[1] * 3, img.shape[0] * 1))
    
    volume_img = np.zeros((img.shape[0], img.shape[1], 3))
    volume_img[..., 0] = lum_trans(img)
    volume_img[..., 1] = lum_trans(img)
    volume_img[..., 2] = lum_trans(img)
    volume_img = Image.fromarray(volume_img.astype('uint8'), 'RGB')
    to_image.paste(volume_img, (0, 0))

    liver_mask_img = np.zeros((target.shape[0], target.shape[1], 3))
    liver_mask_img[..., 0] = (pred > 0) * 255
    liver_mask_img[..., 1] = (target > 0) * 255
    liver_mask_img = Image.fromarray(liver_mask_img.astype('uint8'), 'RGB')
    to_image.paste(liver_mask_img, (img.shape[1], 0))

    tumor_mask_img = np.zeros((target.shape[0], target.shape[1], 3))
    tumor_mask_img[..., 0] = (pred == 2) * 255
    tumor_mask_img[..., 1] = (target == 2) * 255
    tumor_mask_img = Image.fromarray(tumor_mask_img.astype('uint8'), 'RGB')
    to_image.paste(tumor_mask_img, (img.shape[1] * 2, 0))

    to_image.save(save_path)


def save_pred(pred, target, img, name, pred_save_dir):
    for i in range(pred.shape[2]):
        result_save_dir = os.path.join(pred_save_dir, name + '_' + str(i + 1) + '.png')
        img_merge(target[:, :, i], pred[:, :, i], img[:, :, i], result_save_dir)


def compute_dice(pred, target, margin=None):
    if margin:
        pred = pred[:, margin[0]:-margin[0], margin[1]:-margin[1]]
        target = target[:, margin[0]:-margin[0], margin[1]:-margin[1]]
    pred = pred.flatten().astype(np.bool)
    target = target.flatten().astype(np.bool)
    
    # Compute dice, reference from MedPy
    # pred = np.atleast_1d(pred.astype(np.bool))
    # target = np.atleast_1d(target.astype(np.bool))
    smooth = 1.0
    
    intersection = np.count_nonzero(pred & target)
    card_sum = np.count_nonzero(pred) + np.count_nonzero(target)
    return (2.0*intersection+smooth) / (card_sum+smooth)


def plot_curve(x, y, title='fig', xlbl='epoch', ylbl=None, save_path=''):
    plt.figure()
    plt.plot(x, y, 'o-')
    # plt.title('Test accuracy vs. epoches')
    plt.title(title)
    plt.xlabel(xlbl)
    plt.ylabel(ylbl)
    # plt.ylabel('Test accuracy')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(save_path+title+'.jpg')
    
    
def save_output(output_list, padding, nhw, patch_nums, image_indices, dataset,
                save_dir, split_combiner, scale=None):
        current_i = 0
        for i in range(len(patch_nums)):
            # name = dataset[image_indices[i]].split('/')[-1].split('.')[0]
            _, name = os.path.split(dataset[image_indices[i]])
            name = name.split('.')[0]
            # print("-------", name)
            output = output_list[current_i:current_i+patch_nums[i]]

            output = split_combiner.combine(output, nhw=nhw[i], padding=padding[i])
            if scale and scale != 1.0:
                output = zoom(output, [1.0 / scale, 1.0 / scale, 1], order=1)
            np.save(os.path.join(save_dir, name + '_output.npy'), output.astype(np.float16))

            current_i += patch_nums[i]
            
            
def img_clean(img, is_val, volume_name):
    volume_id = int(volume_name.split('-')[-1])
    if not is_val:
        if (volume_id >= 5) and (volume_id <= 14):
            img = np.fliplr(img)

        elif (volume_id >= 15) and (volume_id <= 29):
            img = np.flipud(img)

        elif (volume_id >= 30) and (volume_id <= 41):
            img = np.fliplr(img)
        elif volume_id == 59:
            img = np.fliplr(img)
    else:
        if (volume_id >= 53) and (volume_id <= 67):
            img = np.fliplr(img)

        elif (volume_id >= 68) and (volume_id <= 82):
            img = np.flipud(img)

        elif (volume_id >= 83) and (volume_id <= 130):
            img = np.fliplr(img)
    return img

def set_scale(test_loader, test_scale):
    test_loader.dataset.test_scale = test_scale
    

def output_multi2one(id, name_list, out_dirs, merge_dir, thred):
    volume_names = list(set([name.split('_')[0] for name in name_list]))
    slice_names = list(set(name_list))
    
    vlm_name = volume_names[id]
    vlm_slc_names = [name for name in slice_names if vlm_name==name.split('_')[0]]
    
    vlm_dir = os.path.join(merge_dir, vlm_name)
    if not os.path.exists(vlm_dir):
        os.makedirs(vlm_dir) 
    for slc_name in vlm_slc_names:
        name = slc_name.replace('clean', 'clean_output')+'.npy'
        # print(name)
        liver_sum = 0
        for output_dir in out_dirs:
            liver = np.load(os.path.join(output_dir, name))
            liver_sum += liver
        liver_pred = (liver_sum / len(out_dirs)) > thred
        save_path = os.path.join(vlm_dir, name.replace('clean_output', 'label'))
        # print(save_path)
        np.save(save_path, liver_pred.astype(np.uint8))
        
        
def multithread_output_multi2one(out_dirs, name_list, merge_dir, thred=0.5, is_parall=True):
    volume_names = list(set([name.split('_')[0] for name in name_list]))
    slice_names = list(set(name_list))

    N = len(volume_names)
    print(N)
    if not is_parall:
        for idx in range(N):
            output_multi2one(idx, slice_names, out_dirs, merge_dir, thred)
    else:
        pool = Pool(8)
        
        partial_output_multi2one = partial(output_multi2one,
                                           name_list=slice_names,
                                           out_dirs= out_dirs,
                                           merge_dir=merge_dir,
                                           thred = thred)
        pool.map(partial_output_multi2one, range(N))
        pool.close()
        pool.join()
    
"""
def output_multi2one(id, slice_names, out_dirs, volume_dir):    
    name = slice_names[id].replace('clean', 'clean_output')+'.npy'
    # print(name)
    liver_sum = 0
    for output_dir in out_dirs:
        liver = np.load(os.path.join(output_dir, name))
        liver_sum += liver
    liver_pred = (liver_sum / len(out_dirs)) > 0.5
    save_path = os.path.join(volume_dir, name.replace('clean_output', 'label'))
    # print(save_path)
    np.save(save_path, liver_pred.astype(np.uint8))
            

def multithread_output_multi2one(out_dirs, name_list, merge_dir, is_parall=True):
    volume_names = list(set([name.split('_')[0] for name in name_list]))
    slice_names = list(set(name_list))
    
    vlm_slc_names = []
    for vlm_name in volume_names:
        vlm_slc_names = [name for name in slice_names if vlm_name in name]
        vlm_dir = os.path.join(merge_dir, vlm_name)
        if not os.path.exists(vlm_dir):
            os.makedirs(vlm_dir)
        N = len(vlm_slc_names)
        print(N)
        if not is_parall:
            for idx in range(N):
                output_multi2one(idx, vlm_slc_names, out_dirs, vlm_dir)
        else:
            pool = Pool(8)
            
            partial_output_multi2one = partial(output_multi2one,
                                               slice_names=vlm_slc_names,
                                               out_dirs= out_dirs,
                                               volume_dir=vlm_dir)
            pool.map(partial_output_multi2one, range(N))
            pool.close()
            pool.join()
"""           

def main():
    
    name_list = glob.glob(os.path.join(r'D:\PyCode\pairwise_segmentation\pre_data\val\volume-0','*clean.npy'))+ \
    glob.glob(os.path.join(r'D:\PyCode\pairwise_segmentation\pre_data\val\volume-5','*clean.npy'))
    
    new_list = []
    for name in name_list:
        _, name = os.path.split(name)
        name = name.split('.')[0]
        new_list.append(name)
        
    volume_names = list(set([name.split('_')[0] for name in new_list]))
    out_dirs = [r'D:\PyCode\pairwise_segmentation\demos\result\val_1', 
                r'D:\PyCode\pairwise_segmentation\demos\result\val_3',
                r'D:\PyCode\pairwise_segmentation\demos\result\val_5']
    merge_dir = r'D:\PyCode\pairwise_segmentation\pre_data\semi'
    
    print(new_list)
    print(volume_names)
    
    import time
    start_time = time.time()
    multithread_output_multi2one(out_dirs, new_list, merge_dir, is_parall=True)
    end_time = time.time()
    print(end_time-start_time)  ##8.769879341125488
if __name__ == '__main__':
    main()
