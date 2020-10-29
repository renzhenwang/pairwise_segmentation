#!/usr/bin/env python
# -*- coding: utf-8 -*-

## python prepare_data.py --data_path '/data1/home/renzhenwang/lits/' --pre_data_result '/data1/home/renzhenwang/semi_lits/' --val_num 20 --semi_num 30

import argparse
import glob
import os
from functools import partial
from multiprocessing import Pool

import numpy as np
import gzip
import pickle
import random
import nibabel as nib
from scipy.ndimage.interpolation import zoom


def arg_parser():
    parser = argparse.ArgumentParser(
        prog='prepare_data',
        formatter_class=argparse.RawTextHelpFormatter,
        description='Prepare data for input of deep model,'
                    ' after this, we get extracted liver data'
                    ' and ground truth',
    )

    parser.add_argument(
        '--data_path',
        dest='data_path',
        default='',
        type=str,
        help='original lungs',
    )

    parser.add_argument(
        '--pre_data_result',
        dest='pre_data_result',
        default='',
        type=str,
        help='the save path of the processed data',
    )
    parser.add_argument('--val_num', type=int, default=10, help='validation num')
    parser.add_argument('--semi_num', type=int, default=0, help='unlabeled traini num')
    parser.add_argument('--context_num', type=int, default=2, help='context num')

    parser.add_argument(
        '--test',
        dest='test',
        action='store_true',
        help='Whether to test func',
    )

    return parser


def read_image(img_path):
    data = nib.load(img_path)
    hdr = data.header

    # convert to numpy
    data = data.get_data()
    return data, hdr


def img_clean(img, train_val_test, volume_name):
    volume_id = int(volume_name.split('-')[-1])
    if train_val_test == 'test':
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


def savenpy(id, filelist, pre_data_result, train_val_test, context_num):
    print('start processing %s \t %d/%d' % (
        filelist[id], id + 1,
        len(filelist),
    ))
    name = os.path.splitext(os.path.split(filelist[id])[1])[0]
    volume_id = int(name.split('-')[-1])

    pre_data_result = os.path.join(pre_data_result, name)
    if not os.path.exists(pre_data_result):
        os.makedirs(pre_data_result)

    img, hdr = read_image(filelist[id])
    if volume_id == 59 and train_val_test == 'test':
        resize_factor = 512. / img.shape[2]
        img = zoom(img, [1, 1, resize_factor], order=1).transpose([0, 2, 1])

    x_num, y_num, z_num = img.shape
    if context_num:
        img = np.pad(img, [[0, 0], [0, 0], [context_num, context_num]], 'constant')
    
    if train_val_test == 'train' or train_val_test == 'val':
        label, _ = read_image(filelist[id].replace('volume', 'segmentation'))
        if np.sum(label > 0):
            _, _, liver_zz = np.where(label > 0)
            min_liver_z = np.min(liver_zz)
            max_liver_z = np.max(liver_zz)
            liver_range = [min_liver_z, max_liver_z]
        else:
            liver_range = [None, None]

        if np.sum(label == 2):
            _, _, tumor_zz = np.where(label == 2)
            min_tumor_z = np.min(tumor_zz)
            max_tumor_z = np.max(tumor_zz)
            tumor_range = [min_tumor_z, max_tumor_z]
        else:
            tumor_range = [None, None]
    if train_val_test == 'train' or train_val_test == 'val':

        liver_infos = []
        tumor_infos = []
        for i in range(0, z_num):
            save_path = os.path.join(pre_data_result, name + '_%d_clean.npy' % i)
            img_slice = img_clean(img[:, :, i:i + context_num * 2 + 1], train_val_test, name)
            np.save(save_path, img_slice.transpose([2, 0, 1]).astype(np.float16))
            label_slice = img_clean(label[:, :, i], train_val_test, name)
            np.save(save_path.replace('clean', 'label'), label_slice.astype(np.uint8))

            if np.sum(label > 0):
                if min_liver_z <= i <= max_liver_z:
                    liver_infos.append([save_path, np.sum(label[:, :, i] > 0)])

            if np.sum(label == 2):
                if min_tumor_z <= i <= max_tumor_z:
                    tumor_infos.append([save_path, np.sum(label[:, :, i] == 2)])

        save_path = os.path.join(pre_data_result, name + '_info.pkl.gz')
        file = gzip.open(save_path, 'wb')
        pickle.dump(z_num, file, protocol=-1)
        pickle.dump(liver_range, file, protocol=-1)
        pickle.dump(liver_infos, file, protocol=-1)
        pickle.dump(tumor_range, file, protocol=-1)
        pickle.dump(tumor_infos, file, protocol=-1)
        file.close()

    else:
        for i in range(0, z_num):
            save_path = os.path.join(pre_data_result, name + '_%d_clean.npy' % i)
            img_slice = img_clean(img[:, :, i:i + context_num * 2 + 1], train_val_test, name)
            np.save(save_path, img_slice.transpose([2, 0, 1]).astype(np.float16))

    print('end processing %s \t %d/%d' % (filelist[id], id + 1, len(filelist)))


def process_data(filelist, train_val_test, pre_data_result, context_num):
    print('starting %s preprocessing' % train_val_test)
    pre_result_path = os.path.join(pre_data_result, train_val_test)
    if not os.path.exists(pre_result_path):
        os.mkdir(pre_result_path)

    pool = Pool(8)
    if train_val_test == 'val':
        partial_savenpy = partial(
            savenpy,
            filelist=filelist,
            pre_data_result=pre_result_path,
            train_val_test=train_val_test,
            context_num=context_num,

        )
    else:
        partial_savenpy = partial(
            savenpy,
            filelist=filelist,
            pre_data_result=pre_result_path,
            train_val_test=train_val_test,
            context_num=context_num
        )

    N = len(filelist)
    print(N)
    pool.map(partial_savenpy, range(N))
    pool.close()
    pool.join()

    print('end %s preprocessing' % train_val_test)


def full_prep(data_path, pre_data_result, val_num, semi_num, context_num):
    finished_flag = os.path.join(pre_data_result, '.flag_pre_lits')
    if not os.path.exists(pre_data_result):
        os.mkdir(pre_data_result)

    if not os.path.exists(finished_flag):
        train_dir1 = os.path.join(data_path, 'Training_Batch1')
        train_dir2 = os.path.join(data_path, 'Training_Batch2')

        filelist = glob.glob(os.path.join(train_dir1, 'volume-*.nii')) + \
            glob.glob(os.path.join(train_dir2, 'volume-*.nii'))
        random.shuffle(filelist)

        if args.val_num and args.semi_num == 0:
            process_data(
                filelist[:-val_num], 'train', pre_data_result, context_num
            )
            process_data(
                filelist[-val_num:], 'val', pre_data_result, context_num
            )
        elif args.semi_num:
            process_data(
                filelist[:val_num], 'val', pre_data_result, context_num
            )
            process_data(
                filelist[val_num:val_num+semi_num], 'semi', pre_data_result, context_num
            )
            process_data(
                filelist[val_num+semi_num:], 'train', pre_data_result, context_num
            )
        else:
            process_data(
                filelist, 'train', pre_data_result, context_num
            )

        test_dir = os.path.join(data_path, 'test')
        filelist = glob.glob(os.path.join(test_dir, '*.nii'))
        process_data(filelist, 'test', pre_data_result, context_num)

    f = open(finished_flag, "w+")
    f.close()


def test():
    savenpy(0, ['/data/home/eliasslcao/code/data/lits/test/test-volume-59.nii'],
            '/data/home/eliasslcao/code/data/processed_lits2d_data_ch5/test', 'test', args.context_num)


if __name__ == '__main__':
    parser = arg_parser()
    args = parser.parse_args()

    if args.test:
        test()
    else:
        full_prep(args.data_path, args.pre_data_result, args.val_num, args.semi_num, args.context_num)