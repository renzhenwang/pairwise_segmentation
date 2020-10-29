#!/usr/bin/env python
# -*- coding: utf-8 -*-

## python prepare_data.py --data_path E:/wrzhen/dataset/segthor_data/ --pre_data_result E:/wrzhen/segthor_bayes_segmentation/data/ --val_num 10

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

    return parser


def read_image(img_path):
    data = nib.load(img_path)
    hdr = data.header

    # convert to numpy
    data = data.get_data()
    return data, hdr


def savenpy(id, filelist, pre_data_result, train_val_test, context_num):
    print('start processing %s \t %d/%d' % (
        filelist[id], id + 1,
        len(filelist),
    ))
    name = os.path.split(filelist[id])[1]
    # volume_id = int(name.split('_')[-1])

    pre_data_result = os.path.join(pre_data_result, name)
    if not os.path.exists(pre_data_result):
        os.makedirs(pre_data_result)

    print(filelist[id])
    img, hdr = read_image(os.path.join(filelist[id], name +'.nii'))

    x_num, y_num, z_num = img.shape
    if context_num:
        img = np.pad(img, [[0, 0], [0, 0], [context_num, context_num]], 'constant')
    
    if train_val_test == 'train' or train_val_test == 'val':
        label, _ = read_image(os.path.join(filelist[id], 'GT.nii'))
        if np.sum(label > 0):
            _, _, organ_zz = np.where(label > 0)
            min_organ_z = np.min(organ_zz)
            max_organ_z = np.max(organ_zz)
            organ_range = [min_organ_z, max_organ_z]
        else:
            organ_range = [None, None]

    if train_val_test == 'train' or train_val_test == 'val':
        organ_infos = []
        for i in range(0, z_num):
            save_path = os.path.join(pre_data_result, name + '_%d_clean.npy' % i)
            img_slice = img[:, :, i:i + context_num * 2 + 1]
            np.save(save_path, img_slice.transpose([2, 0, 1]).astype(np.float16))
            label_slice = label[:, :, i]
            np.save(save_path.replace('clean', 'label'), label_slice.astype(np.uint8))

            if np.sum(label > 0):
                if min_organ_z <= i <= max_organ_z:
                    organ_infos.append([save_path, np.sum(label[:, :, i] > 0)])

        save_path = os.path.join(pre_data_result, name + '_info.pkl.gz')
        file = gzip.open(save_path, 'wb')
        print(z_num, organ_range)
        pickle.dump(z_num, file, protocol=-1)
        pickle.dump(organ_range, file, protocol=-1)
        pickle.dump(organ_infos, file, protocol=-1)
        file.close()

    else:
        for i in range(0, z_num):
            save_path = os.path.join(pre_data_result, name + '_%d_clean.npy' % i)
            img[:, :, i:i + context_num * 2 + 1]
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

        filelist = os.listdir(data_path)
        filelist = [os.path.join(data_path, file) for file in filelist]
        
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


if __name__ == '__main__':
    parser = arg_parser()
    args = parser.parse_args()

    full_prep(args.data_path, args.pre_data_result, args.val_num, args.semi_num, args.context_num)