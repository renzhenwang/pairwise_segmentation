# -*- coding: utf-8 -*-

import glob
import os
from functools import partial
from multiprocessing import Pool

import numpy as np
import gzip
import pickle
import random
import nibabel as nib


def read_image(img_path):
    data = nib.load(img_path)
    hdr = data.header

    # convert to numpy
    data = data.get_data()
    return data, hdr


def image_norm(img):
    pixels = img[img > 0]
    mean = pixels.mean()
    std  = pixels.std()
    out = (img - mean)/std
    out[img==0] = 0
    return out


def savenpy(id, filelist, pre_data_result, train_val_test, context_num):
    print('start processing %s \t %d/%d' % (
        filelist[id], id + 1,
        len(filelist),
    ))
    name = os.path.split(filelist[id])[1].split('.')[0]
    print(name)

    pre_data_result = os.path.join(pre_data_result, name)
    if not os.path.exists(pre_data_result):
        os.makedirs(pre_data_result)

    _img, hdr = read_image(filelist[id])
    # img = image_norm(_img)
    img = _img

    x_num, y_num, z_num = img.shape
    if context_num:
        img = np.pad(img, [[0, 0], [0, 0], [context_num, context_num]], 'constant')
    
    if train_val_test == 'train' or train_val_test == 'val' or 'fold' in train_val_test:
        label, _ = read_image(filelist[id].replace(name.split('_')[0], name[0:2]+'Segmentation'))
        if np.sum(label > 0):
            _, _, organ_zz = np.where(label > 0)
            min_organ_z = np.min(organ_zz)
            max_organ_z = np.max(organ_zz)
            organ_range = [min_organ_z, max_organ_z]
        else:
            organ_range = [None, None]

    if train_val_test == 'train' or train_val_test == 'val' or 'fold' in train_val_test:
        organ_infos = []
        for i in range(0, z_num):
            save_path = os.path.join(pre_data_result, name + '_%d_clean.npy' % i)
            img_slice = img[:, :, i:i + context_num * 2 + 1]
            np.save(save_path, img_slice.transpose([2, 0, 1]).astype(np.float32))  #MRI is 12bit, so it must not be np.float16
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
            img_slice = img[:, :, i:i + context_num * 2 + 1]
            np.save(save_path, img_slice.transpose([2, 0, 1]).astype(np.float16))

    print('end processing %s \t %d/%d' % (filelist[id], id + 1, len(filelist)))


def process_data(filelist, train_val_test, pre_data_result, context_num):
    print('starting %s preprocessing' % train_val_test)
    pre_result_path = os.path.join(pre_data_result, train_val_test)
    if not os.path.exists(pre_result_path):
        os.mkdir(pre_result_path)

    pool = Pool(4)
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


if __name__ == '__main__':
    data_path = 'G:/wrzhen/dataset/chaos/training/'   
    training_paths = [data_path+mode for mode in ['t1Inphase', 't1Outphase', 't2']]
    pre_data_result = 'G:/wrzhen/dataset/cross_chaos/'
    if not os.path.exists(pre_data_result):
        os.mkdir(pre_data_result)
    
    val_num = 10
    cross_fold = 5
    context_num = 2

    finished_flag = os.path.join(pre_data_result, '.flag_pre_lits')
    if not os.path.exists(finished_flag):
        filelist = glob.glob(os.path.join(training_paths[0], 't1Inphase_*.nii'))
        filelist = [os.path.split(file)[-1] for file in filelist]
        random.shuffle(filelist)
        print(filelist)
        
        filelists = []
        filelists.append([os.path.join(training_paths[0], file) for file in filelist])
        filelists.append([os.path.join(training_paths[1], file.replace('Inphase', 'Outphase')) for file in filelist])
        filelists.append([os.path.join(training_paths[2], file.replace('t1Inphase', 't2')) for file in filelist])
        
        for filelist in filelists:
            if val_num and cross_fold == 0:
                process_data(
                    filelist[:-val_num], 'train', pre_data_result, context_num
                )
                process_data(
                    filelist[-val_num:], 'val', pre_data_result, context_num
                )
            elif cross_fold:
                for i in range(cross_fold):
                    process_data(
                            filelist[i*4: (i+1)*4], 'fold'+str(i), pre_data_result, context_num
                    )
            else:
                process_data(
                    filelist, 'train', pre_data_result, context_num
                )

        test_dir = 'G:/wrzhen/dataset/cross_chaos/test/'
        filelist = glob.glob(os.path.join(test_dir, '*.nii'))
        print(filelist)
        process_data(filelist, 'test', pre_data_result, context_num)

    f = open(finished_flag, "w+")
    f.close()