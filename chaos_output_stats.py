#!/usr/bin/env python

import argparse
import pickle
import gzip
import os
import time
from functools import partial
from multiprocessing import Manager, Pool

import numpy as np
import glob
import utils.chaos_utils as utils
from scipy.ndimage.interpolation import zoom
from skimage import measure
from utils.surface import Surface
from medpy import metric

parser = argparse.ArgumentParser(description='PyTorch 2D Segmentation')
parser.add_argument(
    '--output-dir',
    default='',
    type=str,
    metavar='DATA',
    help='directory to data computed by 3d cnn')
parser.add_argument(
    '--data_root',
    type=str,
    default='',
    help='path to dataset')
parser.add_argument(
    '--pred-dir',
    default='',
    type=str,
    metavar='DATA',
    help='directory to data computed by 3d cnn')
parser.add_argument(
    '--is_val',
    action='store_true',
    help="use for validation")
parser.add_argument('--pred_th', default=0.2, type=float,
                    help="prediction threshold")


def get_scores(pred,label,vxlspacing):
	volscores = {}

	volscores['dice'] = metric.dc(pred,label)
	volscores['jaccard'] = metric.binary.jc(pred,label)
	volscores['voe'] = 1. - volscores['jaccard']
	volscores['rvd'] = metric.ravd(label,pred)

	if np.count_nonzero(pred) ==0 or np.count_nonzero(label)==0:
		volscores['assd'] = 0
		volscores['msd'] = 0
	else:
		# evalsurf = Surface(pred,label,physical_voxel_spacing = vxlspacing,mask_offset = [0.,0.,0.], reference_offset = [0.,0.,0.])
		# volscores['assd'] = evalsurf.get_average_symmetric_surface_distance()
		volscores['assd'] = metric.assd(label, pred, voxelspacing=vxlspacing)
		volscores['msd'] = metric.hd(label,pred,voxelspacing=vxlspacing)

	return volscores


def process_output(id, namelist, args, dice_stats, assd_stats):
    print('start processing %s \t %d/%d' % (namelist[id], id + 1,
                                            len(namelist)))
    s = time.time()
    slice_list = glob.glob(os.path.join(args.output_dir, namelist[id] + '_' + '*.npy'))
    slice_list = sorted(slice_list, key=lambda slice_name: int(slice_name.split(os.sep)[-1].split('_')[2]))
    
    back_outputs = []
    liver_outputs = []
    left_kidney_outputs = []
    right_kidney_outputs = []
    spleen_outputs = []
    
    for slice in slice_list:
        output_list = np.load(slice)
        back_output = output_list == 0
        back_outputs.append(back_output[..., np.newaxis])

        liver_output = output_list == 1
        liver_outputs.append(liver_output[..., np.newaxis])
        
        right_kidney_output = output_list == 2
        right_kidney_outputs.append(right_kidney_output[..., np.newaxis])
        
        left_kidney_output = output_list == 3
        left_kidney_outputs.append(left_kidney_output[..., np.newaxis])
        
        spleen_output = output_list == 4
        spleen_outputs.append(spleen_output[..., np.newaxis])
        

    back_pred = np.concatenate(back_outputs, axis=2).astype('int32')
    liver_pred = np.concatenate(liver_outputs, axis=2).astype('int32')
    left_kidney_pred = np.concatenate(left_kidney_outputs, axis=2).astype('int32')
    right_kidney_pred = np.concatenate(right_kidney_outputs, axis=2).astype('int32')
    spleen_pred = np.concatenate(spleen_outputs, axis=2).astype('int32')
    
    
    final_pred = np.zeros_like(back_pred).astype('int32')
    final_pred[liver_pred > 0] = 1
    final_pred[right_kidney_pred > 0] = 2
    final_pred[left_kidney_pred > 0] = 3
    final_pred[spleen_pred > 0] = 4
    # final_pred = img_clean(final_pred, args.is_val, namelist[id])

    clean_pred = np.copy(final_pred)
    measure_labels = measure.label(final_pred > 0)
    iou_area = []
    for l in range(1, np.amax(measure_labels) + 1):
        iou_area.append(np.sum(measure_labels == l))

    if len(iou_area) > 0:
        max_iou_area = np.amax(iou_area)
        for l in range(len(iou_area)):
            if iou_area[l] < max_iou_area:
                clean_pred[measure_labels == l + 1] = 0

    if args.is_val:
        train_dir1 = os.path.join(args.data_root, 't1Inphase')
        train_dir2 = os.path.join(args.data_root, 't1Outphase')
        train_dir3 = os.path.join(args.data_root, 't2')

        filelist = glob.glob(os.path.join(train_dir1, '*.nii')) + \
                   glob.glob(os.path.join(train_dir2, '*.nii')) + \
                   glob.glob(os.path.join(train_dir3, '*.nii'))

        for f_name in filelist:
            if namelist[id] + '.nii' in f_name:
                img_path = f_name
                target_path = img_path.replace('t1Inphase', 't1Segmentation')
                target_path = target_path.replace('t1Outphase', 't1Segmentation')
                target_path = target_path.replace('t2', 't2Segmentation')
                print(target_path)
                break

        img, affine, header = utils.read_image(img_path)
        _target, _, _ = utils.read_image(target_path)
        target = np.zeros(_target.shape)
        target[(_target>=55)&(_target<=70)]=1   # liver
        target[(_target>=110)&(_target<=135)]=2   # right kidney
        target[(_target>=175)&(_target<=200)]=3   # left kidney
        target[(_target>=240)&(_target<=255)]=4   # spleen
        target = target.astype('int32')
        
        # background = target == 0
        liver = target ==1
        right_kidney = target == 2
        left_kidney = target == 3
        spleen = target == 4

        current_dice_stat = [namelist[id]]
        current_assd_stat = [namelist[id]]

        liver_scores = get_scores(liver_pred, liver, header.get_zooms()[:3])
        liver_dice, liver_assd = liver_scores['dice'], liver_scores['assd']
        current_dice_stat.append(liver_dice)
        current_assd_stat.append(liver_assd)
        
        rkid_scores = get_scores(right_kidney_pred, right_kidney, header.get_zooms()[:3])
        rkid_dice, rkid_assd = rkid_scores['dice'], rkid_scores['assd']
        current_dice_stat.append(rkid_dice)
        current_assd_stat.append(rkid_assd)
        
        lkid_scores = get_scores(left_kidney_pred, left_kidney, header.get_zooms()[:3])
        lkid_dice, lkid_assd = lkid_scores['dice'], lkid_scores['assd']
        current_dice_stat.append(lkid_dice)
        current_assd_stat.append(lkid_assd)

        spleen_scores = get_scores(spleen_pred, spleen, header.get_zooms()[:3])
        spleen_dice, spleen_assd = spleen_scores['dice'], spleen_scores['assd']
        current_dice_stat.append(spleen_dice)
        current_assd_stat.append(spleen_assd)

        dice_stats.append(current_dice_stat)
        assd_stats.append(current_assd_stat)
       
        utils.save_pred(final_pred, target, img, namelist[id], args.pred_dir)

    else:
        test_dir = os.path.join(args.data_root, 'test')

        filelist = glob.glob(os.path.join(test_dir, 'test-volume-*.nii'))

        for f_name in filelist:
            if namelist[id] + '.nii' in f_name:
                img_path = f_name
                break

        img, affine, header = utils.read_image(img_path)
        utils.save_nii(final_pred, os.path.join(args.pred_dir, namelist[id].replace('volume', 'segmentation')
                                                + '.nii'), affine, header)

    e = time.time()
    print('%d: %s %3.2f s' % (id + 1, namelist[id], e - s))
    print('end processing %s \t %d/%d ' % (namelist[id], id + 1,
                                           len(namelist), ))


def main():
    manager = Manager()
    dice_stats = manager.list()
    assd_stats = manager.list()
    
    args = parser.parse_args()
    if not args.output_dir:
        print('must give a output dir for eval')
        return
    utils.mkdir_if_missing(args.pred_dir)

    with gzip.open(os.path.join(args.output_dir, 'namelist.pkl.gz')) as file:
        namelist = pickle.load(file)

    volume_names = list(set([name.split('_')[0]+'_'+name.split('_')[1] for name in namelist]))
    print(volume_names)
    partial_process_output = partial(
        process_output,
        namelist=volume_names,
        args=args,
        dice_stats=dice_stats,
        assd_stats=assd_stats)

    pool = Pool(8)
    N = len(volume_names)
    print(N)
    _ = pool.map(partial_process_output, range(N))
    pool.close()
    pool.join()

    print('dice_stats: ', dice_stats)
    print('assd_stats: ', assd_stats)
    print('mean_dice: ', np.mean(np.array(dice_stats)[:, 1:].astype('float32'), axis=0))
    print('mean_assd: ', np.mean(np.array(assd_stats)[:, 1:].astype('float32'), axis=0))


if __name__ == '__main__':

    main()
