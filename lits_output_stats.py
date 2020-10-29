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
from scipy.ndimage.interpolation import zoom
from skimage import measure
from medpy import metric
import utils.lits_utils as utils
from utils.surface import Surface

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
parser.add_argument('--pred_th', default=0.5, type=float,
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
		evalsurf = Surface(pred,label,physical_voxel_spacing = vxlspacing,mask_offset = [0.,0.,0.], reference_offset = [0.,0.,0.])
		volscores['assd'] = evalsurf.get_average_symmetric_surface_distance()

		volscores['msd'] = metric.hd(label,pred,voxelspacing=vxlspacing)

	return volscores


def process_output(id, namelist, args, dice_stats):
    print('start processing %s \t %d/%d' % (namelist[id], id + 1,
                                            len(namelist)))
    s = time.time()
    slice_list = glob.glob(os.path.join(args.output_dir, namelist[id] + '_' + '*.npy')) + \
                 glob.glob(os.path.join(args.output_dir, namelist[id], '*.npy'))

    slice_list = sorted(slice_list, key=lambda slice_name: int(slice_name.split(os.sep)[-1].split('_')[1]))
    liver_outputs = []
    for slice in slice_list:
        output_list = np.load(slice)
        # print("output_list.shape", output_list.shape)
        if len(output_list.shape) == 2:
            liver_output = utils.img_clean(output_list, args.is_val, namelist[id])
        else:
            liver_output = utils.img_clean(output_list[..., 0], args.is_val, namelist[id])
        liver_outputs.append(liver_output[..., np.newaxis])
     
    liver_outputs = np.concatenate(liver_outputs, axis=2)
    liver_pred = utils.sigmoid(liver_outputs) > args.pred_th
    final_pred = liver_pred.astype('int32')
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
        train_dir1 = os.path.join(args.data_root, 'Training_Batch1')
        train_dir2 = os.path.join(args.data_root, 'Training_Batch2')

        filelist = glob.glob(os.path.join(train_dir1, 'volume-*.nii')) + \
                   glob.glob(os.path.join(train_dir2, 'volume-*.nii'))

        for f_name in filelist:
            if namelist[id] + '.nii' in f_name:
                img_path = f_name
                target_path = f_name.replace('volume', 'segmentation')
                break

        img, affine, header = utils.read_image(img_path)
        target, _, _ = utils.read_image(target_path)
        liver = target > 0

        current_stat = [namelist[id]]
        
        liver_scores = get_scores(liver_pred, liver, header.get_zooms()[:3])
        liver_clean_scores = get_scores(clean_pred, liver, header.get_zooms()[:3])

        if np.sum(liver) > 0:
            liver_dice = liver_scores['dice']
            current_stat.append(liver_dice)

            liver_dice = liver_clean_scores['dice']
            current_stat.append(liver_dice)
            
            liver_assd = liver_clean_scores['assd']
            current_stat.append(liver_assd)
            liver_msd = liver_clean_scores['msd']
            current_stat.append(liver_msd)
        else:
            current_stat.extend([0, 0, 0, 0])

        dice_stats.append(current_stat)
        utils.save_pred(final_pred, target, img, namelist[id], args.pred_dir)
        
        utils.save_nii(final_pred, os.path.join(args.pred_dir, namelist[id].replace('volume', 'segmentation')
                                        + '.nii'), affine, header)

    else:
        test_dir = os.path.join(args.data_root, 'test')

        filelist = glob.glob(os.path.join(test_dir, 'test-volume-*.nii'))

        for f_name in filelist:
            if namelist[id] + '.nii' in f_name:
                img_path = f_name
                break

        img, affine, header = utils.read_image(img_path)
        volume_id = int(namelist[id].split('-')[-1])
        if volume_id == 59:
            resize_factor = img.shape[2] / 512.
            final_pred = zoom(final_pred, [1, resize_factor, 1], order=1).transpose([0, 2, 1])

        utils.save_nii(final_pred, os.path.join(args.pred_dir, namelist[id].replace('volume', 'segmentation')
                                                + '.nii'), affine, header)

    e = time.time()
    print('%d: %s %3.2f s' % (id + 1, namelist[id], e - s))
    print('end processing %s \t %d/%d ' % (namelist[id], id + 1,
                                           len(namelist), ))


def main():
    manager = Manager()
    dice_stats = manager.list()
    args = parser.parse_args()
    if not args.output_dir:
        print('must give a output dir for eval')
        return
    utils.mkdir_if_missing(args.pred_dir)

    with gzip.open(os.path.join(args.output_dir, 'namelist.pkl.gz')) as file:
        namelist = pickle.load(file)

    volume_names = list(set([name.split('_')[0] for name in namelist]))
    # print(volume_names)
    partial_process_output = partial(
        process_output,
        namelist=volume_names,
        args=args,
        dice_stats=dice_stats)

    pool = Pool(1)
    N = len(volume_names)
    print(N)
    _ = pool.map(partial_process_output, range(N))
    pool.close()
    pool.join()

    print('dice_stats: ', dice_stats)
    print('mean_dice: ', np.mean(np.array(dice_stats)[:, 1:].astype('float32'), axis=0))


if __name__ == '__main__':

    main()
