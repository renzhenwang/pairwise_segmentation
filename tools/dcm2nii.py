# -*- coding: utf-8 -*-

import os
from functools import partial
from multiprocessing import Pool
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from PIL import Image


def read_nii(img_path):
    nii = nib.load(img_path)
    # convert to numpy
    data = nii.get_data()
    return data, nii.affine, nii.header


def save_nii(prediction, save_name, affine, header):
    new_img = nib.Nifti1Image(prediction.astype(np.int16), affine, header)
    nib.save(new_img, save_name)
    print("Image saved at: " + str(save_name))


def dcm2nii(id, dcm_list, ground_list, volume_names, nii_dir):
    print('start processing %s \t %d/%d' % (dcm_list[id], id + 1,
                                            len(dcm_list)))   
    reader = sitk.ImageSeriesReader()
    # dicom_names = reader.GetGDCMSeriesFileNames('H:\\dicomdata\\test1labeldcm')
    dicom_names = reader.GetGDCMSeriesFileNames(dcm_list[id])
    reader.SetFileNames(dicom_names)
    image2 = reader.Execute()
    sitk.WriteImage(image2, os.path.join(nii_dir, 't2_'+volume_names[id]+'.nii'))
    
    # save groundtruth
    img, affine, header = read_nii(os.path.join(nii_dir, 't2_'+volume_names[id]+'.nii'))
    print(img.shape)
    png_files = os.listdir(ground_list[id])
    png_files = [os.path.join(ground_list[id], file) for file in png_files]
    png_list = [np.asarray(Image.open(file)).transpose()[..., np.newaxis] for file in png_files]
    png_sequence = np.concatenate(png_list, axis=2)
    png_sequence = png_sequence.astype('int32')
    print(png_sequence.shape)
    save_nii(png_sequence, os.path.join(nii_dir, 't2Segmentation_'+volume_names[id]+'.nii'), affine, header)

def main():
    
    dcm_dir = 'G:/wrzhen/dataset/chaos/CHAOS_Train_Sets/Train_Sets/MR'
    nii_dir = 'G:/wrzhen/dataset/chaos/training'
    if not os.path.exists(dcm_dir):
        print('must give a dcm dir')
        return
    if not os.path.exists(nii_dir):
        os.makedirs(nii_dir)
    
    volume_names =  os.listdir(dcm_dir)
    dcm_dirs = [os.path.join(dcm_dir, name, 'T2SPIR\DICOM_anon') for name in volume_names]
    ground_dirs = [dcm_dir.replace('DICOM_anon', 'Ground') for dcm_dir in dcm_dirs]
    # print(dcm_dirs)
    
    partial_dcm2nii = partial(
            dcm2nii,
            dcm_list = dcm_dirs,
            ground_list = ground_dirs,
            volume_names = volume_names,
            nii_dir = nii_dir
        )
    
    pool = Pool(1)
    N = len(dcm_dirs)
    print(N)
    pool.map(partial_dcm2nii, range(N))
    pool.close()
    pool.join()

if __name__ == '__main__':

    main()
