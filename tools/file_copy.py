# -*- coding: utf-8 -*-

import os
import shutil

source_path = os.path.abspath(r'E:\source\path')
target_path = os.path.abspath(r'E:\target\path')


def copy_files(source_path, target_path):
    if not os.path.exists(target_path):
        os.makedirs(target_path)

    if os.path.exists(source_path):
        for root, dirs, files in os.walk(source_path):
            for file in files:
                src_file = os.path.join(root, file)
                shutil.copy(src_file, target_path)
                print(src_file)
    
    print('copy files finished!')



def main():
    
    dcm_dir = 'G:/wrzhen/dataset/chaos/CHAOS_Train_Sets/Train_Sets/MR'
    nii_dir = 'G:/wrzhen/dataset/chaos/training/t1Outphase/'
    if not os.path.exists(dcm_dir):
        print('must give a dcm dir')
        return
    if not os.path.exists(nii_dir):
        os.makedirs(nii_dir)
    
    volume_names =  os.listdir(dcm_dir)
    dcm_dirs = [os.path.join(dcm_dir, name, 'T1DUAL\DICOM_anon\OutPhase') for name in volume_names]
    ground_dirs = [dcm_dir.replace('DICOM_anon', 'Ground') for dcm_dir in dcm_dirs]
    
    for i in range(len(dcm_dirs)):
        copy_files(dcm_dirs[i], os.path.join(nii_dir, 't1Outphase_'+volume_names[i]))

if __name__ == '__main__':

    main()