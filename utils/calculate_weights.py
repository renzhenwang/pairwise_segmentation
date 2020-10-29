import os
from tqdm import tqdm
import numpy as np
from mypath import Path

def calculate_weigths_labels(dataset, dataloader, num_classes):
    # Create an instance from the data loader
    z = np.zeros((num_classes,))
    # Initialize tqdm
    tqdm_batch = tqdm(dataloader)
    print('Calculating classes weights')
    for sample in tqdm_batch:
        y = sample[2] if dataset.find('pairwise')>=0 or dataset.find('symmetric')>=0 else sample[0]['label']
        # y = sample['label']
        y = y.detach().cpu().numpy()
        if len(y.shape) > 3:
            count_l = y.sum(axis=-1).sum(axis=-1).sum(axis=0)
        else:
            mask = (y >= 0) & (y < num_classes)
            labels = y[mask].astype(np.uint8)
            count_l = np.bincount(labels, minlength=num_classes)
        z += count_l
    tqdm_batch.close()
    total_frequency = np.sum(z)
    class_weights = []
    for frequency in z:
        class_weight = 1 / (np.log(1.02 + (frequency / total_frequency)))
        class_weights.append(class_weight)
    ret = np.array(class_weights)
    classes_weights_path = os.path.join(Path.db_root_dir(dataset), dataset+'_classes_weights.npy')
    np.save(classes_weights_path, ret)
    print('class weight is ', ret)
    return ret