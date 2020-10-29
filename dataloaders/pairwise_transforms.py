# -*- coding: utf-8 -*-

import torch
import random
import numpy as np

from PIL import Image, ImageOps, ImageFilter

class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std
    
    def _normalize(self, sample):
        img = sample['image']
        mask = sample['label']
        img = np.array(img).astype(np.float32)
        mask = np.array(mask).astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std
        return {'image': img,
                'label': mask}

    def __call__(self, sample1, sample2):
        return self._normalize(sample1), self._normalize(sample2)


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def _totensor(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = sample['image']
        mask = sample['label']
        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        mask = np.array(mask).astype(np.float32)

        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).float()

        return {'image': img,
                'label': mask}

    def __call__(self, sample1, sample2):
        return self._totensor(sample1), self._totensor(sample2) 


class RandomHorizontalFlip(object):
    def _horizontal_flip(self, sample, is_flip):
        img = sample['image']
        mask = sample['label']
        if is_flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        return {'image': img,
                'label': mask}
    def __call__(self, sample1, sample2):
        is_flip = random.random() < 0.5
        sample1_new = self._horizontal_flip(sample1, is_flip)
        sample2_new = self._horizontal_flip(sample2, is_flip)
        return sample1_new, sample2_new


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree
    
    def _rotate(self, sample, rotate_degree):
        img = sample['image']
        mask = sample['label']
        img = img.rotate(rotate_degree, Image.BILINEAR)
        mask = mask.rotate(rotate_degree, Image.NEAREST)
        return {'image': img,
                'label': mask}

    def __call__(self, sample1, sample2):
        rotate_degree = random.uniform(-1*self.degree, self.degree)        
        return self._rotate(sample1, rotate_degree), self._rotate(sample2, rotate_degree)


class RandomGaussianBlur(object):
    def _gaussian_blur(self, sample, is_blur):
        img = sample['image']
        mask = sample['label']
        if is_blur:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))
        return {'image': img,
                'label': mask}

    def __call__(self, sample1, sample2):
        is_blur = random.random() < 0.5
        sample1_new = self._gaussian_blur(sample1, is_blur)
        sample2_new = self._gaussian_blur(sample2, is_blur)
        return sample1_new, sample2_new


class RandomScaleCrop(object):
    def __init__(self, base_size, crop_size, fill=0):
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill

    def __call__(self, sample1, sample2):
        img1 = sample1['image']
        mask1 = sample1['label']
        img2 = sample2['image']
        mask2 = sample2['label']
        # random scale (short edge)
        short_size = random.randint(int(self.base_size * 0.8), int(self.base_size * 1.2))
        w, h = img1.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img1 = img1.resize((ow, oh), Image.BILINEAR)
        mask1 = mask1.resize((ow, oh), Image.NEAREST)
        img2 = img2.resize((ow, oh), Image.BILINEAR)
        mask2 = mask2.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            img1 = ImageOps.expand(img1, border=(0, 0, padw, padh), fill=0)
            mask1 = ImageOps.expand(mask1, border=(0, 0, padw, padh), fill=self.fill)
            img2 = ImageOps.expand(img2, border=(0, 0, padw, padh), fill=0)
            mask2 = ImageOps.expand(mask2, border=(0, 0, padw, padh), fill=self.fill)
        # random crop crop_size
        w, h = img1.size
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)
        img1 = img1.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask1 = mask1.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        img2 = img2.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask2 = mask2.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'image': img1, 'label': mask1}, {'image': img2, 'label': mask2}


class FixScaleCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample1, sample2):
        img1 = sample1['image']
        mask1 = sample1['label']
        img2 = sample2['image']
        mask2 = sample2['label']
        w, h = img1.size
        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)
        img1 = img1.resize((ow, oh), Image.BILINEAR)
        mask1 = mask1.resize((ow, oh), Image.NEAREST)
        img2 = img2.resize((ow, oh), Image.BILINEAR)
        mask2 = mask2.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = img1.size
        x1 = int(round((w - self.crop_size) / 2.))
        y1 = int(round((h - self.crop_size) / 2.))
        img1 = img1.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask1 = mask1.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        img2 = img2.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask2 = mask2.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'image': img1, 'label': mask1}, {'image': img2, 'label': mask2}


class FixedResize(object):
    def __init__(self, size):
        self.size = (size, size)  # size: (h, w)
    
    def _resize(self, sample):
        img = sample['image']
        mask = sample['label']

        assert img.size == mask.size

        img = img.resize(self.size, Image.BILINEAR)
        mask = mask.resize(self.size, Image.NEAREST)

        return {'image': img,
                'label': mask}      

    def __call__(self, sample1, sample2):
        return self._resize(sample1), self._resize(sample2)

 
class RandomZoom(object):
    def __init__(self, base_size, crop_size, fill=0):
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill

    def __call__(self, sample1, sample2):
        img1 = sample1['image']
        mask1 = sample1['label']
        img2 = sample2['image']
        mask2 = sample2['label']
        # random scale (short edge)
        short_size = random.randint(int(self.base_size * 0.8), int(self.base_size * 1.05))
        w, h = img1.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img1 = img1.resize((ow, oh), Image.BILINEAR)
        mask1 = mask1.resize((ow, oh), Image.NEAREST)
        img2 = img2.resize((ow, oh), Image.BILINEAR)
        mask2 = mask2.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0  
            _ph = int(random.uniform(0, padh / 2))
            _pw = int(random.uniform(0, padw / 2))
            img1 = ImageOps.expand(img1, border=(_pw, _ph, padw-_pw, padh-_ph), fill=0)
            mask1 = ImageOps.expand(mask1, border=(_pw, _ph, padw-_pw, padh-_ph), fill=self.fill)
            img2 = ImageOps.expand(img2, border=(_pw, _ph, padw-_pw, padh-_ph), fill=0)
            mask2 = ImageOps.expand(mask2, border=(_pw, _ph, padw-_pw, padh-_ph), fill=self.fill)
        # random crop crop_size
        w, h = img1.size
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)
        img1 = img1.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask1 = mask1.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        img2 = img2.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask2 = mask2.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'image': img1, 'label': mask1}, {'image': img2, 'label': mask2}
