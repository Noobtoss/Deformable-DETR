import random
import numpy as np
import torch

import torchvision.transforms.functional as F


class ToNp(object):
    def __call__(self, img, target):
        return np.asarray(img), {k: np.asarray(v) for k, v in target.items()}


class ToTensor(object):
    def __call__(self, img, target):
        return img, {k: torch.from_numpy(v) for k, v in target.items()}


class Albumentations: # Semmel
    # YOLOv5 Albumentations class (optional, only used if package is installed)
    def __init__(self):
        self.transform = None
        self.transform_pre = None

        hyp = type('', (), {})()
        hyp.semmel_flag = 7
        hyp.semmel_prob = 0.05
        
        try:
            import albumentations as A
            # hyp.semmel_flag # 0=standAug 1=noAug 2=selectAug 3=selectAug+ 4=standAugSelectAug+scaleAug 5=scaleAug 6=Dropout 7=Dropout+standAug
            
            if hyp.semmel_flag == 0:
                
                T = [
                    A.Blur(p=hyp.semmel_prob),
                    A.MedianBlur(blur_limit=5, p=hyp.semmel_prob), # Semmel blur_limit=5 had to be changed for some reason
                    A.ToGray(p=hyp.semmel_prob),
                    A.CLAHE(p=hyp.semmel_prob),
                    A.RandomBrightnessContrast(p=0.0),
                    A.RandomGamma(p=0.0),
                    A.ImageCompression(quality_lower=75, p=0.0)]  # transforms
                
                self.transform = A.Compose(T, bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
            
            if hyp.semmel_flag == 1:
                
                T = [A.Blur(p=0.0)]
                
                self.transform = A.Compose(T, bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
            
            if hyp.semmel_flag == 2:
                
                T1 = [A.Equalize(p=hyp.semmel_prob)]
                T2 = [A.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=hyp.semmel_prob)]
                T3 = [A.HueSaturationValue(hue_shift_limit=30, sat_shift_limit=40, val_shift_limit=30, p=hyp.semmel_prob)]
                T4 = [A.ISONoise(p=hyp.semmel_prob)]
                T5 = [A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=hyp.semmel_prob)]
                
                self.transform = [
                            A.Compose(T1, bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'])),
                            A.Compose(T2, bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'])),
                            A.Compose(T2, bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'])),
                            A.Compose(T3, bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'])),
                            A.Compose(T4, bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'])),
                            A.Compose(T5, bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
                            ]

            if hyp.semmel_flag == 3:
                 
                T1 = [A.Equalize(p=hyp.semmel_prob)]
                T2 = [A.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=hyp.semmel_prob*2)]
                T3 = [A.HueSaturationValue(hue_shift_limit=30, sat_shift_limit=40, val_shift_limit=30, p=hyp.semmel_prob*2)]
                T4 = [A.ISONoise(p=hyp.semmel_prob)]
                T5 = [A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=hyp.semmel_prob)]
            
                self.transform = [
                            A.Compose(T1, bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'])),
                            A.Compose(T2, bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'])),
                            A.Compose(T2, bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'])),
                            A.Compose(T3, bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'])),
                            A.Compose(T4, bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'])),
                            A.Compose(T5, bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
                            ]

            if hyp.semmel_flag == 4:

                T0 = [
                    A.Blur(p=hyp.semmel_prob),
                    A.MedianBlur(p=hyp.semmel_prob),
                    A.ToGray(p=hyp.semmel_prob),
                    A.CLAHE(p=hyp.semmel_prob),] #transforms
                 
                T = [
                    A.ShiftScaleRotate(shift_limit=0.0, scale_limit=(-0.4, 0.1), rotate_limit=0, interpolation=1, border_mode=0, rotate_method='ellipse', p=hyp.semmel_prob*4),
                    A.ShiftScaleRotate(shift_limit=0.0, scale_limit=0.2, rotate_limit=45, interpolation=1, border_mode=0, rotate_method='ellipse', p=hyp.semmel_prob*4)]

                T1 = [A.Equalize(p=hyp.semmel_prob)]
                T2 = [A.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=hyp.semmel_prob*2)]
                T3 = [A.HueSaturationValue(hue_shift_limit=30, sat_shift_limit=40, val_shift_limit=30, p=hyp.semmel_prob*2)]
                T4 = [A.ISONoise(p=hyp.semmel_prob)]
                T5 = [A.RandomBrightnessT0+T+A.Contrast(brightness_limit=0.3, contrast_limit=0.3, p=hyp.semmel_prob)]

                self.transform = [
                            A.Compose(T0+T+T1, bbox_params=A.BboxParams(format='pascal_voc', min_area=100, label_fields=['class_labels'])),
                            A.Compose(T0+T+T2, bbox_params=A.BboxParams(format='pascal_voc', min_area=100, label_fields=['class_labels'])),
                            A.Compose(T0+T+T2, bbox_params=A.BboxParams(format='pascal_voc', min_area=100, label_fields=['class_labels'])),
                            A.Compose(T0+T+T3, bbox_params=A.BboxParams(format='pascal_voc', min_area=100, label_fields=['class_labels'])),
                            A.Compose(T0+T+T4, bbox_params=A.BboxParams(format='pascal_voc', min_area=100, label_fields=['class_labels'])),
                            A.Compose(T0+T+T5, bbox_params=A.BboxParams(format='pascal_voc', min_area=100, label_fields=['class_labels']))
                            ]

            if hyp.semmel_flag == 5:

                T = [
                    A.ShiftScaleRotate(shift_limit=0.0, scale_limit=(-0.4, 0.1), rotate_limit=0, interpolation=1, border_mode=0, rotate_method='ellipse', p=hyp.semmel_prob*4),
                    A.ShiftScaleRotate(shift_limit=0.0, scale_limit=0.2, rotate_limit=45, interpolation=1, border_mode=0, rotate_method='ellipse', p=hyp.semmel_prob*4)]

                self.transform = A.Compose(T, bbox_params=A.BboxParams(format='pascal_voc', min_area=100, label_fields=['class_labels']))

            if hyp.semmel_flag == 6:

                T_pre = [A.CoarseDropout(max_holes=50, max_height=0.1, max_width=0.1, min_holes=10, min_height=0.01, min_width=0.01, fill_value=0, p=hyp.semmel_prob*4)]

                T = [
                    A.PixelDropout(dropout_prob=0.05, per_channel=False, drop_value=0, p=hyp.semmel_prob*4),
                    A.ShiftScaleRotate(shift_limit=0.0, scale_limit=(-0.4, 0.1), rotate_limit=0, interpolation=1, border_mode=0, rotate_method='ellipse', p=hyp.semmel_prob*4),
                    A.ShiftScaleRotate(shift_limit=0.0, scale_limit=0.2, rotate_limit=45, interpolation=1, border_mode=0, rotate_method='ellipse', p=hyp.semmel_prob*4)
                    ]

                self.transform_pre = A.Compose(T_pre)
                self.transform = A.Compose(T, bbox_params=A.BboxParams(format='pascal_voc', min_area=100, label_fields=['class_labels']))
                    
            if hyp.semmel_flag == 7:
               
                T_pre = [A.CoarseDropout(max_holes=50, max_height=0.1, max_width=0.1, min_holes=10, min_height=0.01, min_width=0.01, fill_value=0, p=hyp.semmel_prob*4)]
                
                T = [
                    A.PixelDropout(dropout_prob=0.05, per_channel=False, drop_value=0, p=hyp.semmel_prob*4),
                    A.ShiftScaleRotate(shift_limit=0.0, scale_limit=(-0.4, 0.1), rotate_limit=0, interpolation=1, border_mode=0, rotate_method='ellipse', p=hyp.semmel_prob*4),
                    A.ShiftScaleRotate(shift_limit=0.0, scale_limit=0.2, rotate_limit=45, interpolation=1, border_mode=0, rotate_method='ellipse', p=hyp.semmel_prob*4),
                    A.Blur(p=hyp.semmel_prob),
                    A.MedianBlur(blur_limit=5, p=hyp.semmel_prob), # Semmel blur_limit=5 had to be changed for some reason
                    A.ToGray(p=hyp.semmel_prob),
                    A.CLAHE(p=hyp.semmel_prob),
                    A.RandomBrightnessContrast(p=0.0),
                    A.RandomGamma(p=0.0),
                    A.ImageCompression(quality_lower=75, p=0.0)
                    ]

                self.transform_pre = A.Compose(T_pre)
                self.transform = A.Compose(T, bbox_params=A.BboxParams(format='pascal_voc', min_area=100, label_fields=['class_labels']))
        except ImportError:#  # package not installed, skip
            pass

    def __call__(self, im, labels, p=1.0):
        im, labels = ToNp()(im, labels)
        h, w = im.shape[:2]
        cls = labels['labels']
        bboxes = labels['boxes']
        
        if random.random() < p:
            if self.transform_pre:
                new = self.transform_pre(image=im) # transformed
                im = new['image']
            if self.transform:
                if isinstance(self.transform, list):
                    new = random.choice(self.transform)(image=im, bboxes=bboxes, class_labels=cls) # transformed
                else:
                    new = self.transform(image=im, bboxes=bboxes, class_labels=cls) # transformed
                
                if len(new['class_labels']) > 0: # skip update if no bbox in new im
                    im = new['image']
                    labels['labels'] = np.array(new['class_labels'])
                    labels['boxes'] = np.array(new['bboxes']).astype(int)

        im, labels = ToTensor()(im, labels)
        
        return im, labels
