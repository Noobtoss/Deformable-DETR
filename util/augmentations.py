import random
import numpy as np
import torch

class Albumentations:
    
    def __init__(self, size=640):
        self.transform = None
        self.transform_pre = None
        prefix = colorstr('albumentations: ')
        try:
            import albumentations as A
            check_version(A.__version__, '1.0.3', hard=True)  # version requirement
            version = 7 # 0=standAug 1=noAug 2=selectAug 3=selectAug+ 4=standAugSelectAug+scaleAug 5=scaleAug 6=Dropout 7=Dropout+standAug
            
            if version == 0:
                
                T = [
                    A.RandomResizedCrop(height=size, width=size, scale=(0.8, 1.0), ratio=(0.9, 1.11), p=0.0),
                    A.Blur(p=0.01),
                    A.MedianBlur(p=0.01),
                    A.ToGray(p=0.01),
                    A.CLAHE(p=0.01),
                    A.RandomBrightnessContrast(p=0.0),
                    A.RandomGamma(p=0.0),
                    A.ImageCompression(quality_lower=75, p=0.0)]  # transforms
                
                self.transform = A.Compose(T, bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
            
            if version == 1:
                
                T = [A.Blur(p=0.0)]
                
                self.transform = A.Compose(T, bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
            
            if version == 2:
                
                T1 = [A.Equalize(p=0.2)]
                T2 = [A.RGBShift(p=0.2, r_shift_limit=30, g_shift_limit=30, b_shift_limit=30)]
                T3 = [A.HueSaturationValue(p=0.2, hue_shift_limit=30, sat_shift_limit=40, val_shift_limit=30)]
                T4 = [A.ISONoise(p=0.2)]
                T5 = [A.RandomBrightnessContrast(p=0.2, brightness_limit=0.3, contrast_limit=0.3)]
                
                self.transform = [
                            A.Compose(T1, bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])),
                            A.Compose(T2, bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])),
                            A.Compose(T2, bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])),
                            A.Compose(T3, bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])),
                            A.Compose(T4, bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])),
                            A.Compose(T5, bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
                            ]

            if version == 3:
                 
                T1 = [A.Equalize(p=0.2)]
                T2 = [A.RGBShift(p=0.4, r_shift_limit=30, g_shift_limit=30, b_shift_limit=30)]
                T3 = [A.HueSaturationValue(p=0.4, hue_shift_limit=30, sat_shift_limit=40, val_shift_limit=30)]
                T4 = [A.ISONoise(p=0.2)]
                T5 = [A.RandomBrightnessContrast(p=0.2, brightness_limit=0.3, contrast_limit=0.3)]
            
                self.transform = [
                            A.Compose(T1, bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])),
                            A.Compose(T2, bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])),
                            A.Compose(T2, bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])),
                            A.Compose(T3, bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])),
                            A.Compose(T4, bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])),
                            A.Compose(T5, bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
                            ]

            if version == 4:

                T0 = [
                    A.Blur(p=0.01),
                    A.MedianBlur(p=0.01),
                    A.ToGray(p=0.01),
                    A.CLAHE(p=0.01),] #transforms
                 
                T = [
                    A.ShiftScaleRotate(shift_limit=0.0, scale_limit=(-0.4, 0.1), rotate_limit=0, interpolation=1, border_mode=0, rotate_method='ellipse', p=0.5),
                    A.ShiftScaleRotate(shift_limit=0.0, scale_limit=0.2, rotate_limit=45, interpolation=1, border_mode=0, rotate_method='ellipse', p=0.5)]

                T1 = [A.Equalize(p=0.1)]
                T2 = [A.RGBShift(p=0.2, r_shift_limit=30, g_shift_limit=30, b_shift_limit=30)]
                T3 = [A.HueSaturationValue(p=0.2, hue_shift_limit=30, sat_shift_limit=40, val_shift_limit=30)]
                T4 = [A.ISONoise(p=0.1)]
                T5 = [A.RandomBrightnessT0+T+A.Contrast(p=0.1, brightness_limit=0.3, contrast_limit=0.3)]

                self.transform = [
                            A.Compose(T0+T+T1, bbox_params=A.BboxParams(format='yolo', min_area=100, label_fields=['class_labels'])),
                            A.Compose(T0+T+T2, bbox_params=A.BboxParams(format='yolo', min_area=100, label_fields=['class_labels'])),
                            A.Compose(T0+T+T2, bbox_params=A.BboxParams(format='yolo', min_area=100, label_fields=['class_labels'])),
                            A.Compose(T0+T+T3, bbox_params=A.BboxParams(format='yolo', min_area=100, label_fields=['class_labels'])),
                            A.Compose(T0+T+T4, bbox_params=A.BboxParams(format='yolo', min_area=100, label_fields=['class_labels'])),
                            A.Compose(T0+T+T5, bbox_params=A.BboxParams(format='yolo', min_area=100, label_fields=['class_labels']))
                            ]

            if version == 5:

                T = [
                    A.ShiftScaleRotate(shift_limit=0.0, scale_limit=(-0.4, 0.1), rotate_limit=0, interpolation=1, border_mode=0, rotate_method='ellipse', p=1.0),
                    A.ShiftScaleRotate(shift_limit=0.0, scale_limit=0.2, rotate_limit=45, interpolation=1, border_mode=0, rotate_method='ellipse', p=1.0)]

                self.transform = A.Compose(T, bbox_params=A.BboxParams(format='yolo', min_area=100, label_fields=['class_labels']))

            if version == 6:

                T_pre = [A.CoarseDropout(max_holes=20, max_height=0.03, max_width=0.03, min_holes=10, min_height=0.01, min_width=0.01, fill_value=0, p=0.1)]

                T = [
                    A.PixelDropout(dropout_prob=0.05, per_channel=False, drop_value=0, p=0.1),
                    A.ShiftScaleRotate(shift_limit=0.0, scale_limit=(-0.4, 0.1), rotate_limit=0, interpolation=1, border_mode=0, rotate_method='ellipse', p=1.0),
                    A.ShiftScaleRotate(shift_limit=0.0, scale_limit=0.2, rotate_limit=45, interpolation=1, border_mode=0, rotate_method='ellipse', p=1.0)
                    ]

                self.transform_pre = A.Compose(T_pre)
                self.transform = A.Compose(T, bbox_params=A.BboxParams(format='yolo', min_area=100, label_fields=['class_labels']))
                    
            if version == 7:
               
                T_pre = [A.CoarseDropout(max_holes=20, max_height=0.03, max_width=0.03, min_holes=10, min_height=0.01, min_width=0.01, fill_value=0, p=0.05)]
                
                T = [
                    A.PixelDropout(dropout_prob=0.05, per_channel=False, drop_value=0, p=0.05),
                    A.ShiftScaleRotate(shift_limit=0.0, scale_limit=(-0.4, 0.1), rotate_limit=0, interpolation=1, border_mode=0, rotate_method='ellipse', p=0.2),
                    A.ShiftScaleRotate(shift_limit=0.0, scale_limit=0.2, rotate_limit=45, interpolation=1, border_mode=0, rotate_method='ellipse', p=0.2),
                    A.RandomResizedCrop(height=size, width=size, scale=(0.8, 1.0), ratio=(0.9, 1.11), p=0.0),
                    A.Blur(p=0.05),
                    A.MedianBlur(p=0.05),
                    A.ToGray(p=0.05),
                    A.CLAHE(p=0.05),
                    A.RandomBrightnessContrast(p=0.0),
                    A.RandomGamma(p=0.0),
                    A.ImageCompression(quality_lower=75, p=0.0)
                    ]

                self.transform_pre = A.Compose(T_pre)
                self.transform = A.Compose(T, bbox_params=A.BboxParams(format='yolo', min_area=100, label_fields=['class_labels']))
                              
            LOGGER.info(prefix + ', '.join(f'{x}'.replace('always_apply=False, ', '') for x in T if x.p))
        except ImportError:#  # package not installed, skip
            pass
        except Exception as e:
            LOGGER.info(f'{prefix}{e}')

    def __call__(self, im, labels, p=1.0):

        if random.random() < p:

            if self.transform_pre:
                new = self.transform_pre(image=im) # transformed
                im = new['image']

            if self.transform:

                if isinstance(self.transform, list):
                    new = random.choice(self.transform)(image=im, bboxes=labels[:, 1:], class_labels=labels[:, 0]) # transformed
                else:
                    new = self.transform(image=im, bboxes=labels[:, 1:], class_labels=labels[:, 0])  # transformed
                im, labels = new['image'], np.array([[c, *b] for c, b in zip(new['class_labels'], new['bboxes'])])

        return im, labels
