import numpy as np
import cv2
from abc import ABC, abstractmethod
from torch.utils.data import Dataset
import torch


"""
Basic Dataset:
    1. MedicalImageBasicDataset;
    2. MIBasicTrain;
    3. MIBasicValid;

Basic Evaluation

Test passed.

@Jin (jin.zhu@cl.cam.ac.uk) June 22 2020

@Jin Zhu (jin.zhu@cl.cam.ac.uk) Modified at Oct 23 2020
"""


class MedicalImageBasicDataset(Dataset):
    """
    Basic Dataset for medical images, by default it provides four functions:
        1, 2 numpy_2_tensor() and tensor_2_numpy();
        3. normalize;
        4. resize()
    """
    def __init__(self):
        self.inputs = []

    def __len__(self):
        return len(self.inputs)

    @staticmethod
    def numpy_2_tensor(a):
        if isinstance(a, list):
            a = np.array(a)
        if a.ndim == 3:
            return torch.tensor(a.transpose(2, 0, 1), dtype=torch.float)
        elif a.ndim == 4:
            return torch.tensor(a.transpose(0, 3, 1, 2), dtype=torch.float)
        else:
            raise ValueError('Image should have 3 or 4 channles')

    @staticmethod
    def tensor_2_numpy(t):
        if t.ndim == 3:
            return t.detach().cpu().numpy().transpose(1, 2, 0)
        elif t.ndim == 4:
            return t.detach().cpu().numpy().transpose(0, 2, 3, 1)
        else:
            return t.detach().cpu().numpy()

    @staticmethod
    def normalize(imgs):
        min_val = np.min(imgs)
        max_val = np.max(imgs)
        imgs_norm = (imgs - min_val) / (max_val - min_val)
        return imgs_norm, min_val, max_val

    @staticmethod
    def resize(data):
        """
        data:
          [img, size, interpolation_method, blur_method, blur_kernel, blur_sigma]
        cv2 coordinates:
          [horizontal, vertical], which is different as numpy array image.shape
          'cubic': cv2.INTER_LINEAR
          'linear': cv2.INTER_CUBIC
          'nearest' or None(default): cv2.INTER_NEAREST
        Caution: cubic interpolation may generate values out of original data range (e.g. negative values)

        """
        data += [None, ] * (6 - len(data))

        img, size, interpolation_method, blur_method, blur_kernel, blur_sigma = data

        #
        if interpolation_method == 'nearest':
            interpolation_method = cv2.INTER_NEAREST
        elif interpolation_method is None or interpolation_method == 'cubic':
            interpolation_method = cv2.INTER_CUBIC
        elif interpolation_method == 'linear':
            interpolation_method = cv2.INTER_LINEAR
        else:
            raise ValueError('cv2 Interpolation methods: None, nearest, cubic, linear')

        if blur_kernel is None:
            blur_kernel = 3
        if blur_sigma is None:
            blur_sigma = 0

        # calculate the output size
        if isinstance(size, (float, int)):
            size = [size, size]
        if not isinstance(size, (list, tuple)):
            raise TypeError('The input Size of LR image should be (float, int, list or tuple)')
        if isinstance(size[0], float):
            size = int(img.shape[0] * size[0]), int(img.shape[1] * size[1])
        if size[0] <= 0 or size[1] <= 0:
            raise ValueError('Size of output image should be positive')

        # resize the image
        if size[0] == img.shape[0] and size[1] == img.shape[1]:
            output_img = img
        else:
            # opencv2 is [horizontal, vertical], so the output_size should be reversed
            size = size[1], size[0]
            output_img = cv2.resize(img, dsize=size, interpolation=interpolation_method)

        # blur the image if necessary
        if blur_method == 'gaussian':
            output_img = cv2.GaussianBlur(output_img, (blur_kernel, blur_kernel), blur_sigma)
        else:
            # todo: add more blur methods
            pass

        if img.ndim != output_img.ndim:
            output_img = output_img[:, :, np.newaxis]
        return output_img


class MIBasicTrain(MedicalImageBasicDataset, ABC):

    """
    Abstract dataset, for training. Similar as pytorch Dataset
    """

    def __init__(self):
        super(MIBasicTrain, self).__init__()
        pass

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, item):
        pass


class MIBasicValid(MedicalImageBasicDataset, ABC):

    """
    Abstract Dataset, for validation.
    As a subclass of pytorch Dataset, but instead using __len__ and __getitem__,
    we should call test_len() and get_test_pair(), to avoid confusion.

    This dataset will also provide two evaluation functions:
        1. quick_eva_func: to valid the training process;
        2. final_eva_func: to know the final performance of model;
    """

    def __init__(self):
        super(MIBasicValid, self).__init__()
        self.quick_eva_func = None
        self.final_eva_func = None

    def __len__(self):
        return self.test_len()

    def __getitem__(self, item):
        return self.get_test_pair(item)

    @abstractmethod
    def test_len(self):
        # return the length of all data
        pass

    @abstractmethod
    def get_test_pair(self, item):
        # return a sample of all data
        pass

    def get_quick_eva_func(self):
        return self.quick_eva_func

    def get_final_eva_func(self):
        return self.final_eva_func

    def get_quick_eva_metrics(self):
        return self.quick_eva_func.get_metrics()

    def get_final_eva_metrics(self):
        return self.final_eva_func.get_metrics()


class BasicCropTransform(ABC):
    # ## todo: test new size / margin behaviours
    def __init__(self, size, margin):
        if isinstance(size, int):
            self.size = (size, size)
        elif isinstance(size, (list, tuple)):
            if all(isinstance(_, int) for _ in size):
                self.size = size
            else:
                raise TypeError('Crop size should be int, list(int) or tuple(int)')
        else:
            raise TypeError('Crop size should be int, list(int), or tuple(int)')

        if self.size[0] == 0 and self.size[1] == 0:
            self.size = None

        if isinstance(margin, int):
            self.margin = (margin, margin)
        elif isinstance(margin, (list, tuple)):
            if all(isinstance(_, int) for _ in margin):
                self.margin = margin
            else:
                raise TypeError('Crop margin should be int, list(int) or tuple(int)')
        else:
            raise TypeError('Crop margin should be int, list(int), or tuple(int)')

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass


class SingleImageRandomCrop(BasicCropTransform):

    def __init__(self, size, margin=0):
        super(SingleImageRandomCrop, self).__init__(size, margin)

    def __call__(self, in_img):
        if self.size is None:
            return in_img[self.margin[0]:-self.margin[0], self.margin[1]:-self.margin[1]]
        else:
            ori_H, ori_W = in_img.shape[:2]
            x_top_left = np.random.randint(
                self.margin[0], ori_H - self.size[0] - self.margin[0]
            )
            y_top_left = np.random.randint(
                self.margin[1], ori_W - self.size[1] - self.margin[1]
            )

            return in_img[x_top_left:x_top_left + self.size[0], y_top_left:y_top_left + self.size[1]]


class SRImagePairRandomCrop(BasicCropTransform):

    def __init__(self, size, sr_factor, margin=0):
        """
        Randomly crop a [lr, hr] pair correspondingly and return patches.
        :param size: patch size, if 0 return the full image without boundaries (margins)
        :param sr_factor: should be int
        :param margin: This margin is corresponded to the HR image, boundaries of the image
        """
        super(SRImagePairRandomCrop, self).__init__(size, margin)
        self.sr_factor = int(sr_factor)

        self.margin = [_//self.sr_factor for _ in self.margin]

    def __call__(self, data):
        in_img, out_img = data
        if self.size is None:
            cropped_data = [
                in_img[self.margin[0]:-self.margin[0], self.margin[1]:-self.margin[1]],
                out_img[self.margin[0]*self.sr_factor:-self.margin[0]*self.sr_factor,
                        self.margin[1]*self.sr_factor:-self.margin[1]*self.sr_factor]
            ]
        else:
            ori_H, ori_W = in_img.shape[:2]
            x_top_left = np.random.randint(
                self.margin[0], ori_H - self.size[0] - self.margin[0]
            )
            y_top_left = np.random.randint(
                self.margin[1], ori_W - self.size[1] - self.margin[1]
            )
            cropped_data = [
                in_img[x_top_left:x_top_left + self.size[0], y_top_left:y_top_left + self.size[1]],
                out_img[
                    x_top_left*self.sr_factor:(x_top_left+self.size[0])*self.sr_factor,
                    y_top_left*self.sr_factor:(y_top_left+self.size[1])*self.sr_factor
                ]
            ]
        return cropped_data


class CentreCrop(BasicCropTransform):

    def __init__(self, size):
        super(CentreCrop, self).__init__(size, 0)

    def __call__(self, in_img):
        ori_H, ori_W = in_img.shape[:2]
        x_top_left = (ori_H - self.size[0]) // 2
        x_top_left = 0 if x_top_left < 0 else x_top_left
        y_top_left = (ori_W - self.size[1]) // 2
        y_top_left = 0 if y_top_left < 0 else y_top_left
        return in_img[x_top_left:x_top_left + self.size[0], y_top_left:y_top_left + self.size[1]]

