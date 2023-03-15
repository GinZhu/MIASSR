from datasets.basic_dataset import MIBasicTrain, MIBasicValid
from datasets.basic_dataset import SingleImageRandomCrop, SRImagePairRandomCrop
from metrics.sr_evaluation import BasicSREvaluation, MetaSREvaluation
import numpy as np
import cv2

from os.path import join
from glob import glob

from multiprocessing import Pool

"""
Dataset for nature images (e.g. DIV2K):
    1. read .png images;
    2. covert to YCbCr and only use the Y channel;
    3. training images and validation images are in different folders
    4. No testing dataset;
    5. this is to pre-train the meta-sr model

Under working ... :
    1. RGBRawDataset
    2. RGBMetaSRDataset
    3. RGBSRDataset
    4. RGBMetaSREvaluation (should be the same as for medical images?)
    
Test passed:

@Jin (jin.zhu@cl.cam.ac.uk) Sep 15 2020
"""


class RGBRawDataset(MIBasicTrain, MIBasicValid):
    """
    Loading data from the DIV2K dataset for training / validation
    Image data example information:
        (2040, 1356, 3) 255 0
    Note:
        Images are with different shapes
    To pre-process:
        0. save images in a list
        1. loading training and validation images from different folders
        2. normalise;
        3. merge to a list
    """

    def __init__(self, data_folder, toy_problem=True, color='YCbCr', multi_threads=8, norm=True):
        super(RGBRawDataset, self).__init__()

        self.toy_problem = toy_problem
        self.color = color
        color_channels = {
            'RGB': 3,
            'YCbCr': 1,
        }
        self.input_channels = color_channels[color]

        self.multi_pool = Pool(multi_threads)

        # ## data loading
        self.raw_data_folder = data_folder
        train_folder = 'DIV2K_train_HR'
        valid_folder = 'DIV2K_valid_HR'
        train_image_paths = sorted(glob(join(self.raw_data_folder, train_folder, '*.png')))
        valid_image_paths = sorted(glob(join(self.raw_data_folder, valid_folder, '*.png')))

        if self.toy_problem:
            train_image_paths = train_image_paths[:20]
            valid_image_paths = valid_image_paths[:5]

        self.training_imgs = [self.imread(p) for p in train_image_paths]
        self.testing_imgs = [self.imread(p) for p in valid_image_paths]

        self.training_img_ids = [p.split('/')[-1].replace('.png', '') for p in train_image_paths]
        self.testing_img_ids = [p.split('/')[-1].replace('.png', '') for p in valid_image_paths]

        # ## make all images as zero-mean-unit-variance
        # ## note: this will be done in the model
        self.norm = norm
        self.mean = [0. for _ in range(self.input_channels)]
        self.std = [1. for _ in range(self.input_channels)]
        mean, std = self.cal_mean_std(self.training_imgs)
        if 'zero_mean' in self.norm:
            self.mean = mean
        if 'unit_std' in self.norm:
            self.std = std

    def imread(self, path):
        img = cv2.imread(path)
        if self.color == 'YCbCr':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)[:, :, :1]
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255
        return img

    @staticmethod
    def cal_mean_std(imgs):
        """
        Calulate the mean and std for a list of images with different shape
        :param imgs: a list of images, with shape (H, W, C) where c = 1/3/...
        :return: [mean, std] where mean -> (x, x, x) or (x,)
        """
        means = [np.mean(img, axis=(0, 1)) for img in imgs]
        variances = [np.var(img, axis=(0, 1)) for img in imgs]

        mean = np.mean(means, axis=0)
        v = np.mean(variances, axis=0) + np.var(means, axis=0)
        return [mean, np.sqrt(v)]

    def __len__(self):
        return len(self.training_imgs)

    def __getitem__(self, item):
        pass

    def test_len(self):
        return len(self.testing_imgs)

    def get_test_pair(self, item):
        pass


class RGBMetaSRDataset(RGBRawDataset):
    """
    Behaviours:
        1. generate training patches in a batch;
        2. patches in each batch have the same sr_factor;
        3. how to design validation process?
        4. how to define SR-Evaluation?
    """

    def __init__(self, paras):
        data_folder = paras.data_folder
        toy_problem = paras.toy_problem
        color = paras.color_mode
        multi_threads = paras.multi_threads
        norm = paras.normal_inputs

        super(RGBMetaSRDataset, self).__init__(data_folder, toy_problem, color, multi_threads, norm)

        self.sr_factors = paras.all_sr_scales
        self.batch_size = paras.batch_size
        self.return_res_image = paras.return_res_image

        # ## generate LR - HR pairs when get item
        self.training_outputs = self.training_imgs

        # patch size is converted to LR dimensions
        self.patch_size = paras.patch_size

        self.testing_gts = self.testing_imgs

        # ## crop function, with LR patch size
        self.random_crop = SingleImageRandomCrop(self.patch_size, 0)

        self.test_sr_factors = paras.test_sr_scales

        # ## eva function
        quick_eva_metrics = paras.quick_eva_metrics
        final_eva_metrics = paras.eva_metrics
        eva_gpu = paras.eva_gpu_id
        self.quick_eva_func = MetaSREvaluation(quick_eva_metrics, self.test_sr_factors, eva_gpu, 'mean')
        self.final_eva_func = MetaSREvaluation(final_eva_metrics, self.test_sr_factors, eva_gpu, 'full')

        self.test_crop = SingleImageRandomCrop(256, 0)

    def __getitem__(self, item):
        # ## return a batch every time
        ids = np.random.choice(self.__len__(), self.batch_size, False)
        sr_factor = np.random.choice(self.sr_factors)
        img_outputs = []
        for i in ids:
            img = self.training_outputs[i]
            img = self.random_crop(img)
            img_outputs.append(img)
        img_inputs = [self.resize([_, int(self.patch_size/sr_factor), 'cubic', 'gaussian']) for _ in img_outputs]
        img_outputs = [self.resize([_, int(int(self.patch_size/sr_factor)*sr_factor)]) for _ in img_outputs]

        if self.return_res_image:
            res_imgs = [self.resize([_, int(int(self.patch_size/sr_factor)*sr_factor)]) for _ in img_inputs]
            res_imgs = self.numpy_2_tensor(res_imgs)
        else:
            res_imgs = [[]] * self.batch_size

        img_inputs = self.numpy_2_tensor(img_inputs)
        img_outputs = self.numpy_2_tensor(img_outputs)

        return {'in': img_inputs, 'out': img_outputs, 'sr_factor': sr_factor, 'res': res_imgs}

    def get_test_pair(self, item):
        # # return gt and LR images with various sr_factors
        ori_img = self.testing_gts[item]
        # ## have to cut the ori_img to smaller patch for GPU memory limitations
        ori_img = self.test_crop(ori_img)
        H, W = ori_img.shape[:2]

        img_inputs = [self.resize([ori_img, (int(H//s), int(W//s)), 'cubic', 'gaussian']) for s in self.test_sr_factors]
        img_outputs = [
            self.resize([ori_img, (int(H//s*s), int(W//s*s))]) for s in self.test_sr_factors
        ]

        if self.return_res_image:
            res_imgs = [self.resize([img, (int(H//s*s), int(W//s*s))])
                        for img, s in zip(img_inputs, self.test_sr_factors)]
            res_imgs = [self.numpy_2_tensor(_).unsqueeze(0) for _ in res_imgs]
        else:
            res_imgs = [[]] * len(self.test_sr_factors)

        img_id = self.testing_img_ids[item]

        img_inputs = [self.numpy_2_tensor(_) for _ in img_inputs]
        img_inputs = [img.unsqueeze(0) for img in img_inputs]

        sample = {}
        for img_in, img_out, s, res in zip(img_inputs, img_outputs, self.test_sr_factors, res_imgs):
            sample[s] = {'in': img_in, 'gt': img_out, 'sr_factor': s, 'id': img_id, 'res': res}

        return sample


class RGBSRDataset(RGBRawDataset):

    def __init__(self, paras):

        data_folder = paras.data_folder
        toy_problem = paras.toy_problem
        color = paras.color_mode
        multi_threads = paras.multi_threads
        norm = paras.normal_inputs

        super(RGBSRDataset, self).__init__(data_folder, toy_problem, color, multi_threads, norm)

        self.sr_factor = paras.sr_scale
        self.return_res_image = paras.return_res_image

        # ## prepare LR - HR pairs
        self.training_outputs = self.training_imgs
        self.training_inputs = self.multi_pool.map(
            self.resize, [[_, 1/self.sr_factor, 'cubic', 'gaussian'] for _ in self.training_imgs]
        )

        # ## patch size is converted to LR dimensions
        self.patch_size = paras.patch_size // int(self.sr_factor)

        # ## prepare testing LR - HR image pairs
        self.testing_gts = self.testing_imgs
        self.testing_inputs = self.multi_pool.map(
            self.resize, [[_, 1/self.sr_factor, 'cubic', 'gaussian'] for _ in self.testing_imgs]
        )

        # ## crop function, with LR patch size
        self.random_crop = SRImagePairRandomCrop(self.patch_size, self.sr_factor, 0)

        # ## eva function
        quick_eva_metrics = paras.quick_eva_metrics  # 'psnr ssim' for simple case
        final_eva_metrics = paras.eva_metrics
        eva_gpu = paras.eva_gpu_id
        self.quick_eva_func = BasicSREvaluation(quick_eva_metrics, self.sr_factor, eva_gpu, 'mean')
        self.final_eva_func = BasicSREvaluation(final_eva_metrics, self.sr_factor, eva_gpu, 'full')

        self.test_crop = SRImagePairRandomCrop(256, self.sr_factor, 0)

    def __getitem__(self, item):
        img_input = self.training_inputs[item]
        img_output = self.training_outputs[item]

        img_input, img_output = self.random_crop([img_input, img_output])

        if self.return_res_image:
            res_img = self.resize(
                [img_input, img_output.shape[:2]]
            )
            res_img = self.numpy_2_tensor(res_img)
        else:
            res_img = []

        img_input = self.numpy_2_tensor(img_input)
        img_output = self.numpy_2_tensor(img_output)

        return {'in': img_input, 'out': img_output, 'res': res_img}

    def get_test_pair(self, item):
        img_input = self.testing_inputs[item]
        img_output = self.testing_gts[item]

        img_input, img_output = self.test_crop([img_input, img_output])

        if self.return_res_image:
            res_img = self.resize(
                [img_input, img_output.shape[:2]]
            )
            res_img = self.numpy_2_tensor(res_img)
        else:
            res_img = []

        img_input = self.numpy_2_tensor(img_input).unsqueeze(0)

        return {'in': img_input, 'gt': img_output, 'id': item, 'res': res_img}


