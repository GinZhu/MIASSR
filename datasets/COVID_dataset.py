from datasets.basic_dataset import MIBasicTrain, MIBasicValid
from datasets.basic_dataset import SRImagePairRandomCrop, SingleImageRandomCrop, CentreCrop
from metrics.sr_evaluation import BasicSREvaluation, MetaSREvaluation

import numpy as np
import nibabel as nib

from os.path import join
from os import listdir
from glob import glob
import copy

from multiprocessing import Pool

"""
Dataset for medical image segmentation:
    1. loading data with nib from the .nii / .mnc / ... file
    2. feed patch (image and label) to networks
    3. validation dataset:
    4. post processing
    
Dataset for medical image super-resolution:
    1. loading data with nib from the .mnc file
    2. feed patch to 
    
Training / Validation / Testing (GT + SR results)

Under working ... :
    OASISSRTest -> patient wise dataset for inference
    OASISMetaSRDataset 
    OASISMetaSRTest -> patient wise dataset for inference
    OASISSegTest -> patient wise dataset for inference

Test passed:
    OASISRawDataset
    OASISSegDataset
    OASISSRDataset

Todo: @ Oct 23 2020
    1. re-organise
    2. re-test the datasets
    
@Jin (jin.zhu@cl.cam.ac.uk) Aug 11 2020
"""


class COVIDRawDataset(MIBasicTrain, MIBasicValid):
    """
    Loading data from the OASIS dataset for training / validation
    Image data example information:
        OAS1_0041_MR1 (176, 208, 176, 1) 3046.0 0.0 231.66056320277733
    To pre-process:
        0. reshape to 3D
        1. select slices (remove background on three dimensions);
        2. normalise;
        3. merge to a list
    """

    def __init__(self, data_folder, training_patient_ids=None, validation_patient_ids=None, margin=40,
                 dim=2, toy_problem=True, multi_threads=8, norm=''):
        super(COVIDRawDataset, self).__init__()

        self.raw_data_folder = data_folder

        self.training_folder = 'Train'
        self.validation_folder = 'Validation'

        self.image_postfix = '_ct.nii.gz'
        self.label_postfix = '_seg.nii.gz'

        self.dim = dim
        self.margin = margin

        self.toy_problem = toy_problem

        self.multi_pool = Pool(multi_threads)

        # ## define training data and validation data
        if training_patient_ids is None and validation_patient_ids:
            raise ValueError('Validation list should be None as Training list')
        elif validation_patient_ids is None and training_patient_ids:
            raise ValueError('Training list should be None as Validation list')
        # ## if not specified, randomly choose training the training and validation datasets
        elif validation_patient_ids is None and training_patient_ids is None:
            training_patient_paths = glob(join(
                self.raw_data_folder, self.training_folder, '*{}'.format(self.image_postfix)
            ))
            training_patient_ids = [_.split('/')[-1].replace(self.image_postfix, '') for _ in training_patient_paths]
            validation_patient_paths = glob(join(
                self.raw_data_folder, self.validation_folder, '*{}'.format(self.image_postfix)
            ))
            validation_patient_ids = [_.split('/')[-1].replace(self.image_postfix, '') for _ in validation_patient_paths]

        if self.toy_problem:
            np.random.shuffle(training_patient_ids)
            np.random.shuffle(validation_patient_ids)
            self.training_patient_ids = training_patient_ids[:5]
            self.validation_patient_ids = validation_patient_ids[:1]
        else:
            self.training_patient_ids = training_patient_ids
            self.validation_patient_ids = validation_patient_ids

        self.norm_paras = {}

        self.training_imgs = []

        # ## loading training data and validation data
        # ## Training dataset should be merged and shuffled, while validation dataset should be patient-wise
        for pid in self.training_patient_ids:
            pid_data = self.load_data(pid, self.training_folder)
            for img in pid_data:
                self.training_imgs.append(img)

        # ## crop image with margin
        self.crop = SingleImageRandomCrop(0, self.margin)
        self.training_imgs = self.multi_pool.map(self.crop, self.training_imgs)

        # ## make all images as zero-mean-unit-variance
        # ## note: this will be done in the model
        self.norm = norm
        self.mean = [0.]
        self.std = [1.]
        if 'zero_mean' in self.norm and len(self.training_imgs):
            self.mean = np.mean(self.training_imgs, axis=(0, 1, 2))
        if 'unit_std' in self.norm and len(self.training_imgs):
            self.std = np.std(self.training_imgs, axis=(0, 1, 2))

        # ## loading validation dataset
        self.testing_imgs = []
        self.testing_img_ids = []
        for pid in self.validation_patient_ids:
            pid_data = self.load_data(pid, self.validation_folder)
            for img in pid_data:
                self.testing_imgs.append(img)
                self.testing_img_ids.append(pid)

        self.testing_imgs = self.multi_pool.map(self.crop, self.testing_imgs)

    def load_data(self, pid, folder):

        image_path = join(self.raw_data_folder, folder, '{}{}'.format(pid, self.image_postfix))
        image_data = nib.load(image_path).get_fdata()
        image_data = np.swapaxes(image_data, 0, self.dim)
        image_data, image_min, image_max = self.normalize(image_data)

        self.norm_paras[pid] = [image_min, image_max]
        if image_data.ndim == 3:
            image_data = image_data[:, :, :, np.newaxis]
        return image_data

    def __len__(self):
        return len(self.training_imgs)

    def __getitem__(self, item):
        pass

    def test_len(self):
        return len(self.testing_imgs)

    def get_test_pair(self, item):
        pass


class COVIDSRDataset(COVIDRawDataset):

    """
    Loading data generated by SR methods, and feed to the segmentation model.
    """
    def __init__(self, paras):

        data_folder = paras.data_folder
        toy_problem = paras.toy_problem
        medical_image_dim = paras.medical_image_dim_covid
        training_patient_ids = paras.training_patient_ids_covid
        validation_patient_ids = paras.validation_patient_ids_covid
        margin = paras.margin_covid
        multi_threads = paras.multi_threads
        norm = paras.normal_inputs

        super(COVIDSRDataset, self).__init__(
            data_folder=data_folder, training_patient_ids=training_patient_ids,
            validation_patient_ids=validation_patient_ids, dim=medical_image_dim,
            margin=margin, toy_problem=toy_problem, multi_threads=multi_threads,
            norm=norm
        )

        self.sr_factor = paras.sr_scale
        self.return_res_image = paras.return_res_image

        # ## prepare LR - HR pairs
        self.training_outputs = self.training_imgs
        self.training_inputs = self.multi_pool.map(
            self.resize, [[_, 1/self.sr_factor, 'cubic', 'gaussian'] for _ in self.training_imgs]
        )
        # patch size is converted to LR dimensions
        self.patch_size = paras.patch_size//int(self.sr_factor)

        self.testing_gts = self.testing_imgs
        self.testing_inputs = self.multi_pool.map(
            self.resize, [[_, 1/self.sr_factor, 'cubic', 'gaussian'] for _ in self.testing_imgs]
        )
        # ## crop function, with LR patch size
        self.random_crop = SRImagePairRandomCrop(self.patch_size, self.sr_factor, 20//int(self.sr_factor))

        # ## eva function
        quick_eva_metrics = paras.quick_eva_metrics  # 'psnr ssim' for simple case
        final_eva_metrics = paras.eva_metrics
        eva_gpu = paras.eva_gpu_id
        self.quick_eva_func = BasicSREvaluation(quick_eva_metrics, self.sr_factor, eva_gpu, 'mean')
        self.final_eva_func = BasicSREvaluation(final_eva_metrics, self.sr_factor, eva_gpu, 'full')

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
        img_id = self.testing_img_ids[item]

        if self.return_res_image:
            res_img = self.resize(
                [img_input, img_output.shape[:2]]
            )
            res_img = self.numpy_2_tensor(res_img)
        else:
            res_img = []

        img_input = self.numpy_2_tensor(img_input).unsqueeze(0)

        return {'in': img_input, 'gt': img_output, 'id': img_id, 'res': res_img}


class COVIDSRTest(COVIDSRDataset):

    def __init__(self, paras, patient_id, data_folder=None, dim=None, sr_factor=None):
        test_paras = copy.deepcopy(paras)

        test_paras.validation_patient_ids_covid = [patient_id]
        test_paras.training_patient_ids_covid = []

        if data_folder:
            test_paras.data_folder = data_folder
        if dim:
            test_paras.medical_image_dim_covid = dim
        if sr_factor:
            test_paras.sr_scale = sr_factor

        super(COVIDSRDataset, self).__init__(test_paras)

        self.patient_id = patient_id


class COVIDMetaSRDataset(COVIDRawDataset):
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
        medical_image_dim = paras.medical_image_dim_covid
        training_patient_ids = paras.training_patient_ids_covid
        validation_patient_ids = paras.validation_patient_ids_covid
        margin = paras.margin_covid
        multi_threads = paras.multi_threads
        norm = paras.normal_inputs

        super(COVIDMetaSRDataset, self).__init__(
            data_folder=data_folder, training_patient_ids=training_patient_ids,
            validation_patient_ids=validation_patient_ids, dim=medical_image_dim,
            margin=margin, toy_problem=toy_problem, multi_threads=multi_threads,
            norm=norm
        )

        self.sr_factors = paras.all_sr_scales
        self.batch_size = paras.batch_size
        self.return_res_image = paras.return_res_image

        # ## generate LR - HR pairs when get item
        self.training_outputs = self.training_imgs

        # patch size is converted to LR dimensions
        self.patch_size = paras.patch_size

        self.testing_gts = self.testing_imgs

        # ## crop function, with LR patch size
        self.random_crop = SingleImageRandomCrop(self.patch_size)

        self.test_sr_factors = paras.test_sr_scales

        # ## eva function
        quick_eva_metrics = paras.quick_eva_metrics
        final_eva_metrics = paras.eva_metrics
        eva_gpu = paras.eva_gpu_id
        self.quick_eva_func = MetaSREvaluation(quick_eva_metrics, self.test_sr_factors, eva_gpu, 'mean')
        self.final_eva_func = MetaSREvaluation(final_eva_metrics, self.test_sr_factors, eva_gpu, 'full')

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
        H, W = ori_img.shape[:2]

        img_inputs = [self.resize([ori_img, (int(H // s), int(W // s)), 'cubic', 'gaussian']) for s in self.test_sr_factors]
        img_outputs = [
            self.resize([ori_img, (int(H // s * s), int(W // s * s))]) for s in self.test_sr_factors
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


class COVIDMetaSRTest(COVIDMetaSRDataset):

    def __init__(self, paras, patient_id, test_sr_factors=None, dim=None):

        test_paras = copy.deepcopy(paras)

        test_paras.validation_patient_ids_covid = [patient_id]
        test_paras.training_patient_ids_covid = []
        test_paras.eva_metrics = ''
        test_paras.quick_eva_metrics = ''
        test_paras.normal_inputs = ''

        if test_sr_factors is not None:
            test_paras.test_sr_scales = test_sr_factors
        else:
            test_paras.test_sr_scales = paras.sr_scales_for_final_testing
        if dim is not None:
            test_paras.medical_image_dim_covid = dim

        super(COVIDMetaSRTest, self).__init__(test_paras)

        self.patient_id = patient_id




