from utils.param_loader import ParametersLoader
from datasets.OASIS_dataset import OASISMetaSRDataset, OASISSRDataset
from datasets.BraTS_dataset import BraTSSRDataset, BraTSMetaSRDataset
from datasets.DIV2K_dataset import RGBMetaSRDataset, RGBSRDataset
from datasets.ACDC_dataset import ACDCMetaSRDataset, ACDCSRDataset
from datasets.COVID_dataset import COVIDMetaSRDataset, COVIDSRDataset
from models.meta_sr_trainer import MetaSRTrainer
from models.sota_sr_trainer import SRTrainer
import argparse

"""
Example:
    python -W ignore train.py --model-type sota_sr --config-file config_files/sota_sr_example.ini
"""


parser = argparse.ArgumentParser(description='Training Parameters')
parser.add_argument('--config-file', type=str, required=True, metavar='CONFIG',
                    help='Path to config file.')
parser.add_argument('--gpu-id', type=int, metavar='GPU',
                    help='Which gpu to use.')
parser.add_argument('--generator', type=str, metavar='G', default='',
                    choices=['', 'SRResNet', 'EDSR', 'SRDenseNet', 'RDN', 'ESRGAN', 'MDSR'],
                    help='Optional, to specify which generator to use.')
parser.add_argument('--model-type', type=str, required=True, choices=['meta_sr', 'sota_sr'], metavar='SR Model',
                    help='meta_sr or sota_sr')

args = parser.parse_args()
# do distributed training here
config_file = args.config_file
gpu_id = args.gpu_id
model_type = args.model_type
generator = args.generator

paras = ParametersLoader(config_file)

if gpu_id is not None:
    paras.gpu_id = gpu_id
    paras.eva_gpu_id = gpu_id

if generator is not '':
    paras.feature_generator = generator
    paras.sr_generator = generator

data_folder = paras.data_folder

if 'OASIS' in data_folder:
    if model_type == 'meta_sr':
        ds = OASISMetaSRDataset(paras)
    elif model_type == 'sota_sr':
        ds = OASISSRDataset(paras)
elif 'BraTS' in data_folder:
    if model_type == 'meta_sr':
        ds = BraTSMetaSRDataset(paras)
    elif model_type == 'sota_sr':
        ds = BraTSSRDataset(paras)
elif 'ACDC' in data_folder:
    if model_type == 'meta_sr':
        ds = ACDCMetaSRDataset(paras)
    elif model_type == 'sota_sr':
        ds = ACDCSRDataset(paras)
elif 'COVID' in data_folder:
    if model_type == 'meta_sr':
        ds = COVIDMetaSRDataset(paras)
    elif model_type == 'sota_sr':
        ds = COVIDSRDataset(paras)
elif 'DIV2K' in data_folder:
    if model_type == 'meta_sr':
        ds = RGBMetaSRDataset(paras)
    elif model_type == 'sota_sr':
        ds = RGBSRDataset(paras)
else:
    raise ValueError('Only support data: [OASIS, DIV2K, BraTS, ACDC, COVID]')

print('DS info:', len(ds), 'training samples, and', ds.test_len(), 'testing cases.')

# ## training
if model_type == 'meta_sr':
    trainer = MetaSRTrainer(paras, ds, ds)
elif model_type == 'sota_sr':
    trainer = SRTrainer(paras, ds, ds)
trainer.setup()
trainer.train()

# # ## testing / inference
# for pid in paras.testing_patient_ids:
#     ds_test = OASISSRTest(paras.data_folder, pid, paras.dim, paras.sr_factor)
#     trainer.inference(ds_test, False)

