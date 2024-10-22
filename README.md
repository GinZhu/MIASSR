# MIASSR
An Approach for Medical Image Arbitrary Scale Super-Resolution

## Introduction
In this work, we present an approach for medical image arbitrary-scale super-resolution (MIASSR), coupling meta-learning with generative adversarial networks (GANs) to super-resolve medical images at any scale of magnification in (1, 4]. Compared to SOTA SISR algorithms on single-modal magnetic resonance (MR) brain images (OASIS-brains) and multi-modal MR brain images (BraTS), MIASSR achieves comparable fidelity performance and the best perceptual quality with the smallest model size. We also employ transfer learning to enable MIASSR to tackle SR tasks of new medical modalities, such as cardiac MR images (ACDC) and chest computed tomography images (COVID-CT). The source code of our work is also public. Thus, MIASSR has the potential to become a new foundational pre-/post-processing step in clinical image analysis tasks such as reconstruction, image quality enhancement, and segmentation.

![](./figures/MIASSR_net.png)
> Framework of MIASSR.

## Results
### Broad applicability on medical images
[OASIS](https://www.oasis-brains.org/)
![](./figures/OAS1_0003_MR1_slice_60.png)
[BraTS](https://www.med.upenn.edu/cbica/brats2020/data.html)
![](./figures/BraTS_example.png)
[ACDC](https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html) and [COVID-CT](https://covid-segmentation.grand-challenge.org/Data/)
![](./figures/acdc_covid_example.png)

### Comparing with SOTA methods (PSNR + FID)
![On OASIS](./figures/NumParas.png)


## Train & Test
To setup:
```bash
git clone https://github.com/GinZhu/MIASSR.git
cd MIASSR
pip install -r requirements.txt
```
To train:
```bash
python -W ignore train.py --model-type meta_sr --config-file config_files/meta_sr_example.ini
```
To test:
```bash
python -W ignore test.py --sr-scales multi --config-file config_files/testing_meta_sr_example.ini
```
To compare with SOTA SR methods, such as [EDSR](https://arxiv.org/abs/1707.02921), we also provide:
```bash
python -W ignore train.py --model-type sota_sr --config-file config_files/sota_sr_example.ini
```
To test:
```bash
python -W ignore test.py --sr-scales multi --config-file config_files/testing_sota_sr_example.ini
```
Notice that multi pre-trained models are required for specific-scale methods.

## Pre-trained models
Here we provide pre-trained models to download (on the OASIS dataset):
- [MIASSR]()

## Publications & citations
This work is available at [IJNS](https://www.worldscientific.com/doi/10.1142/S0129065721500374), please cite as:
```
@article{zhu2021miassr,
  title={Arbitrary scale super-resolution for medical images},
  author={Zhu, Jin and Tan, Chuan and Yang, Junwei and Yang, Guang and Lio’, Pietro},
  journal={International Journal of Neural Systems},
  volume={31},
  number={10},
  pages={2150037},
  year={2021},
  publisher={World Scientific}
}
```
We refer to the previous works for better understanding of this project:
- Lesion focused multi-scale GAN: [paper1](https://arxiv.org/abs/1810.06693), [paper2](https://arxiv.org/abs/1901.03419), [code](https://github.com/GinZhu/msgan)