# Lung-CT-nodule-segmentation-using-multi-resolution-neural-networks

This repository contains a testing script of my proposed semi-residual MCNN that perform lung nodule segmentation on the LIDC-IDRI dataset. All implementations in this repository are in accordance with my final year project report. 

## Prerequisites

A minimum of 4GB GPU memory is required to run testing scripts.
This repository also requires the following Python packages to run:

* Pytorch 1.10.0+ (Previous version of Pytorch may work but not tested)
* SimpleITK
* OpenCV-Python
* Nibabel
* Numpy
* Tqdm
* Matplotlib

## To run

1. The following command would unzip pre-processed LIDC-IDRI test images in the folder ./images:

```bash
$ unzip ./images
```

2. Predict the nodule segmentation map of Lung CT images using the following command:

```bash
$ python predict_one_img.py --img_path './images/LIDC-IDRI-0838.nii' --checkpoint_path './checkpoints/best_model.pth' --save_path './results'
```

3. Visualise predicted images using the following command:

```bash
$ python nii_plotter.py --img_path './images/LIDC-IDRI-0838.nii' --mask_path './results/result_0838.nii' --save_path './results_visualisation/LIDC-IDRI-0838'
```
