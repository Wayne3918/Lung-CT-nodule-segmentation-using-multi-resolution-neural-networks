import nibabel as nib
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use("Agg")
import numpy as np
import tqdm
import os
import cv2
import argparse

def plot_every_slice(args):
    img_filename = args.img_path
    mask_filename = args.mask_path
    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    mask = nib.load(mask_filename)
    img = nib.load(img_filename)
    mask = np.array(mask.dataobj)
    img = np.array(img.dataobj)
    mask_raw = mask
    img  = (img - np.min(img))/ (np.max(img) - np.min(img)) * 220
    img = img.astype(np.uint8)
    print(mask.shape)
    for i in tqdm.trange(len(img[0][0])):
        #i += 184
        plt.figure()

        mask = mask_raw[:,:,i]
        mask = mask.astype(np.uint8)
        original_image = cv2.cvtColor(img[:,:,i], cv2.COLOR_GRAY2BGR)
        original_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        original_mask = original_mask.astype(np.uint8)
        #print(original_mask.shape)
        original_mask = cv2.cvtColor(original_mask, cv2.COLOR_BGR2GRAY)

        prediction_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        prediction_mask = cv2.cvtColor(prediction_mask, cv2.COLOR_BGR2GRAY)
        contours, hierarchy = cv2.findContours(original_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        target_img = original_image.copy()
        cv2.drawContours(target_img, contours, -1, (255,0,0), 1)

        plt.imshow(target_img, interpolation='none')
        img_path = os.path.join(save_path, str(i) + '.png')
        plt.savefig(img_path)
        plt.close('all')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualise predicted images')
    parser.add_argument('--img_path',default = './images/LIDC-IDRI-0838.nii',help='path to your CT image')
    parser.add_argument('--mask_path',default = './results/result_0838.nii',help='path to your prediction mask')
    parser.add_argument('--save_path',default = './results_visualisation/LIDC-IDRI-0838',help='image save folder')

    args = parser.parse_args()
    plot_every_slice(args)
    