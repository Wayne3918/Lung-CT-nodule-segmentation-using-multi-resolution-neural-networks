import torch
import SimpleITK as sitk
import os
from tqdm import tqdm
import numpy as np
from models import Semi_resMCNN
import argparse

def predict_one_img(model, sliced_img):
    
    all_pred = np.zeros((len(sliced_img)+32-1,256,256))
    model.eval()
    with torch.no_grad():
        for index in tqdm(range(len(sliced_img))):
            img_slice = sliced_img[index]
            img_slice = np.expand_dims(img_slice, axis = 0)
            img_slice = np.expand_dims(img_slice, axis = 0)
            img_slice = torch.from_numpy(img_slice)
            img_slice=img_slice.type(torch.FloatTensor)
            #print(img.shape)
            
            output = model(img_slice)
            output = output.cpu().detach().numpy()
            all_pred[index:index+32,:,:] = all_pred[index:index+32,:,:] + output[0,1,:,:,:]
    all_pred = all_pred / 32
    th = 0.5
    all_pred = np.int32(all_pred>th)
    print(all_pred.shape)
    all_pred = sitk.GetImageFromArray(all_pred)
    return all_pred

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Visualise predicted images')
    parser.add_argument('--img_path',default = './images/LIDC-IDRI-0838.nii',help='path to your CT image')
    parser.add_argument('--checkpoint_path',default = './checkpoints/best_model.pth',help='path to your checkpoint file')
    parser.add_argument('--save_path',default = './results',help='result save folder')

    args = parser.parse_args()
    
    img_path = args.img_path
    img_ct = sitk.ReadImage(img_path,sitk.sitkInt16)
    img_np = sitk.GetArrayFromImage(img_ct) / 1000
    sliced_img = []

    img_s, img_h, img_w = img_np.shape
    number_of_slices = img_s - 32 + 1

    for i in range(number_of_slices):
        img_slice = img_np[i:i+32,:,:]
        sliced_img.append(img_slice)

    save_path = args.save_path
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    device = torch.device('cuda')
    # model info
    model = Semi_resMCNN(in_channel=1, out_channel=2,training=False).to(device)
    model = torch.nn.DataParallel(model, device_ids=[0])  # multi-GPU
    pth_path = args.checkpoint_path
    ckpt = torch.load(pth_path)
    model.load_state_dict(ckpt['net'])

    pred_img = predict_one_img(model, sliced_img)
    sitk.WriteImage(pred_img, os.path.join(save_path, 'result_' + str(args.img_path.split('-')[-1][0:4]) + '.nii'))
    print('Done')
