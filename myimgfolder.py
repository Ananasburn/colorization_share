import os
from torchvision.transforms import transforms, ToTensor
from torch.utils.data import Dataset
from skimage.color import rgb2lab, rgb2gray
import torch
import numpy as np
from PIL import Image
from colorizers import *
#import matplotlib.pyplot as plt



class TrainImageFolder(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_filenames = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]

    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_filenames[idx])
        image = np.asarray(Image.open(img_name).convert("RGB"))

        (tens_graytrain_rs, tens_abtrain_rs) = train_preprocess_img(image, HW=(256,256))    #?

        
        return (tens_graytrain_rs, tens_abtrain_rs)




    # def __getitem__(self, index):
    #     path, target = self.imgs[index]
    #     img = self.loader(path)
    #     ## 对图像进行预处理
    #     if self.transform is not None:
    #         img_original = self.transform(img)
    #         img_original = np.asarray(img_original)

    #         # 转换为 Lab 色彩空间
    #         img_lab = rgb2lab(img_original)
    #         img_lab = (img_lab + 128) / 255   # 归一化到 [-1, 1]
    #         img_ab = img_lab[:, :, 1:3]    # 提取色度信息
    #         img_ab = torch.from_numpy(img_ab.transpose((2, 0, 1)))    # 转为 PyTorch 张量
    #         img_original_gray = rgb2gray(img_original)
    #         img_original_gray = torch.from_numpy(img_original).unsqueeze(0)  # 添加一个维度，变为 (1, H, W)
    #     if self.target_transform is not None:
    #         target = self.target_transform(target)

    #          # 返回处理后的图像和标签
    #     return (img_original_gray, img_ab), target


class GrayImageFolder(Dataset):

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_filenames = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_filenames[idx])
        image = Image.open(img_name).convert('RGB')

        image_scale = scale_transform(image)
        image_scale = rgb2gray(np.array(image_scale))
        image_scale = torch.from_numpy(image_scale).float()
        if self.transform:
            image = self.transform(image)

        # # Convert image to tensor if not already
        # if not isinstance(image, torch.Tensor):
        #     image = ToTensor()(image)
        Test_img_gray = rgb2gray(np.array(image))   #rgb2gray(np.array(image))
        Test_img_gray = torch.from_numpy(Test_img_gray).float()

        return Test_img_gray, image_scale


    # def __getitem__(self, index):
    #     path, target = self.imgs[index]
    #     img = self.loader(path)

    #     img_scale = img.copy()
    #     img_original = img
    #     img_scale = scale_transform(img_scale)

    #     img_scale = np.asarray(img_scale)
    #     img_original = np.asarray(img_original)

    #     img_scale = rgb2gray(img_scale)
    #     img_scale = torch.from_numpy(img_scale)
    #     img_original = rgb2gray(img_original)
    #     img_original = torch.from_numpy(img_original)
    #     return (img_original, img_scale), target
