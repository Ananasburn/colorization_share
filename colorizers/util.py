
from PIL import Image
import numpy as np
from skimage import color
import torch
import torch.nn.functional as F
from IPython import embed
import os

def load_img(img_root_dir):
    image_filenames = [f for f in os.listdir(img_root_dir) if os.path.isfile(os.path.join(img_root_dir, f))]
    # 按照文件名排序以保证索引的一致性
    image_filenames.sort()
    
    images = []
    for idx, filename in enumerate(image_filenames):
        img_name = os.path.join(img_root_dir, filename)
        out_np = np.asarray(Image.open(img_name))
        if out_np.ndim == 2:
            out_np = np.tile(out_np[:, :, None], 3)
        images.append(out_np)
    
    return images

def resize_img(img, HW=(256,256), resample=3):
	return np.asarray(Image.fromarray(img).resize((HW[1],HW[0]), resample=resample))  #?

def preprocess_img(img_rgb_orig, HW=(256,256), resample=3):
	# return original size L and resized L as torch Tensors
	img_rgb_rs = resize_img(img_rgb_orig, HW=HW, resample=resample)
	
	img_lab_orig = color.rgb2lab(img_rgb_orig)
	img_lab_rs = color.rgb2lab(img_rgb_rs)

	img_l_orig = img_lab_orig[:,:,0]  #从转换后的 LAB 图像中提取亮度通道（L），即取出第一通道（[:,:,0]）
	img_l_rs = img_lab_rs[:,:,0]

	tens_orig_l = torch.Tensor(img_l_orig)[None,None,:,:]    #增加两个维度，使得张量的形状变为 (1, 1, H, W)，这是典型的神经网络输入格式，其中第一个维度是批量大小，第二个维度是通道数。
	tens_rs_l = torch.Tensor(img_l_rs)[None,None,:,:]

	return (tens_orig_l, tens_rs_l)

def postprocess_tens(tens_orig_l, out_ab, mode='bilinear'):
	# tens_orig_l 	1 x 1 x H_orig x W_orig
	# out_ab 		1 x 2 x H x W

	HW_orig = tens_orig_l.shape[2:]
	HW = out_ab.shape[2:]

	# call resize function if needed
	if(HW_orig[0]!=HW[0] or HW_orig[1]!=HW[1]):
		out_ab_orig = F.interpolate(out_ab, size=HW_orig, mode='bilinear')
	else:
		out_ab_orig = out_ab

	out_lab_orig = torch.cat((tens_orig_l, out_ab_orig), dim=1)
	return color.lab2rgb(out_lab_orig.data.cpu().numpy()[0,...].transpose((1,2,0)))  #将 NumPy 数组的维度从 [C, H, W] 转换为 [H, W, C]，使其符合图像处理库的标准输入格式。


def train_preprocess_img(train_img_orig, HW=(256,256), resample=3):
	train_img_rs = resize_img(train_img_orig, HW=HW, resample=resample)  #?
	train_gray_rs = np.asarray(color.rgb2gray(train_img_rs))   # H*W
	train_lab_rs = np.asarray(color.rgb2lab(train_img_rs))
	train_ab_rs = train_lab_rs[:,:,1:3]   # H*W*C（3 channel:LAB，其中取第1、2个channel，即ab）
	tens_graytrain_rs = torch.Tensor(train_gray_rs)[None,:,:]   #在H, W前面增加2个维度  [None,None,:,:] 
	tens_abtrain_rs = torch.Tensor(train_ab_rs.transpose((2, 0, 1)))#[None,:,:,:]   # Convert to tensor format C*H*W，在C, H, W前面增加一个维度
	return (tens_graytrain_rs, tens_abtrain_rs)


def float2unit8_resize(img, HW=(256, 256), resample=Image.BILINEAR):
    if not isinstance(img, np.ndarray):
        img = np.asarray(img)
    img = (img * 255).astype(np.uint8)  # Ensure the image is in uint8 format
    return np.asarray(Image.fromarray(img).resize((HW[1], HW[0]), resample=resample))
