import os
import traceback
from PIL import Image
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from colorizers import *
from myimgfolder import TrainImageFolder
import tensorflow as tf

# 关闭oneDNN优化
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Check if model parameters are updating
def check_params(model):
    for name, param in model.named_parameters():
        print(f'{name} - mean: {param.data.mean()}')

original_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

#have_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_dir = 'train/images'
# 创建自定义数据集实例
train_set = TrainImageFolder(data_dir, transform = original_transform)

# 数据加载器
train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4)
color_model = eccv16_test(pretrained=False)


# 如果存在预训练参数，则加载
# if os.path.exists('./colornet_params.pth'):
#     color_model.load_state_dict(torch.load('colornet_params.pth'))

color_model.load_state_dict(torch.load('colornet_params.pth'))
    

color_model.to(device)

     
optimizer = optim.Adam(color_model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1) 
criterion = nn.MSELoss() 


def train(epoch):
    color_model.train()

    try:
        for batch_idx, (tens_graytrain_rs, tens_abtrain_rs) in enumerate(train_loader):  #?  # img_original_gray，  original_img
            messagefile = open('./message.txt', 'a')
            # 检查输入数据
            print(f"Input shape: {tens_graytrain_rs.shape}, Input mean: {tens_graytrain_rs.mean()}, Input std: {tens_graytrain_rs.std()}")
            print(f"Label shape: {tens_abtrain_rs.shape}, Label mean: {tens_abtrain_rs.mean()}, Label std: {tens_abtrain_rs.std()}")
           
        

            tens_graytrain_rs = tens_graytrain_rs.to(device)
            tens_abtrain_rs = tens_abtrain_rs.to(device)
            
            optimizer.zero_grad()
            output = color_model(tens_graytrain_rs) #??
            # 打印模型输出
            print(f"Model output: {output}")

             # 计算损失
            loss = criterion(output, tens_abtrain_rs)
            # 打印损失值
            print(f'Loss: {loss.data.item()}')

            # 反向传播和优化
            loss.backward()
            # 打印梯度信息
            for name, param in color_model.named_parameters():
                if param.grad is not None:
                    print(f"Gradients of {name}: {param.grad.abs().mean().item()}")

            optimizer.step()
            for param_group in optimizer.param_groups:
                for param in param_group['params']:
                    print(param)
            scheduler.step()
            
            # Check model parameters
            check_params(color_model)

             # 记录损失
            with open('./message.txt', 'a') as messagefile:
                messagefile.write(f'loss: {loss.data.item():.9f}\n')

             # 打印损失
            print(f'Epoch: {epoch}, Batch: {batch_idx}, loss: {loss.data.item():.9f}')

            
             # 停止早期的迭代以调试!!!非调试记得注释此代码
            if batch_idx == 10:
                break
            

            

    except Exception:
        with open('log.txt', 'w') as logfile:
            logfile.write(traceback.format_exc())
            
    finally:
        torch.save(color_model.state_dict(), 'colornet_weights.pth')

if __name__ == '__main__':
    epochs = 10
    for epoch in range(1, epochs + 1):
        train(epoch)
    