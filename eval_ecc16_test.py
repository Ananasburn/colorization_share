
import matplotlib.pyplot as plt
import os
from colorizers import *


img_root_dir = 'test_images'
colorimg_dir = './colorimg/'
os.makedirs(colorimg_dir, exist_ok=True)

# load colorizers
colorizer_eccv16 = eccv16_test(pretrained=True).eval()

colorizer_eccv16.cuda()
	
predictions = []

img_list = load_img(img_root_dir)   
for idx, img_np in enumerate(img_list):
    (tens_l_orig, tens_l_rs) = preprocess_img(img_np, HW=(256,256))

    tens_l_rs = tens_l_rs.cuda()

# colorizer outputs 256x256 ab map
# resize and concatenate to original L channel
    img_bw = postprocess_tens(tens_l_orig, torch.cat((0*tens_l_orig,0*tens_l_orig),dim=1))
    out_img_eccv16 = postprocess_tens(tens_l_orig, colorizer_eccv16(tens_l_rs).cpu())
    
    out_img_rs = float2unit8_resize(out_img_eccv16,HW=(224,224))
    predictions.append(out_img_rs)
    
    colorized_name =  os.path.join(colorimg_dir, f'{idx}.jpg')
    plt.imsave(colorized_name, out_img_rs)
predictions = np.array(predictions)
np.save('prediction.npy', predictions)
print(f'Saved predictions to prediction.npy with shape {predictions.shape}')  


