# Zero Shot Image Restoration Using Denoising Diffusion Null-Space Model
## 📖[**Paper**](https://arxiv.org/pdf/2212.00490.pdf)|🖼️[**Project Page**](https://wyhuai.github.io/ddnm.io/)| <a href="https://colab.research.google.com/drive/1SRSD6GXGqU0eO2CoTNY-2WykB9qRZHJv?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="google colab logo"></a>

[Yinhuai Wang*](https://wyhuai.github.io/info/), [Jiwen Yu*](https://scholar.google.com.hk/citations?user=uoRPLHIAAAAJ), [Jian Zhang](https://jianzhang.tech/)  
Peking University and PCL  
\*denotes equal contribution



This repository contains the code release for *Zero Shot Image Restoration Using ***D***enoising ***D***iffusion ***N***ull-Space ***M***odel*. **DDNM** can solve various image restoration tasks **without any optimization or training! Yes, in a zero-shot manner**.


***Supported Applications:***
- **Arbitrary Size**🆕
- **Old Photo Restoration**🆕
- Super-Resolution
- Denoising
- Colorization
- Inpainting
- Deblurring
- Compressed Sensing


![front](https://user-images.githubusercontent.com/95485229/206398186-bc7128d1-9fa9-40c8-8b51-36869b17b1bd.png) 

## 🧩News
- A Colab demo for high-quality results is now avaliable! <a href="https://colab.research.google.com/drive/1SRSD6GXGqU0eO2CoTNY-2WykB9qRZHJv?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="google colab logo"></a>

## Installation
### Code
```
git clone https://github.com/wyhuai/DDNM.git
```
### Environment
```
pip install numpy torch blobfile tqdm pyYaml pillow    # e.g. torch 1.7.1+cu110.
```
### Pre-Trained Models
To restore human face images, download this [model](https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/celeba_hq.ckpt)(from [SDEdit](https://github.com/ermongroup/SDEdit)) and put it into `DDNM/exp/logs/celeba/`. 
```
wget https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/celeba_hq.ckpt
```
To restore general images, download this [model](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt)(from [guided-diffusion](https://github.com/openai/guided-diffusion)) and put it into `DDNM/exp/logs/imagenet/`.
```
wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt
```
### Quick Start
Run below command to get 4x SR results immediately. The results should be in `DDNM/exp/image_samples/demo`.
```
python main.py --ni --simplified --config celeba_hq.yml --path_y celeba_hq --eta 0.85 --deg "sr_averagepooling" --deg_scale 4.0 --sigma_y 0 -i demo
```

## Setting
The detailed sampling command is here:
```
python main.py --ni --simplified --config {CONFIG}.yml --path_y {PATH_Y} --eta {ETA} --deg {DEGRADATION} --deg_scale {DEGRADATION_SCALE} --sigma_y {SIGMA_Y} -i {IMAGE_FOLDER}
```
with following options:
- We implement **TWO** versions of DDNM in this repository. One is SVD-based version, which is more precise in solving noisy tasks. Another one is the simplified version, which does not involve SVD and is flexible for users to define their own degradations. Use `--simplified` to activate the simplified DDNM. Without `--simplified` will turn to the SVD-based DDNM.
- `PATH_Y` is the folder name of the test dataset, in `DDNM/exp/datasets`.
- `ETA` is the DDIM hyperparameter. (default: `0.85`)
- `DEGREDATION` is the supported tasks including `cs_walshhadamard`, `cs_blockbased`, `inpainting`, `denoising`, `deblur_uni`, `deblur_gauss`, `deblur_aniso`, `sr_averagepooling`,`sr_bicubic`, `colorization`, `mask_color_sr`, and user-defined `diy`.
- `DEGRADATION_SCALE` is the scale of degredation. e.g., `--deg sr_bicubic --deg_scale 4` lead to 4xSR.
- `SIGMA_Y` is the noise observed in y.
- `CONFIG` is the name of the config file (see `configs/` for a list), including hyperparameters such as batch size and sampling step.
- `IMAGE_FOLDER` is the folder name of the results.

For the config files, e.g., celeba_hq.yml, you may change following properties:
```
sampling:
    batch_size: 1
    
time_travel:
    T_sampling: 100     # sampling steps
    travel_length: 1    # time-travel parameters l and s, see section 3.3 of the paper.
    travel_repeat: 1    # time-travel parameter r, see section 3.3 of the paper.
```

## Reproduce The Results In The Paper
### Quantitative Evaluation
Dataset download link: [[Google drive](https://drive.google.com/drive/folders/1cSCTaBtnL7OIKXT4SVME88Vtk4uDd_u4?usp=sharing)] [[Baidu drive](https://pan.baidu.com/s/1tQaWBqIhE671v3rrB-Z2mQ?pwd=twq0)]

Download the CelebA testset and put it into `DDNM/exp/datasets/celeba/`.

Download the ImageNet testset and put it into `DDNM/exp/datasets/imagenet/` and replace the file `DDNM/exp/imagenet_val_1k.txt`.

Run the following command. You may increase the batch_size to accelerate evaluation.
```
sh evaluation.sh
```

### High-Quality Results
You can try this [**Colab demo**](https://colab.research.google.com/drive/1SRSD6GXGqU0eO2CoTNY-2WykB9qRZHJv?usp=sharing) for High-Quality results. Note that the High-Quality results presented in the front figure are mostly generated by applying DDNM to the models in [RePaint](https://github.com/andreas128/RePaint).

## 🔥Real-World Applications
### Demo: Real-World Super-Resolution.
![orig_62](https://user-images.githubusercontent.com/95485229/204471148-bf155c60-c7b3-4c3a-898c-859cb9d0d723.png)
![00000](https://user-images.githubusercontent.com/95485229/204971948-7564b536-b562-4187-9d8c-d96db4c55f7c.png)

Run the following command
```
python main.py --ni --simplified --config celeba_hq.yml --path_y solvay --eta 0.85 --deg "sr_averagepooling" --deg_scale 4.0 --sigma_y 0.1 -i demo
```
### Demo: Old Photo Restoration.
![image](https://user-images.githubusercontent.com/95485229/204973149-4818426b-89af-410c-b1b7-f26b8f65358b.png)
![image](https://user-images.githubusercontent.com/95485229/204973288-0f245e93-8980-4a32-a5e9-7f2bfe58d8eb.png)

Run the following command
```
python main.py --ni --simplified --config oldphoto.yml --path_y oldphoto --eta 0.85 --deg "mask_color_sr" --deg_scale 2.0 --sigma_y 0.02 -i demo
```
### Try your own photos.
You may use DDNM to restore your own degraded images. DDNM provides full flexibility for you to define the degradation operator and the noise level. Note that these definitions are critical for a good results. You may reference the following guidance.
1. If your are using CelebA pretrained models, try this [tool](???) to crop and align your photo.
2. If there are local artifacts on your photo, try this [tool](???) to draw a mask to cover them, and save this mask to `DDNM/exp/inp_masks/mask.png`. Then run `DDNM/exp/inp_masks/get_mask.py` to generate `mask.npy`.
3. If your photo is faded, you need a grayscale operator as part of the degradation.
4. If your photo is blur, you need a downsampler operator as part of the degradation. Also, you need to set a proper SR scale `--deg_scale`.
5. If your photo suffers global artifacts, e.g., jpeg-like artifacts or unkown noise, you need to set a proper `sigma_y` to remove these artifacts.
6. Search `args.deg =='diy'` in `DDNM/guided_diffusion/diffusion.py` and change the definition of $\mathbf{A}$ correspondingly.
Then run
```
python main.py --ni --simplified --config celeba_hq.yml --path_y {YOUR_OWN_PATH} --eta 0.85 --deg "diy" --deg_scale {YOUR_OWN_SCALE} --sigma_y {YOUR_OWN_LEVEL} -i diy
```

## 🆕DDNM for Arbitrary Size
![123456](https://user-images.githubusercontent.com/95485229/206181069-0f134804-63e3-4ba1-9ad1-0a4ccd9dd72e.png)![image](https://user-images.githubusercontent.com/95485229/206333186-c240fad9-3602-46f5-8bc7-a66e8c463196.png)

Above we show an example of using DDNM to SR a 64x256 input image into a 256x1024 result. We call this newly conceived method the **Mask-Shift trick**, whose details can be found in the updated version of our [paper](https://arxiv.org/pdf/2212.00490.pdf), section 7.

We implement the **Mask-Shift trick** in the folder `hq_demo`, based on [RePaint](https://github.com/andreas128/RePaint). You can try this [**Colab demo**](https://colab.research.google.com/drive/1SRSD6GXGqU0eO2CoTNY-2WykB9qRZHJv?usp=sharing). <a href="https://colab.research.google.com/drive/1SRSD6GXGqU0eO2CoTNY-2WykB9qRZHJv?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="google colab logo"></a>

To try this function in your own device, you need to download the pre-trained models:
```
wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_classifier.pt
wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion.pt
```
and put it to `hq_demo/data/pretrained`. Then run
```
cd hq_demo
sh evaluation.sh
```
This script contains SR results up to **2K** resolution. It may take hours to finish some demos in this script. Setting a smaller sampling step or time-travel parameters in hq_demo/confs/inet256.yml can speed up, but may compromise the generative quality.



## 😊Applying DDNM to Your Own Diffusion Model
It is ***very easy*** to implement a basic DDNM on your own diffusion model! You may reference the following:
1. Copy these operator implementations to the core diffusion sampling file, then define your task type, e.g., set `IR_mode="super resolution"`.
```python
def color2gray(x):
    coef=1/3
    x = x[:,0,:,:] * coef + x[:,1,:,:]*coef +  x[:,2,:,:]*coef
    return x.repeat(1,3,1,1)

def gray2color(x):
    x = x[:,0,:,:]
    coef=1/3
    base = coef**2 + coef**2 + coef**2
    return th.stack((x*coef/base, x*coef/base, x*coef/base), 1)    
    
def PatchUpsample(x, scale):
    n, c, h, w = x.shape
    x = torch.zeros(n,c,h,scale,w,scale) + x.view(n,c,h,1,w,1)
    return x.view(n,c,scale*h,scale*w)

# Implementation of A and its pseudo-inverse Ap    
    
if IR_mode=="colorization":
    A = color2gray
    Ap = gray2color
    
elif IR_mode=="inpainting":
    A = lambda z: z*mask
    Ap = A
      
elif IR_mode=="super resolution":
    A = torch.nn.AdaptiveAvgPool2d((256//scale,256//scale))
    Ap = lambda z: PatchUpsample(z, scale)

elif IR_mode=="old photo restoration":
    A1 = lambda z: z*mask
    A1p = A1
    
    A2 = color2gray
    A2p = gray2color
    
    A3 = torch.nn.AdaptiveAvgPool2d((256//scale,256//scale))
    A3p = lambda z: PatchUpsample(z, scale)
    
    A = lambda z: A3(A2(A1(z)))
    Ap = lambda z: A1p(A2p(A3p(z)))
```
2. Find the variant $\mathbf{x}\_{0|t}$ in the target codes, and use the result of this function to modify the sampling of $\mathbf{x}\_{t-1}$. Your may need to provide the input degraded image $\mathbf{y}$ and the corresponding noise level $\sigma_\mathbf{y}$.
```python
# Core Implementation of DDNM+, simplified denoising solution (Section 3.3).
# For more accurate denoising, please refer to the paper (Appendix I) and the source code.

def ddnm_plus_core(x0t, y, sigma_y=0, sigma_t, a_t):

    #Eq 19
    if sigma_t >= a_t*sigma_y: 
        lambda_t = 1
        gamma_t = sigma_t**2 - (a_t*lambda_t*sigma_y)**2
    else:
        lambda_t = sigma_t/(a_t*sigma_y)
        gamma_t = 0
        
    #Eq 17    
    x0t= x0t + lambda_t*Ap(y - A(x0t))
    
    return x0t, gamma_t
```
3. Actually, this repository contains the above simplified implementation: try search `arg.simplified` in `DDNM/guided_diffusion/diffusion.py` for related codes. 

# References
If you find this repository useful for your research, please cite the following work.
```
@article{wang2022ddnm,
title={Zero Shot Image Restoration Using Denoising Diffusion Null-Space Model},
author={Yinhuai, Wang and Jiwen, Yu and Jian, Zhang},
journal={arXiv:2212.00490},
year={2022}}
}
```
This implementation is based on / inspired by:
- https://github.com/wyhuai/RND (null-space learning)
- https://github.com/andreas128/RePaint (time-travel trick)
- https://github.com/bahjat-kawar/ddrm (code structure)
