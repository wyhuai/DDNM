# DDNM
## 🌟Brief
This repository contains the code release for *Zero Shot Image Restoration Using ***D***enoising ***D***iffusion ***N***ull-Space ***M***odel*.

DDNM can solve various image restoration tasks **without any optimization or finetuning! Yes, in a zero-shot manner**.

📖[Paper](https://openreview.net/forum?id=mRieQgMtNTQ)

🖼️[Project](https://openreview.net/forum?id=mRieQgMtNTQ)

***Supported Applications:***
- **Old Photo Restoration**🆕
- Super-Resolution
- Colorization
- Inpainting
- Deblurring
- Compressed Sensing

![image](https://user-images.githubusercontent.com/95485229/198285474-ff2e43de-9fc5-40c4-840b-f902bac4fa3c.png)

## 🌟Installation
### Code
```
git clone https://github.com/wyhuai/DDNM.git
```
### Environment
```
pip install numpy torch blobfile tqdm pyYaml pillow    # e.g. torch 1.7.1+cu110.
```
### Pre-Trained Models
For human face, download this [model](https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/celeba_hq.ckpt)(from [SDEdit](https://github.com/ermongroup/SDEdit)) and put it into "DDNM/exp/logs/celeba/". 
```
wget https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/celeba_hq.ckpt
```

For general images, download this [model](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt)(from [guided-diffusion](https://github.com/openai/guided-diffusion)) and put it into "DDNM/exp/logs/imagenet/".
```
wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt
```
### Quick Start
Run below command to get 4x SR results immediately. The results should be in `DDNM/exp/image_samples/demo`.
```
python main.py --ni --simplified --config celeba_hq_bs1.yml --path_y celeba_hq --timesteps 100 --eta 0.85 --deg "sr_averagepooling" --deg_scale 4.0 --sigma_y 0 -i demo
```

## 🌟Setting.
The detailed sampling command is here:
```
python main.py --ni --simplified --config {CONFIG}.yml --path_y {PATH_Y} --eta {ETA} --deg {DEGRADATION} --deg_scale {DEGRADATION_SCALE} --sigma_y {SIGMA_Y} -i {IMAGE_FOLDER}
```
with following options:
- Adding `--simplified` leads to a simplified implementation of DDNM that **do not** use SVD. Without `--simplified` leads to a SVD-based DDNM implementation.
- `PATH_Y` is the folder name of the test dataset, in `DDNM/exp/datasets`.
- `ETA` is the DDIM hyperparameter. (default: `0.85`)
- `DEGREDATION` is the type of degredation allowed. (One of: `cs_walshhadamard`, `cs_blockbased`, `inpainting`, `denoising`, `deblur_uni`, `deblur_gauss`, `deblur_aniso`, `sr_averagepooling`,`sr_bicubic`, `colorization`, `mask_color_sr`, `diy`)
- `DEGRADATION_SCALE` is the scale of degredation. e.g., `--deg sr_bicubic --deg_scale 4` lead to 4xSR.
- `SIGMA_Y` is the noise observed in y.
- `CONFIG` is the name of the config file (see `configs/` for a list), including hyperparameters such as batch size and network architectures.
- `DATASET` is the name of the dataset used, to determine where the checkpoint file is found.
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

## 🌟Reproduce The Quantitative Tesults In The Paper.
Download this CelebA testset and put it into `DDNM/exp/datasets/celeba/`.

Download this ImageNet testset and put it into `DDNM/exp/datasets/imagenet/`.

Run the following command
```
sh evaluation.sh
```

## 🔥Real-World Applications.
### Real-World Super-Resolution.
![orig_62](https://user-images.githubusercontent.com/95485229/204471148-bf155c60-c7b3-4c3a-898c-859cb9d0d723.png)
![noise](https://user-images.githubusercontent.com/95485229/204470898-cd729024-c2de-4088-b35d-6b31b8863dae.gif)



Run the following command
```
python main.py --ni --simplified --config celeba_hq_bs1.yml --path_y solvay --timesteps 100 --eta 0.85 --deg "sr_averagepooling" --deg_scale 4.0 --sigma_y 0.1 -i demo
```
### Old Photo Restoration.
![image](https://user-images.githubusercontent.com/95485229/204471696-e27e14f1-c903-4405-a002-2d07a9cf557f.png)
![oldnoise](https://user-images.githubusercontent.com/95485229/204470916-109a068d-5623-460b-be33-5b6b304e52d8.gif)

Run the following command
```
python main.py --ni --simplified --config celeba_hq_bs1.yml --path_y web_photo --timesteps 100 --eta 0.85 --deg "mask_color_sr" --deg_scale 4.0 --sigma_y 0.1 -i demo
```
### DIY.
You may use DDNM to handle self-defined real-world IR tasks.
1. If your are using CelebA pretrained models, try this [tool](???) to crop and align your photo.
2. If there are local artifacts on your photo, try this [tool](???) to draw a mask to cover them, and save this mask to `DDNM/exp/inp_masks/mask.png`. Then run `DDNM/exp/inp_masks/get_mask.py` to generate `mask.npy`. Correspondingly, you need a mask operator as a component of $\mathbf{A}$.
3. If your photo is faded, you need a grayscale operator as a component of $\mathbf{A}$.
4. If your photo is blur, you need a downsampler operator as a component of $\mathbf{A}$ and need to set a proper SR scale `--deg_scale`.
5. If your photo suffers global artifacts, e.g., jpeg-like artifacts or unkown noise, you need to set a proper `sigma_y`. Tips: You can start with a big one, e.g., `--sigma_y 0.5` then scale it down.

Search `args.deg =='diy'` in `DDNM/runners/diffusion.py` and change the definition of $\mathbf{A}$ correspondingly.
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
2. Find the variant $\mathbf{x}\_{0|t}$ in the target codes, using the result of this function to modify the sampling of $\mathbf{x}\_{t-1}$. Your may need to provide the input degraded image $\mathbf{y}$ and the corresponding noise level $\sigma_\mathbf{y}$.
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
3. Actually, this repository contains the above simplified implementation: try search `arg.simplified` in `DDNM/runners/diffusion.py` for related codes. 




