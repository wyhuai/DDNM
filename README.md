# DDNM
## 🌟Brief
This repository contains the code release for *Zero Shot Image Restoration Using ***D***enoising ***D***iffusion ***N***ull-Space ***M***odel*.

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

## 🌟Evaluation
### Quick Start
Run below command to get results immediately. The results should be in "DDNM/exp/image_samples/demo_sr4".
```
python main.py --ni -s simplified --config celeba_hq_simple_ddnm.yml --doc celeba_hq --timesteps 100 --eta 0.85 --deg "sr" --sigma_y 0 -i demo_sr4
```
### Reproduce the quantitative result in the paper.
Download this CelebA testset and put it into "DDNM/exp/datasets/celeba/".

Download this ImageNet testset and put it into "DDNM/exp/datasets/imagenet/".

```
cd DDNM
sh demo.sh
```

## 😊Applying DDNM to Your Own Diffusion Model
It is ***very easy*** to implement a basic DDNM on your own diffusion model! You may reference the following:
1. Copy these operator implementations to the core diffusion sampling file, then define your task type, e.g., IR_mode="super resolution".
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
3. Actually, this repository contains the above simplified implementation: try use "-s simplified". 




