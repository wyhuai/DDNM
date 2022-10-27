# Zero Shot Image Restoration Using Denoising Diffusion Null-Space Model

![image](https://user-images.githubusercontent.com/95485229/198285474-ff2e43de-9fc5-40c4-840b-f902bac4fa3c.png)

## Brief
This repository contains the code release for ***Zero Shot Image Restoration Using Denoising Diffusion Null-Space Model (DDNM)***.
***Supported Applications:***
- Super-Resolution
- Inpainting
- Deblurring
- Compressed Sensing
- Old Photo Restoration🌟


## Installation
`git clone https://github.com/wyhuai/DDNM.git; cd nerfocus`  
`conda install pip; pip install --upgrade pip`  
`pip install -r requirements.txt`  

## Evaluation
We provide a pretrained model in experiments/horns, so you can run the following command to generate a video with defocus effects. You may change the lens parameters "l" and "a" in eval_vid.py to adjust the focus distance and aperture size. 
`python -m eval_vid --data_dir=horns --train_dir=experiments/horns --chunk=3196 --gin_file=configs/llff.gin --logtostderr`

## Data
You can download the datasets from the [NeRF official Google Drive](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1). 

### Generate multi-blur datasets
You can generate the multi-blur datasets by running datatool.py, remember to change your desired data path and the blur kernel size. 

## Training
Run the following command, make sure the path is correct. You also need to change the path inside train.py to your data path.  
`python -m train --data_dir=horns --train_dir=experiments/horns --gin_file=configs/llff.gin --logtostderr`  


You can also train your own dataset, as long as it confroms to NeRF data format.  


## Results
[Click to watch video demonstration](https://www.bilibili.com/video/BV1ZV4y1x7mj?spm_id_from=333.999.0.0&vd_source=76ca20f72a423dd9323e2492733dffa5)  

![image](https://user-images.githubusercontent.com/95485229/157253266-c9c70953-9a7e-4f84-b10a-e5d1dbccdb95.png)
![image](https://user-images.githubusercontent.com/95485229/157253365-d5d371f0-192b-4ea8-9ed6-7364848ea767.png)
![image](https://user-images.githubusercontent.com/95485229/157254773-1d30b1de-27f5-4b82-b106-024698255c36.png)




