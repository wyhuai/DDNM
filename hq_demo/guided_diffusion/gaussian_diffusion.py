# Copyright (c) 2022 Huawei Technologies Co., Ltd.
# Licensed under CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike 4.0 International) (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#
# The code is released for academic research use only. For commercial use, please contact Huawei Technologies Co., Ltd.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This repository was forked from https://github.com/openai/guided-diffusion, which is under the MIT license

"""
This code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py

Docstrings have been added, as well as DDIM sampling and a new collection of beta schedules.
"""

import enum

import numpy as np
import torch as th
import torch
import os
from PIL import Image

from collections import defaultdict

from guided_diffusion.scheduler import get_schedule_jump
from tqdm.auto import tqdm

import math


def tensor2im(var):
    # var shape: (3, H, W)
    var = var.cpu().detach().transpose(0, 2).transpose(0, 1).numpy()
    var = ((var + 1) / 2)
    var[var < 0] = 0
    var[var > 1] = 1
    var = var * 255
    return Image.fromarray(var.astype('uint8'))

def save_image(img, save_dir, idx):
    result = tensor2im(img)
    im_save_path = os.path.join(save_dir, f"{idx:05d}.png")
    Image.fromarray(np.array(result)).save(im_save_path)
    
def color2gray(x):
    coef=1/3
    x = x[:,0,:,:] * coef + x[:,1,:,:]*coef +  x[:,2,:,:]*coef
    return x.repeat(1,3,1,1)

def gray2color(x):
    x = x[:,0,:,:]
    coef=1/3
    base = coef**2 + coef**2 + coef**2
    return torch.stack((x*coef/base, x*coef/base, x*coef/base), 1)    

def MeanUpsample(x, scale):
    n, c, h, w = x.shape
    out = th.zeros(n, c, h, scale, w, scale).to(x.device) + x.view(n,c,h,1,w,1)
    out = out.view(n, c, scale*h, scale*w)
    return out

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps, use_scale):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.

        if use_scale:
            scale = 1000 / num_diffusion_timesteps
        else:
            scale = 1

        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )

class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.

    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = (
        enum.auto()
    )  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL


class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """

    def __init__(
        self,
        *,
        betas,
        model_mean_type,
        model_var_type,
        loss_type,
        rescale_timesteps=False,
        conf=None
    ):
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps

        self.conf = conf

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_prev_prev = np.append(
            1.0, self.alphas_cumprod_prev[:-1])

        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)

        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_alphas_cumprod_prev = np.sqrt(self.alphas_cumprod_prev)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(
            1.0 / self.alphas_cumprod - 1)

        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) /
            (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) /
            (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

    def undo(self, image_before_step, img_after_model, est_x_0, t, debug=False):
        return self._undo(img_after_model, t)

    def _undo(self, img_out, t):
        beta = _extract_into_tensor(self.betas, t, img_out.shape)

        img_in_est = th.sqrt(1 - beta) * img_out + \
            th.sqrt(beta) * th.randn_like(img_out)

        return img_in_est

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1,
                                 t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2,
                                   t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(
            self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
        self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'x0_t': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        assert t.shape == (B,)

        model_output = model(x, self._scale_timesteps(t), **model_kwargs)

        assert model_output.shape == (B, C * 2, *x.shape[2:])
        model_output, model_var_values = th.split(model_output, C, dim=1)

        if self.model_var_type == ModelVarType.LEARNED:
            model_log_variance = model_var_values
            model_variance = th.exp(model_log_variance)
        else:
            min_log = _extract_into_tensor(
                self.posterior_log_variance_clipped, t, x.shape
            )
            max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
            frac = (model_var_values + 1) / 2
            model_log_variance = frac * max_log + (1 - frac) * min_log
            model_variance = th.exp(model_log_variance)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x
        


        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            x0_t = process_xstart(
                self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
            )
            model_mean = model_output
        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
            if self.model_mean_type == ModelMeanType.START_X:
                x0_t = process_xstart(model_output)
            else:
                x0_t = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
                )
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=x0_t, x_t=x, t=t
            )
            

            ## DDNM core ##
            
            if x0_t is not None:

                A, Ap = model_kwargs['A'], model_kwargs['Ap']
                sigma_y = model_kwargs['sigma_y']
                #y = model_kwargs['y_img']
                Apy = model_kwargs['Apy']
                
                sigma_t = th.sqrt(_extract_into_tensor(self.posterior_variance, t, x0_t.shape))[0][0][0][0]
                a_t = _extract_into_tensor(self.posterior_mean_coef1, t, x0_t.shape)[0][0][0][0]

                # Eq. 19
                if sigma_t >= a_t*sigma_y:
                    lambda_t = 1
                    gamma_t = _extract_into_tensor(self.posterior_variance, t, x0_t.shape) - (a_t*lambda_t*sigma_y)**2
                else:
                    lambda_t = sigma_t/a_t*sigma_y
                    gamma_t = 0.

                # Eq. 17
                #x0_t_hat = x0_t + lambda_t*Ap(y-A(x0_t))
                x0_t_hat = lambda_t*Apy + x0_t - lambda_t*Ap(A(x0_t))
                
                

                # mask-shift trick 
                if model_kwargs['shift_w']==0 and model_kwargs['shift_h']==0:
                    pass
                elif model_kwargs['shift_w']==0 and model_kwargs['shift_h']!=0:
                    h_l = int(128*model_kwargs['shift_h'])
                    h_r = h_l+128
                    if (model_kwargs['shift_h']==model_kwargs['shift_h_total']-1) and (model_kwargs['H_target']%128!=0):
                        h_l = h_l-128+model_kwargs['H_target']%128
                        x0_t_hat[:,:,0:256-model_kwargs['H_target']%128,:] = model_kwargs['x_temp'][:,:,h_l:h_r,0:256].to('cuda')
                    else:
                        x0_t_hat[:,:,0:128,:] = model_kwargs['x_temp'][:,:,h_l:h_r,0:256].to('cuda')
                else:
                    w_l = int(128*model_kwargs['shift_w'])
                    w_r = w_l+128
                    h_l = int(128*model_kwargs['shift_h'])
                    h_r = h_l+256
                    if (model_kwargs['shift_w']==model_kwargs['shift_w_total']-1) and (model_kwargs['W_target']%128!=0): 
                        w_l = w_l-128+model_kwargs['W_target']%128
                        if (model_kwargs['shift_h']==model_kwargs['shift_h_total']-1) and (model_kwargs['H_target']%128!=0): 
                            h_l_tmp = h_l-128+model_kwargs['H_target']%128
                            x0_t_hat[:,:,:,0:256-model_kwargs['W_target']%128] = model_kwargs['x_temp'][:,:,h_l_tmp:h_r,w_l:w_r].to('cuda')
                        else:
                            x0_t_hat[:,:,:,0:256-model_kwargs['W_target']%128] = model_kwargs['x_temp'][:,:,h_l:h_r,w_l:w_r].to('cuda')
                    else:
                        if (model_kwargs['shift_h']==model_kwargs['shift_h_total']-1) and (model_kwargs['H_target']%128!=0):
                            h_l_tmp = h_l-128+model_kwargs['H_target']%128
                            x0_t_hat[:,:,:,0:128] = model_kwargs['x_temp'][:,:,h_l_tmp:h_r,w_l:w_r].to('cuda')
                        else:
                            x0_t_hat[:,:,:,0:128] = model_kwargs['x_temp'][:,:,h_l:h_r,w_l:w_r].to('cuda')
                    if model_kwargs['shift_h']!=0:
                        h_r = h_l+128
                        w_r = w_l+256
                        if (model_kwargs['shift_h']==model_kwargs['shift_h_total']-1) and (model_kwargs['H_target']%128!=0):
                            h_l = h_l-128+model_kwargs['H_target']%128
                            x0_t_hat[:,:,0:256-model_kwargs['H_target']%128,:] = model_kwargs['x_temp'][:,:,h_l:h_r,w_l:w_r].to('cuda')
                        else:
                            x0_t_hat[:,:,0:128,:] = model_kwargs['x_temp'][:,:,h_l:h_r,w_l:w_r].to('cuda')

                # save intermediate results
                if t[0]%25==0:
                    image_savepath = os.path.join('results/'+model_kwargs['save_path']+'/'+str(model_kwargs['shift_h'])+'_'+str(model_kwargs['shift_w']))
                    os.makedirs(image_savepath, exist_ok=True)
                    save_image(x0_t_hat[0], image_savepath, t[0])
            
                model_mean, _, _ = self.q_posterior_mean_variance(x_start=x0_t_hat, x_t=x, t=t)
                model_variance = gamma_t # model_variance                
            
        else:
            raise NotImplementedError(self.model_mean_type)

        assert (
            model_mean.shape == model_log_variance.shape == x0_t_hat.shape == x.shape
        )

        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "x0_t": x0_t_hat,
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(
                self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def condition_mean(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.

        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """

        gradient = cond_fn(x, self._scale_timesteps(t), **model_kwargs)


        new_mean = (
            p_mean_var["mean"].float() + p_mean_var["variance"] *
            gradient.float()
        )
        return new_mean
    
    def p_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        conf=None,
        meas_fn=None,
        x0_t=None,
        idx_wall=-1
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'x0_t': a prediction of x_0.
        """
        

        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs
        )

        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        ) 

        if cond_fn is not None:
            out["mean"] = self.condition_mean(
                cond_fn, out, x, t, model_kwargs=model_kwargs
            )

        noise = th.randn_like(x)
        #sample = out["mean"] + nonzero_mask * \
        #    th.exp(0.5 * out["log_variance"]) * noise# - out["xt_grad"]
        
        sample = out["mean"] + nonzero_mask * \
            th.sqrt(th.ones(1,device='cuda')*out["variance"]) * noise

        result = {"sample": sample,
                  "x0_t": out["x0_t"], 'gt': model_kwargs.get('gt')}

        return result

    def p_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=True,
        return_all=False,
        conf=None
    ):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = None
        for sample in self.p_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            conf=conf
        ):
            final = sample

        if return_all:
            return final
        else:
            return final["sample"]

    def p_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        conf=None
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            image_after_step = noise
        else:
            image_after_step = th.randn(*shape, device=device)

        x0_t = None


        # initialization
        gt = model_kwargs['gt'] 
        scale = model_kwargs['scale'] 

        if 256%scale!=0:
            raise ValueError("Please set a SR scale divisible by 256")
        if gt.shape[2]!=256 and conf.name=='face256':
            print("gt.shape:",gt.shape)
            raise ValueError("Only support output size 256x256 for face images")

        if model_kwargs['resize_y']:
            resize_y = lambda z: MeanUpsample(z,scale)
            gt = resize_y(gt)
            

        if model_kwargs['deg']=='sr_averagepooling':
            scale=model_kwargs['scale'] 
            A = torch.nn.AdaptiveAvgPool2d((256//scale,256//scale))
            Ap = lambda z: MeanUpsample(z,scale)

            A_temp = torch.nn.AdaptiveAvgPool2d((gt.shape[2]//scale,gt.shape[3]//scale))
        elif model_kwargs['deg']=='inpainting' and conf.name=='face256':
            mask = model_kwargs.get('gt_keep_mask')
            A = lambda z: z*mask
            Ap = A

            A_temp = A
        elif model_kwargs['deg']=='mask_color_sr' and conf.name=='face256':
            mask = model_kwargs.get('gt_keep_mask')
            A1 = lambda z: z*mask
            A1p = A1
            
            A2 = lambda z: color2gray(z)
            A2p = lambda z: gray2color(z)
            
            scale=model_kwargs['scale']
            A3 = torch.nn.AdaptiveAvgPool2d((256//scale,256//scale))
            A3p = lambda z: MeanUpsample(z,scale)
            
            A = lambda z: A3(A2(A1(z)))
            Ap = lambda z: A1p(A2p(A3p(z)))

            A_temp = A    
        elif model_kwargs['deg']=='colorization':
            A = lambda z: color2gray(z)
            Ap = lambda z: gray2color(z)

            A_temp = A
        elif model_kwargs['deg']=='sr_color':
            scale=model_kwargs['scale'] 
            A1 = torch.nn.AdaptiveAvgPool2d((256//scale,256//scale))
            A1p = lambda z: MeanUpsample(z,scale)
            A2 = lambda z: color2gray(z)
            A2p = lambda z: gray2color(z)
            A = lambda z: A2(A1(z))
            Ap = lambda z: A1p(A2p(z))

            A1_temp = torch.nn.AdaptiveAvgPool2d((gt.shape[2]//scale,gt.shape[3]//scale))
            A_temp = lambda z: A2(A1_temp(z))            
        else:
            raise NotImplementedError("degradation type not supported")         

        model_kwargs['A'] = A
        model_kwargs['Ap'] = Ap

        y_temp = A_temp(gt)
        Apy_temp = Ap(y_temp)

        H_target, W_target = Apy_temp.shape[2], Apy_temp.shape[3]
        model_kwargs['H_target'] = H_target
        model_kwargs['W_target'] = W_target

        if H_target<256 or W_target<256:
            raise ValueError("Please set a larger SR scale")

        image_savepath = os.path.join('results/'+model_kwargs['save_path']+'/Apy')
        os.makedirs(image_savepath, exist_ok=True)
        save_image(Apy_temp[0], image_savepath, 0)
        
        image_savepath = os.path.join('results/'+model_kwargs['save_path']+'/y')
        os.makedirs(image_savepath, exist_ok=True)
        save_image(y_temp[0], image_savepath, 0)

        finalresult = torch.zeros_like(Apy_temp)

        shift_h_total = math.ceil(H_target/128)-1
        shift_w_total = math.ceil(W_target/128)-1
        model_kwargs['shift_h_total'] = shift_h_total
        model_kwargs['shift_w_total'] = shift_w_total

        with tqdm(total=shift_h_total*shift_w_total) as pbar:
            pbar.set_description('total shifts')

            # shift along H
            for shift_h in range(shift_h_total):
                h_l = int(128*shift_h)
                h_r = h_l+256
                if (shift_h==shift_h_total-1) and (H_target%128!=0): # for the last irregular shift_h
                    h_r = Apy_temp.shape[2]
                    h_l = h_r-256

                # shift along W
                for shift_w in range(shift_w_total):

                    x_temp=finalresult
                    w_l = int(128*shift_w)
                    w_r = w_l+256
                    if (shift_w==shift_w_total-1) and (W_target%128!=0): # for the last irregular shift_w
                        w_r = Apy_temp.shape[3]
                        w_l = w_r-256

                    # get the shifted y
                    Apy = Apy_temp[:,:,h_l:h_r,w_l:w_r]

                    model_kwargs['shift_w'] = shift_w
                    model_kwargs['shift_h'] = shift_h
                    # model_kwargs['y_img'] = y
                    model_kwargs['Apy'] = Apy
                    model_kwargs['x_temp'] = x_temp

                    times = get_schedule_jump(**conf.schedule_jump_params)
                    time_pairs = list(zip(times[:-1], times[1:]))

                    # DDNM loop
                    for t_last, t_cur in tqdm(time_pairs):
                        t_last_t = th.tensor([t_last] * shape[0],
                                             device=device)

                        # normal DDNM sampling
                        if t_cur < t_last:  
                            with th.no_grad():
                                image_before_step = image_after_step.clone()
                                out = self.p_sample(
                                    model,
                                    image_after_step,
                                    t_last_t,
                                    clip_denoised=clip_denoised,
                                    denoised_fn=denoised_fn,
                                    cond_fn=cond_fn,
                                    model_kwargs=model_kwargs,
                                    conf=conf,
                                    x0_t=x0_t
                                )
                                image_after_step = out["sample"]
                                x0_t = out["x0_t"]

                        # time-travel back
                        else:
                            t_shift = conf.get('inpa_inj_time_shift', 1)

                            image_before_step = image_after_step.clone()
                            image_after_step = self.undo(
                                image_before_step, image_after_step,
                                est_x_0=out['x0_t'], t=t_last_t+t_shift, debug=False)
                            x0_t = out["x0_t"]

                    # save the shifted result
                    if (shift_w==shift_w_total-1) and (W_target%128!=0):
                        if (shift_h==shift_h_total-1) and (H_target%128!=0):
                            finalresult[:,:,int(128*shift_h)-128+H_target%128:int(128*shift_h)+128+H_target%128,int(128*shift_w)-128+W_target%128:int(128*shift_w)+128+W_target%128] = out["x0_t"]
                        else:
                            finalresult[:,:,int(128*shift_h):int(128*shift_h)+256,int(128*shift_w)-128+W_target%128:int(128*shift_w)+128+W_target%128] = out["x0_t"]
                    else:
                        if (shift_h==shift_h_total-1) and (H_target%128!=0):
                            finalresult[:,:,int(128*shift_h)-128+H_target%128:int(128*shift_h)+128+H_target%128,int(128*shift_w):int(128*shift_w)+256] = out["x0_t"]
                        else:
                            finalresult[:,:,int(128*shift_h):int(128*shift_h)+256,int(128*shift_w):int(128*shift_w)+256] = out["x0_t"]

                    pbar.update(1)

        # finish!
        image_savepath = os.path.join('results/'+model_kwargs['save_path']+'/final')
        os.makedirs(image_savepath, exist_ok=True)
        save_image(finalresult[0], image_savepath, 0)

        out["sample"] = finalresult
        return out

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)
