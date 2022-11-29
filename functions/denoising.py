import torch
from tqdm import tqdm
import torchvision.utils as tvu
import torchvision
import os

class_num = 951


def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a

def inverse_data_transform(x):
    x = (x + 1.0) / 2.0
    return torch.clamp(x, 0.0, 1.0)

def ddnm_diffusion(x, seq, model, b, A_funcs, y, cls_fn=None, classes=None):
    with torch.no_grad():

        # setup iteration variables
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]

        # iterate over the timesteps
        for i, j in tqdm(zip(reversed(seq), reversed(seq_next))):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to('cuda')
            if cls_fn == None:
                et = model(xt, t)
            else:
                classes = torch.ones(xt.size(0), dtype=torch.long, device=torch.device("cuda"))*class_num
                et = model(xt, t, classes)
                et = et[:, :3]
                et = et - (1 - at).sqrt()[0, 0, 0, 0] * cls_fn(x, t, classes)

            if et.size(1) == 6:
                et = et[:, :3]

            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

            x0_t_hat = x0_t - A_funcs.A_pinv(
                A_funcs.A(x0_t.reshape(x0_t.size(0), -1)) - y.reshape(y.size(0), -1)
            ).reshape(*x0_t.size())
    
            eta = 0.85
            
            c1 = (1 - at_next).sqrt() * eta
            c2 = (1 - at_next).sqrt() * ((1 - eta ** 2) ** 0.5)
            xt_next = at_next.sqrt() * x0_t_hat + c1 * torch.randn_like(x0_t) + c2 * et

            x0_preds.append(x0_t.to('cpu'))
            xs.append(xt_next.to('cpu'))

    #return xs, x0_preds
    return [xs[-1]], [x0_preds[-1]]

def ddnm_plus_diffusion(x, seq, model, b, A_funcs, y, sigma_y, cls_fn=None, classes=None):
    with torch.no_grad():

        # setup iteration variables
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]

        # iterate over the timesteps
        for i, j in tqdm(zip(reversed(seq), reversed(seq_next))):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to('cuda')
            if cls_fn == None:
                et = model(xt, t)
            else:
                classes = torch.ones(xt.size(0), dtype=torch.long, device=torch.device("cuda"))*class_num
                et = model(xt, t, classes)
                et = et[:, :3]
                et = et - (1 - at).sqrt()[0, 0, 0, 0] * cls_fn(x, t, classes)

            if et.size(1) == 6:
                et = et[:, :3]

            # Eq. 12
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            
            eta = 0.85
            sigma_t = (1 - at_next).sqrt()[0, 0, 0, 0]

            # Eq. 17
            x0_t_hat = x0_t - A_funcs.Lambda(A_funcs.A_pinv(
                A_funcs.A(x0_t.reshape(x0_t.size(0), -1)) - y.reshape(y.size(0), -1)
            ).reshape(x0_t.size(0), -1), at_next.sqrt()[0, 0, 0, 0], sigma_y, sigma_t, eta).reshape(*x0_t.size())
            
            # Eq. 51
            xt_next = at_next.sqrt() * x0_t_hat + A_funcs.Lambda_noise(
                torch.randn_like(x0_t).reshape(x0_t.size(0), -1), 
                at_next.sqrt()[0, 0, 0, 0], sigma_y, sigma_t, eta, et.reshape(et.size(0), -1)).reshape(*x0_t.size())
    
            x0_preds.append(x0_t.to('cpu'))
            xs.append(xt_next.to('cpu'))
            
#             #ablation
#             if i%50==0:
#                 os.makedirs('/userhome/wyh/ddnm_open/debug/x0t', exist_ok=True)
#                 tvu.save_image(
#                     inverse_data_transform(x0_t[0]),
#                     os.path.join('/userhome/wyh/ddnm_open/debug/x0t', f"x0_t_{i}.png")
#                 )
                
#                 os.makedirs('/userhome/wyh/ddnm_open/debug/x0_t_hat', exist_ok=True)
#                 tvu.save_image(
#                     inverse_data_transform(x0_t_hat[0]),
#                     os.path.join('/userhome/wyh/ddnm_open/debug/x0_t_hat', f"x0_t_hat_{i}.png")
#                 )
                
#                 os.makedirs('/userhome/wyh/ddnm_open/debug/xt_next', exist_ok=True)
#                 tvu.save_image(
#                     inverse_data_transform(xt_next[0]),
#                     os.path.join('/userhome/wyh/ddnm_open/debug/xt_next', f"xt_next_{i}.png")
#                 )

    #return xs, x0_preds
    return [xs[-1]], [x0_preds[-1]]



