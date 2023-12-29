import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import partial
from copy import deepcopy
from .ema import EMA
from .transform import UnrescaleChannels

# 用于从一系列值a（如 \alpha_{1...T}）中取第t个值
def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1, ) * (len(x_shape) - 1)))

class GaussianDiffusion(nn.Module):
    __doc__ = r"""Gaussian Diffusion model. Forwarding through the module returns diffusion reversal scalar loss tensor.

    Input:
        x: tensor of shape (N, img_channels, *img_size)
        y: tensor of shape (N)
    Output:
        scalar loss tensor
    Args:
        model (nn.Module): model which estimates diffusion noise
        img_size (tuple): image size tuple (H, W)
        img_channels (int): number of image channels
        betas (np.ndarray): numpy array of diffusion betas
        loss_type (string): loss type, "l1" or "l2"
        ema_decay (float): model weights exponential moving average decay
        ema_start (int): number of steps before EMA
        ema_update_rate (int): number of steps before each EMA update
    """
    
    def __init__(self, model, img_size, img_channels, num_classes, betas, loss_type='l2', ema_decay=0.9999, ema_start=5000, ema_update_rate=1):
        super().__init__()
        self.model = model
        self.ema_model = deepcopy(model)
        
        self.ema = EMA(ema_decay)
        self.ema_start = ema_start
        self.ema_update_rate = ema_update_rate
        self.step = 0

        self.img_size = img_size
        self.img_channels = img_channels
        self.num_classes = num_classes
        self.post_process = UnrescaleChannels()

        if loss_type not in ["l1", "l2"]:
            raise ValueError("__init__() got unknown loss type")
        
        self.loss_type = loss_type
        self.num_timesteps = len(betas)
        
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas)

        to_torch = partial(torch.tensor, dtype=torch.float32)
        
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas', to_torch(alphas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))

        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1 - alphas_cumprod)))
        self.register_buffer("reciprocal_sqrt_alphas", to_torch(np.sqrt(1 / alphas)))
        
        self.register_buffer("remove_noise_coeff", to_torch(betas / np.sqrt(1 - alphas_cumprod)))
        self.register_buffer("sigma", to_torch(np.sqrt(betas)))   # sigma = \sqrt{\bata}  ??
        
    def update_ema(self):
        self.step += 1
        if self.step % self.ema_update_rate == 0:
            if self.step < self.ema_start:
                self.ema_model.load_state_dict(self.model.state_dict())
            else:
                self.ema.update_model_average(self.ema_model, self.model)
                
    
    @torch.no_grad()
    def remove_noise(self, x, t, y, use_ema=True):
        # x_{t-1} = (x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}} * \tilde{z}) * \sqrt{\frac{1}{\bar\alpha_t}}
        # 没有加方差，随机采样的噪声
        if use_ema:
            return (
                (x - extract(self.remove_noise_coeff, t, x.shape) * self.ema_model(x, t, y)) *
                extract(self.reciprocal_sqrt_alphas, t, x.shape)
            )
        else:
            return (
                (x - extract(self.remove_noise_coeff, t, x.shape) * self.model(x, t, y)) *
                extract(self.reciprocal_sqrt_alphas, t, x.shape)
            )
        
    @torch.no_grad()
    def sample(self, batch_size, device, y=None, use_ema=True):
        if y is not None and batch_size != len(y):
            raise ValueError("sample batch size different from length of given y")
        
        x = torch.randn(batch_size, self.img_channels, *self.img_size, device=device)
        
        for t in range(self.num_timesteps - 1, -1, -1):
            t_batch = torch.tensor([t], device=device).repeat(batch_size)
            x = self.remove_noise(x, t_batch, y, use_ema)
            
            if t > 0:
                # + \sigma * z
                # 这里的标准差 \sigma = \sqrt{\beta_t}，好像少了个系数呢？？？？ 不过好像这里的系数本来就不一定是固定的，参考 DDIM
                x += extract(self.sigma, t_batch, x.shape) * torch.randn_like(x)
        
        return self.post_process(x.cpu().detach())

    @torch.no_grad()
    def sample_diffusion_sequence(self, batch_size, device, y=None, use_ema=True):
        # 也是sample的过程，不同在于把每一步的去噪结果都保存并返回了
        if y is not None and batch_size != len(y):
            raise ValueError("sample batch size different from length of given y")

        x = torch.randn(batch_size, self.img_channels, *self.img_size, device=device)
        diffusion_sequence = [x.cpu().detach()]
        
        for t in range(self.num_timesteps - 1, -1, -1):
            t_batch = torch.tensor([t], device=device).repeat(batch_size)
            x = self.remove_noise(x, t_batch, y, use_ema)
            
            if t > 0:
                x += extract(self.sigma, t_batch, x.shape) * torch.randn_like(x)
            
            diffusion_sequence.append(self.post_process(x.cpu().detach()))
            
        return diffusion_sequence
    
    def perturb_x(self, x_0, t, noise):
        # 这里的变量名x，我改成x_0了
        # x_t = \sqrt{\bar\alpha_t} * x_0 + \sqrt{1-\bar\alpha_t} * z
        return (
            extract(self.sqrt_alphas_cumprod, t, x_0.shape) * x_0 +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape) * noise
        )
        
    def get_losses(self, x, t, y):
        noise = torch.randn_like(x)
        
        x_t = self.perturb_x(x, t, noise)  # 变量名 perturbed_x -> x_t
        estimated_noise = self.model(x_t, t, y)
        
        if self.loss_type == 'l1':
            loss = F.l1_loss(estimated_noise, noise)
        elif self.loss_type == 'l2':
            loss = F.mse_loss(estimated_noise, noise)
        
        return loss

    def forward(self, x, y=None):
        b, c, h, w = x.shape
        device = x.device

        if h != self.img_size[0]:
            raise ValueError("image height does not match diffusion parameters")
        if w != self.img_size[0]:
            raise ValueError("image width does not match diffusion parameters")
        
        t = torch.randint(0, self.num_timesteps, (b, ), device=device)
        loss = self.get_losses(x, t, y)
        return loss
    
def generate_linear_schedule(T, low, high):
    return np.linspace(low, high, T)
    
def generate_cosine_schedule(T, s=0.008):
    def f(t, T):
        return (np.cos((t / T + s) / (1 + s) * np.pi / 2)) ** 2
    
    alphas = []
    f0 = f(0, T)
    for t in range(T + 1):
        alphas.append(f(t, T) / f0)
    
    betas = []
    for t in range(1, T + 1):
        betas.append(min(1 - alphas[t] / alphas[t-1], 0.999))
        
    return np.array(betas)
