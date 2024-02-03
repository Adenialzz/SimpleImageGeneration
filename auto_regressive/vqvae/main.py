import os
import os.path as osp
import sys
import time

import cv2
import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.vqvae import VQVAE
from models.pixelcnn import PixelCNNWithEmbedding
from loss import VQVAELoss
from simgen.utils import load_yaml, dict2namespace, time_cost
from simgen.image_datasets import get_dataset_class

def train_vqvae(cfg, model: VQVAE, dataloader: DataLoader):
    model.to(cfg.device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), cfg.lr)
    vqvae_loss_func = VQVAELoss(alpha=cfg.alpha, beta=cfg.beta)
    tic = time.time()
    for e in range(cfg.vqvae_epochs):
        total_loss = 0

        for x in dataloader:
            current_batch_size = x.shape[0]
            x = x.to(cfg.device)

            x_hat, ze, zq = model(x)
            loss = vqvae_loss_func(x, x_hat, ze, zq)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * current_batch_size
        total_loss /= len(dataloader.dataset)
        toc = time.time()
        save_path = osp.join(cfg.save_dir, 'vqvae.pth')
        torch.save(model.state_dict(), save_path)
        print(f'epoch {e} loss: {total_loss} elapsed {time_cost(toc - tic)}, model updated at {save_path}')
    print('All epochs for training vqvae done')

def train_generative_model(cfg, vqvae, model, dataloader):
    vqvae.to(cfg.device)
    vqvae.eval()
    model.to(cfg.device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), 1e-3)
    loss_fn = nn.CrossEntropyLoss()
    tic = time.time()
    for e in range(cfg.gen_model_epochs):
        total_loss = 0
        for x in dataloader:
            current_batch_size = x.shape[0]
            with torch.no_grad():
                x = x.to(cfg.device)
                x = vqvae.encode(x)

            predict_x = model(x)
            loss = loss_fn(predict_x, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * current_batch_size
        total_loss /= len(dataloader.dataset)
        toc = time.time()
        save_path = osp.join(cfg.save_dir, 'gen_model.pth')
        torch.save(model.state_dict(), save_path)
        print(f'epoch {e} loss: {total_loss} elapsed {time_cost(toc - tic)}')
    print('All epochs for training generative model done')


def reconstruct(cfg, model, x):
    model.to(cfg.device)
    model.eval()
    with torch.no_grad():
        x_hat, _, _ = model(x)
    bs = x.shape[0]
    n1 = int(bs ** 0.5)
    x_cat = torch.concat((x, x_hat), dim=3)
    x_cat = einops.rearrange(x_cat, '(n1 n2) c h w -> (n1 h) (n2 w) c', n1=n1)
    x_cat = (x_cat.clip(0, 1) * 255).cpu().numpy().astype(np.uint8)
    if cfg.dataset_name.lower() == 'celeb':
        x_cat = cv2.cvtColor(x_cat, cv2.COLOR_RGB2BGR)
    save_path = osp.join(cfg.save_dir, f'vqvae_reconstruct_{cfg.dataset_name}.jpg')
    cv2.imwrite(save_path, x_cat)


def sample_imgs(cfg, vqvae, gen_model):
    vqvae = vqvae.to(cfg.device)
    vqvae.eval()
    gen_model = gen_model.to(cfg.device)
    gen_model.eval()

    latent_h, latent_w = vqvae.get_latent_HW((cfg.img_channels, cfg.img_size, cfg.img_size))
    input_shape = (cfg.n_sample, latent_h, latent_w)
    x = torch.zeros(input_shape).to(cfg.device).to(torch.long)
    with torch.no_grad():
        for i in range(latent_h):
            for j in range(latent_w):
                output = gen_model(x)
                prob_dist = F.softmax(output[:, :, i, j], -1)
                pixel = torch.multinomial(prob_dist, 1)
                x[:, i, j] = pixel[:, 0]

    imgs = vqvae.decode(x)

    imgs = imgs * 255
    imgs = imgs.clip(0, 255)
    imgs = einops.rearrange(imgs,  # make square to display
                            '(n1 n2) c h w -> (n1 h) (n2 w) c',
                            n1=int(cfg.n_sample ** 0.5))

    imgs = imgs.detach().cpu().numpy().astype(np.uint8)
    if cfg.dataset_name.lower() == 'celeba':
        imgs = cv2.cvtColor(imgs, cv2.COLOR_RGB2BGR)

    # cv2.imwrite(f'work_dirs/vqvae_sample_{cfg.dataset_name}.jpg', imgs)
    save_path = osp.join(cfg.save_dir, f"gen_model_vqvae_sample_{cfg.dataset_name}.jpg")
    cv2.imwrite(save_path, imgs)

if __name__ == '__main__':

    config_path = sys.argv[1]
    cfg = load_yaml(config_path)
    cfg = dict2namespace(cfg)
    os.makedirs(cfg.save_dir, exist_ok=True)

    dataset = get_dataset_class(cfg.dataset_name)(data_root=cfg.data_root, img_size=cfg.img_size)
    dataloader = DataLoader(dataset, batch_size=cfg.vqvae_batch_size, num_workers=8)
    vqvae = VQVAE(cfg.img_channels, cfg.dim, cfg.n_embedding)

    # 1. Train VQVAE
    train_vqvae(cfg, vqvae, dataloader)
    vqvae.eval()  # 之后都是eval了

    # 2. Test VQVAE by visualizaing reconstruction result
    # vqvae.load_state_dict(torch.load(cfg.vqvae_model_path, map_location='cpu'))
    dataloader = DataLoader(dataset, batch_size=cfg.gen_model_batch_size, num_workers=8)
    img = next(iter(dataloader)).to(cfg.device)
    reconstruct(cfg, vqvae, img)

    # 3. Train Generative model (Gated PixelCNN in our project)
    gen_model = PixelCNNWithEmbedding(cfg.pixelcnn_n_blocks, cfg.pixelcnn_dim, cfg.pixelcnn_linear_dim, True, cfg.n_embedding) 
    gen_model.train()
    dataset = get_dataset_class(cfg.dataset_name)(data_root=cfg.data_root, img_size=cfg.img_size)
    train_generative_model(cfg, vqvae, gen_model, dataloader)

    # 4. Sample VQVAE
    # vqvae.load_state_dict(torch.load(cfg.vqvae_model.path, map_location='cpu'))
    # gen_model.load_state_dict(torch.load(cfg.gen_model_path, map_location='cpu'))
    gen_model.eval()
    sample_imgs(cfg, vqvae, gen_model)
