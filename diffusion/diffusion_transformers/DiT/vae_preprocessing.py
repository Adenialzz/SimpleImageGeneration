import argparse
import os
from tqdm import tqdm
import numpy as np

import torch
import torch.distributed as dist
from torch.cuda.amp import autocast as autocast
from torchvision.utils import save_image

from diffusers.models import AutoencoderKL

import sys
# from datasets import get_dataset
from ddae_utils import init_seeds, gather_tensor, DataLoaderDDP, print0
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from train import center_crop_arr


def show(imgs, title="debug.png"):
    save_image(imgs, title, normalize=True, value_range=(0, 1))


def main(args):
    local_rank = args.local_rank
    use_amp = args.use_amp

    save_dir = os.path.join('./latent_codes', 'folder')
    if local_rank == 0:
        os.makedirs(save_dir, exist_ok=False)

    device = "cuda:%d" % local_rank
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)

    def encode(img):
        with torch.no_grad():
            code = vae.encode(img.to(device) * 2 - 1)
            return 0.18215 * code.latent_dist.sample()

    def decode(code):
        with torch.no_grad():
            recon = vae.decode(code / 0.18215).sample.cpu()
            return (recon + 1) / 2

    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    dataset = ImageFolder(args.data_path, transform=transform)
    string = 'train'
    loader, sampler = DataLoaderDDP(
        dataset,
        batch_size=1,
        shuffle=False,
    )

    for ep in range(args.num_copies):
        sampler.set_epoch(ep)
        data = []
        label = []
        for i, (x, y) in enumerate(tqdm(loader, disable=(local_rank != 0))):
            x = x.to(device)
            y = y.to(device)
            with autocast(enabled=use_amp):
                code = encode(x).float()
                if local_rank == 0 and i == 0:
                    # for visualization and debugging
                    recon = decode(code).float()
                    show(x, f"{string}_debug_original_{ep}.png")
                    show(recon, f"{string}_debug_reconstruct_{ep}.png")

            dist.barrier()
            code = gather_tensor(code).cpu()
            data.append(code)
            if ep == 0:
                y = gather_tensor(y).cpu()
                label.append(y)

        if local_rank == 0:
            data = torch.cat(data)
            with open(os.path.join(save_dir, f"{string}_code_{ep}.npy"), 'wb') as f:
                np.save(f, data.numpy())
            if ep == 0:
                label = torch.cat(label)
                with open(os.path.join(save_dir, f"{string}_label.npy"), 'wb') as f:
                    np.save(f, label.numpy())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default='/media/data/imagenet/train', type=str)
    parser.add_argument("--image-size", default=256, type=int)
    parser.add_argument('--num-copies', default=10, type=int,
                        help='number of training data copies, higher = more augmentation variations')
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument("--use_amp", action='store_true', default=False)
    args = parser.parse_args()
    print0(args)

    init_seeds(no=args.local_rank)
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(args.local_rank)
    main(args)
