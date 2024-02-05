import os
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import utils as vutils
from models.discriminator import Discriminator
from models.vqgan import VQGAN
from lpips import LPIPS
from simgen.image_datasets import get_dataset_class
from simgen.utils import weights_init

class TrainVQGAN:
    def __init__(self, args):
        self.vqgan = VQGAN(args).to(device=args.device)
        self.discriminator = Discriminator(args).to(device=args.device)
        self.discriminator.apply(weights_init)
        self.perceptual_loss = LPIPS().eval().to(device=args.device)
        self.opt_vq, self.opt_disc = self.configure_optimizers(args)

        self.train(args)

    def configure_optimizers(self, args):
        opt_vq = torch.optim.Adam(
            list(self.vqgan.encoder.parameters()) +
            list(self.vqgan.decoder.parameters()) +
            list(self.vqgan.codebook.parameters()) +
            list(self.vqgan.quant_conv.parameters()) +
            list(self.vqgan.post_quant_conv.parameters()),
            lr=args.lr, eps=1e-08
        )
        opt_disc = torch.optim.Adam(self.discriminator.parameters(), lr=args.lr, eps=1e-08)

        return opt_vq, opt_disc

    def train(self, args):
        dataset = get_dataset_class(args.dataset_name)(args.data_root, img_size=args.img_size)
        print(f"num samples = {len(dataset)} in {args.dataset_name}")
        data_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=8)
        steps_per_epoch = len(data_loader)
        for epoch in range(args.epochs):
            with tqdm(range(len(data_loader))) as pbar:
                for i, imgs in zip(pbar, data_loader):
                    total_step = epoch * steps_per_epoch + i
                    if total_step < args.disc_start:
                        disc_factor = 0.
                    else:
                        disc_factor = args.disc_factor

                    imgs = imgs.to(device=args.device)
                    decoded_images, _, q_loss = self.vqgan(imgs)

                    disc_real = self.discriminator(imgs)
                    disc_fake = self.discriminator(decoded_images)

                    perceptual_loss = self.perceptual_loss(imgs, decoded_images)
                    rec_loss = torch.abs(imgs - decoded_images)
                    perceptual_rec_loss = args.perceptual_loss_factor * perceptual_loss + args.rec_loss_factor * rec_loss
                    perceptual_rec_loss = perceptual_rec_loss.mean()
                    g_loss = -torch.mean(disc_fake)

                    λ = self.vqgan.calculate_lambda(perceptual_rec_loss, g_loss)
                    vq_loss = perceptual_rec_loss + q_loss + disc_factor * λ * g_loss

                    d_loss_real = torch.mean(F.relu(1. - disc_real))
                    d_loss_fake = torch.mean(F.relu(1. + disc_fake))
                    gan_loss = disc_factor * 0.5 * (d_loss_real + d_loss_fake)

                    self.opt_vq.zero_grad()
                    vq_loss.backward(retain_graph=True)

                    self.opt_disc.zero_grad()
                    gan_loss.backward()

                    self.opt_vq.step()
                    self.opt_disc.step()

                    if i % 50 == 0:
                        with torch.no_grad():
                            real_fake_images = torch.cat((imgs[:4], decoded_images[:4]))
                            vutils.save_image(real_fake_images, os.path.join("results", f"ep{epoch}_step{i}.jpg"), nrow=4)

                    pbar.set_postfix(
                        VQ_Loss=np.round(vq_loss.cpu().detach().numpy().item(), 5),
                        GAN_Loss=np.round(gan_loss.cpu().detach().numpy().item(), 3)
                    )
                    pbar.update(0)
                torch.save(self.vqgan.state_dict(), os.path.join("checkpoints", f"vqgan_epoch_{epoch}.pt"))


if __name__ == '__main__':
    from simgen.utils import load_yaml, dict2namespace
    import sys
    config_path = sys.argv[1]
    os.makedirs("results", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    args = load_yaml(config_path)
    args = dict2namespace(args)

    train_vqgan = TrainVQGAN(args)



