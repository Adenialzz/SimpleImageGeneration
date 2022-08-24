import torch

from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch.optim import Adam

import argparse
import os
import os.path as osp

from models import Encoder, Decoder, VaeModel
from losses import vae_loss

def parse_cfg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--dataset_path', type=str, default='./data')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--x_dim', type=int, default=784)
    parser.add_argument('--hidden_dim', type=int, default=400)
    parser.add_argument('--latent_dim', type=int, default=200)

    cfg = parser.parse_args()
    return cfg


def main(cfg):

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    mnist_transform = transforms.Compose([
        transforms.ToTensor()
        ])

    train_set = MNIST(cfg.dataset_path, transform=mnist_transform, train=True, download=True)
    test_set = MNIST(cfg.dataset_path, transform=mnist_transform, train=False, download=True)

    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=10, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=10, pin_memory=True)


    encoder = Encoder(input_dim=cfg.x_dim, hidden_dim=cfg.hidden_dim, latent_dim=cfg.latent_dim)
    decoder = Decoder(output_dim=cfg.x_dim, hidden_dim=cfg.hidden_dim, latent_dim=cfg.latent_dim)

    model = VaeModel(encoder, decoder, DEVICE).to(DEVICE)

    optimizer = Adam(model.parameters(), lr=cfg.lr)

    # Training
    print('Start Training VAE.')
    model.train()
    for epoch in range(cfg.epochs):
        overall_loss = 0
        for idx, (x, _) in enumerate(train_loader):
            bs = x.shape[0]
            x = x.view(bs, cfg.x_dim).to(cfg.device)

            optimizer.zero_grad()
            x_hat, mean, log_var = model(x)
            loss = vae_loss(x, x_hat, mean, log_var)
            
            overall_loss += loss.item()

            loss.backward()
            optimizer.step()

        print(f"\tEpoch {epoch+1} Complete, Averager Loss: {overall_loss / (idx * cfg.batch_size):.4f}")

    print('Finish Training.')

    # Generating from random noise
    model.eval()
    n_samples = 64
    with torch.no_grad():
        noise = torch.randn(n_samples, cfg.latent_dim).to(cfg.device)
        generated_images = decoder(noise)
        save_image(generated_images.view(n_samples, 1, 28, 28), 'generated_sample.jpg')


if __name__ == '__main__':
    cfg = parse_cfg()
    main(cfg)


