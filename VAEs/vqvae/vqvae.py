import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from tqdm import tqdm
from torch.optim import Adam
from torchvision.utils import save_image, make_grid
import cv2
import argparse

from models import VQEmbeddingEMA, Encoder, Decoder, VQVaeModel

# Model Hyperparameters


from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt




input_dim = 3
hidden_dim = 128
n_embeddings= 768
output_dim = 3
img_size = (32, 32) # (width, height)

print_step = 50

def parse_cfg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--dataset_path', type=str, default='./data')
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--x_dim', type=int, default=784)
    parser.add_argument('--hidden_dim', type=int, default=400)
    parser.add_argument('--latent_dim', type=int, default=200)
    parser.add_argument('--img_size', type=int, default=32*32)

    cfg = parser.parse_args()
    return cfg

def main(cfg):
    cifar_transform = transforms.Compose([
            transforms.ToTensor(),
    ])

    train_dataset = CIFAR10(cfg.dataset_path, transform=cifar_transform, train=True, download=True)
    test_dataset  = CIFAR10(cfg.dataset_path, transform=cifar_transform, train=False, download=True)

    train_loader = DataLoader(dataset=train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=10, pin_memory=True)
    test_loader  = DataLoader(dataset=test_dataset,  batch_size=1, shuffle=False,  num_workers=10, pin_memory=True)
        
    encoder = Encoder(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=hidden_dim)
    codebook = VQEmbeddingEMA(n_embeddings=n_embeddings, embedding_dim=hidden_dim)
    decoder = Decoder(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=output_dim)

    model = VQVaeModel(Encoder=encoder, Codebook=codebook, Decoder=decoder).to(cfg.device)
    rect_loss = nn.MSELoss()

    optimizer = Adam(model.parameters(), lr=cfg.lr)
    print("Start training VQ-VAE...")
    model.train()

    for epoch in range(cfg.epochs):
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.to(cfg.device)

            optimizer.zero_grad()

            x_hat, commitment_loss, codebook_loss, perplexity = model(x)
            recon_loss = rect_loss(x_hat, x)
            
            loss =  recon_loss + commitment_loss + codebook_loss
                    
            loss.backward()
            optimizer.step()
            
            if batch_idx % print_step ==0: 
                print("epoch:", epoch + 1, "  step:", batch_idx + 1, "  recon_loss:", recon_loss.item(), "  perplexity: ", perplexity.item(), 
                  "\n\t\tcommit_loss: ", commitment_loss.item(), "  codebook loss: ", codebook_loss.item(), "  total_loss: ", loss.item())
        
    print("Training Finish!!")

    model.eval()
    with torch.no_grad():

        for x, _ in train_loader:

            x = x.to(cfg.device)
            x_hat, commitment_loss, codebook_loss, perplexity = model(x)
     
            print("perplexity: ", perplexity.item(),"commit_loss: ", commitment_loss.item(), "  codebook loss: ", codebook_loss.item())
            cv2.imwrite('original_image.jpg', (x[0].detach().cpu().numpy().transpose(1, 2, 0) * 255.0)[:, :, ::-1])
            cv2.imwrite('rect_image.jpg', (x_hat[0].detach().cpu().numpy().transpose(1, 2, 0) * 255.0)[:, :, ::-1])
            break

    indices_shape = (2, 32 // 4, 32 // 4)
    import pdb; pdb.set_trace()
    random_indices = torch.floor(torch.rand(indices_shape) * n_embeddings).long().to(cfg.device)
    codes = codebook.retrieve_random_codebook(random_indices).to(cfg.device)
    x_hat = decoder(codes)
    cv2.imwrite('randeom_gen_image.jpg', (x_hat[0].detach().cpu().numpy().transpose(1, 2, 0) * 255.0)[:, :, ::-1])


if __name__ == '__main__':
    cfg = parse_cfg()
    main(cfg)



