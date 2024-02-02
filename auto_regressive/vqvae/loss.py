import torch.nn as nn

class VQVAELoss(nn.Module):
    def __init__(self, alpha=1, beta=0.25):
        super().__init__()
        self.alpha = alpha   # weight for embedding loss
        self.beta = beta     # weight for commitment loss
        self.mse_loss = nn.MSELoss()
    
    def forward(self, x, x_hat, ze, zq):
        reconstruction_loss = self.mse_loss(x, x_hat)
        embedding_loss = self.mse_loss(ze.detach(), zq)
        conmmitment_loss = self.mse_loss(ze, zq.detach())
        vqvae_loss = reconstruction_loss + self.alpha * embedding_loss + self.beta * conmmitment_loss
        return vqvae_loss
