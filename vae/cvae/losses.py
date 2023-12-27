
import torch
def vae_loss(X, mu_prime, mu, log_var):
    # reconstruction_loss = F.mse_loss(mu_prime, X, reduction='mean') is wrong!
    reconstruction_loss = torch.mean(torch.square(X - mu_prime).sum(dim=1))

    latent_loss = torch.mean(0.5 * (log_var.exp() + torch.square(mu) - log_var).sum(dim=1))
    return reconstruction_loss + latent_loss
