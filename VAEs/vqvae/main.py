import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import argparse

from new_models import VQVAE, GatedPixelCNN

def parse_cfg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--dataset_path', type=str, default='./data')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--x_dim', type=int, default=784)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--latent_dim', type=int, default=200)

    cfg = parser.parse_args()
    return cfg

def main(cfg):
    if cfg.dataset == 'cifar10':
        image_size = 32
        in_dim = 3
        prior_size = (8, 8) # h, w
        embedding_dim = 64
        num_embeddings = 512
    elif cfg.dataset == 'mnist':
        image_size = 28
        in_dim = 1
        prior_size = (7, 7)
        embedding_dim = 16
        num_embeddings = 128
    else:
        image_size = 0
        in_dim = 0
        prior_size = (0, 0)
        embedding_dim = 0
        num_embeddings = 0


    transform=transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset1 = datasets.CIFAR10('./data', train=True, download=True,
                           transform=transform)
    dataset2 = datasets.CIFAR10('./data', train=False,
                           transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, batch_size=cfg.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset2, batch_size=cfg.batch_size)

    # compute the variance of the whole training set to normalise the Mean Squared Error below.
    train_images = []
    for images, labels in train_loader:
        train_images.append(images)
    train_images = torch.cat(train_images, dim=0)
    train_data_variance = torch.var(train_images)


    model = VQVAE(in_dim, embedding_dim, num_embeddings, train_data_variance, num_hiddens=128, num_residual_layers=2, num_residual_hiddens=32, commitment_cost=0.25, decay=0.99)
    model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), cfg.lr)

    # train VQ-VAE
    print_freq = 500

    for epoch in range(cfg.epochs):
        print("Start training epoch {}".format(epoch,))
        for i, (images, labels) in enumerate(train_loader):
            images = images - 0.5 # normalize to [-0.5, 0.5]
            images = images.cuda()
            loss = model(images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % print_freq == 0 or (i + 1) == len(train_loader):
                print("\t [{}/{}]: loss {}".format(i, len(train_loader), loss.item()))

    # get encode_indices of training images
    train_indices = []
    for images, _ in train_loader:
        images = images - 0.5 # normalize to [-0.5, 0.5]
        images = images.cuda()
        with torch.inference_mode():
            z = model.encoder(images) # [B, C, H, W]
            b, c, h, w = z.size()
            # [B, C, H, W] -> [B, H, W, C]
            z = z.permute(0, 2, 3, 1).contiguous()
            # [B, H, W, C] -> [BHW, C]
            flat_z = z.reshape(-1, c)
            encoding_indices = model.vq_layer.get_code_indices(flat_z) # bs x h x w
            encoding_indices = encoding_indices.reshape(b, h, w)    # bs, h, w
            train_indices.append(encoding_indices.cpu())

    print(f'train_indices: {len(train_indices)}')

    # train PixelCNN to generate new images
    pixelcnn = GatedPixelCNN(num_embeddings, 128, num_embeddings)
    pixelcnn = pixelcnn.cuda()

    optimizer = torch.optim.Adam(pixelcnn.parameters(), cfg.lr)

    # train pixelcnn
    print_freq = 500
    for epoch in range(cfg.epochs):
        print("Start training epoch {}".format(epoch,))
        for i, (indices) in enumerate(train_indices):
            indices = indices.cuda()
            one_hot_indices = F.one_hot(indices, num_embeddings).float().permute(0, 3, 1, 2).contiguous()
            
            outputs = pixelcnn(one_hot_indices)
            
            loss = F.cross_entropy(outputs, indices)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % print_freq == 0 or (i + 1) == len(train_loader):
                print("\t [{}/{}]: loss {}".format(i, len(train_loader), loss.item())) 

    # Create an empty array of priors.
    n_samples = 64
    priors = torch.zeros((n_samples,) + prior_size, dtype=torch.long).cuda()

    # use pixelcnn to generate priors
    pixelcnn.eval()

    # Iterate over the priors because generation has to be done sequentially pixel by pixel.
    for row in range(prior_size[0]):
        for col in range(prior_size[1]):
            # Feed the whole array and retrieving the pixel value probabilities for the next
            # pixel.
            with torch.inference_mode():
                one_hot_priors = F.one_hot(priors, num_embeddings).float().permute(0, 3, 1, 2).contiguous()
                logits = pixelcnn(one_hot_priors)
                probs = F.softmax(logits[:, :, row, col], dim=-1)
                # Use the probabilities to pick pixel values and append the values to the priors.
                priors[:, row, col] = torch.multinomial(probs, num_samples=1).squeeze(dim=-1)

    # Perform an embedding lookup and Generate new images
    with torch.inference_mode():
        z = model.vq_layer.quantize(priors)
        z = z.permute(0, 3, 1, 2).contiguous()
        pred = model.decoder(z)

    generated_samples = np.array(np.clip((pred + 0.5).cpu().numpy(), 0., 1.) * 255, dtype=np.uint8)
    generated_samples = generated_samples.reshape(8, 8, in_dim, image_size, image_size)

    fig = plt.figure(figsize=(10, 10), constrained_layout=True)
    gs = fig.add_gridspec(8, 8)
    for n_row in range(8):
        for n_col in range(8):
            f_ax = fig.add_subplot(gs[n_row, n_col])
            f_ax.imshow(generated_samples[n_row, n_col].transpose(1, 2, 0), cmap="gray")
            f_ax.axis("off")
    plt.savefig('generated_samples.jpg')

if __name__ == '__main__':
    cfg = parse_cfg()
    main(cfg)
