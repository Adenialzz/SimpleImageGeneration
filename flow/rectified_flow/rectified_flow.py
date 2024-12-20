import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from dit import DiT_Llama
from typing import Union
import numpy as np
from PIL import Image

from tqdm import tqdm

def sample_t(batch_size: int) -> torch.Tensor:
    return torch.rand((batch_size, ))

def sample_t_ln(batch_size: int) -> torch.Tensor:
    nt = torch.randn((batch_size,))
    t = torch.sigmoid(nt)
    return t

def get_model_device(model: torch.nn.Module):
    return next(model.parameters()).device

def post_process(images: torch.Tensor):
    # images = images * 0.5 + 0.5
    images = images.clamp(0, 1)
    images = images.permute(0, 2, 3, 1).cpu().numpy()
    images= (images * 255).astype(np.uint8)
    pil_images = []
    for img in images:
        pil_images.append(Image.fromarray(img))
    return pil_images

@torch.no_grad()
def sample_class(
    model: torch.nn.Module,
    classes: Union[torch.Tensor, int],
    num_samples: int,
    num_steps: int,
    guidance_scale: float  # TODO
):

    model.eval()
    device = get_model_device(model)
    with torch.no_grad():

        if isinstance(classes, int):
            classes = torch.ones(num_samples) * classes
            classes = classes.to(torch.int64)

        classes = classes.to(device) 
        x = torch.randn(num_samples, 3, 32, 32).to(device)

        dt = 1.0 / num_steps
        dt = torch.tensor([dt] * num_samples).view([num_samples, *([1] * len(x.shape[1:]))]).to(classes.device)
        pbar = tqdm(range(num_steps, 0, -1), desc='sampling')
        for i in pbar:
            pbar.set_description(f"sampling {num_steps - i + 1} / {num_steps}")

            t = torch.tensor([i / num_steps] * num_samples).to(device)
            v_pred = model(x, t, classes)
            x = x - v_pred * dt

        return x



def main():

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    device = 'cuda'
    model = DiT_Llama(
        3, 32, dim=256, n_layers=10, n_heads=8, num_classes=10
    ).to(device)

    epochs = 10

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    mnist = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    dataloader = DataLoader(mnist, batch_size=256, shuffle=True, drop_last=True)

    mse_loss_func = torch.nn.MSELoss()
    for ep in range(epochs):
        model.train()
        pbar = tqdm(dataloader, desc='training')
        for x0, c in pbar:
            x0, c = x0.to(device), c.to(device)
            bs = x0.size(0)
            optimizer.zero_grad()

            # t = sample_t(bs).to(device)
            t = sample_t_ln(bs).to(device)

            texp = t.view([bs, *([1] * len(x0.shape[1:]))])


            x1 = torch.randn_like(x0).to(device)
            xt = (1 - texp) * x0 + texp * x1
            v_pred = model(xt, t, c)

            loss = mse_loss_func(v_pred, x1 - x0)

            loss.backward()
            optimizer.step()
            pbar.set_description(f'training, loss = {round(loss.item(), 3)}')

            # break

        
        model.eval()
        x = sample_class(model, 1, num_samples=4, num_steps=50, guidance_scale=3.0)
        model.train()

        images = post_process(x)
        for i, img in enumerate(images):
            img.save(f'img_ep{ep}_{i}.png')
        
if __name__ == '__main__':
    main()


