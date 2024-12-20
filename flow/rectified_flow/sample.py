import torch
from rectified_flow import sample_class, post_process

from dit import DiT_Llama

if __name__ == '__main__':
    device = 'cuda'

    model = DiT_Llama( 3, 32, dim=256, n_layers=10, n_heads=8, num_classes=10).to(device)

    sd = torch.load('contents1/model_4.pth', map_location='cpu')
    model.load_state_dict(sd)

    imgs = sample_class(model, classes=1, num_samples=4, num_steps=50, guidance_scale=None)
    imgs = post_process(imgs)

    for i, img in enumerate(imgs):
        img.save(f'cmp_{i}.png')


