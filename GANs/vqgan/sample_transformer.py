import os
import os.path as osp
import argparse
import torch
from torchvision import utils as vutils
from models.transformer import VQGANTransformer
from tqdm import tqdm


parser = argparse.ArgumentParser(description="VQGAN")
parser.add_argument('--latent-dim', type=int, default=256, help='Latent dimension n_z.')
parser.add_argument('--image-size', type=int, default=256, help='Image height and width.)')
parser.add_argument('--num-codebook-vectors', type=int, default=1024, help='Number of codebook vectors.')
parser.add_argument('--beta', type=float, default=0.25, help='Commitment loss scalar.')
parser.add_argument('--image-channels', type=int, default=3, help='Number of channels of images.')
parser.add_argument('--dataset-path', type=str, default='./data', help='Path to data.')
parser.add_argument('--output-path', type=str)
parser.add_argument('--vqgan-checkpoint-path', type=str, default='./checkpoints/last_ckpt.pt',
                    help='Path to checkpoint.')
parser.add_argument('--transformer-checkpoint-path', type=str)
parser.add_argument('--device', type=str, default="cuda", help='Which device the training is on')
parser.add_argument('--batch-size', type=int, default=20, help='Input batch size for training.')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--learning-rate', type=float, default=2.25e-05, help='Learning rate.')
parser.add_argument('--beta1', type=float, default=0.5, help='Adam beta param.')
parser.add_argument('--beta2', type=float, default=0.9, help='Adam beta param.')
parser.add_argument('--disc-start', type=int, default=10000, help='When to start the discriminator.')
parser.add_argument('--disc-factor', type=float, default=1., help='Weighting factor for the Discriminator.')
parser.add_argument('--l2-loss-factor', type=float, default=1.,
                    help='Weighting factor for reconstruction loss.')
parser.add_argument('--perceptual-loss-factor', type=float, default=1.,
                    help='Weighting factor for perceptual loss.')
parser.add_argument('--n-samples', type=int, default=100)

parser.add_argument('--pkeep', type=float, default=0.5, help='Percentage for how much latent codes to keep.')
parser.add_argument('--sos-token', type=int, default=0, help='Start of Sentence token.')

args = parser.parse_args()

transformer = VQGANTransformer(args.latent_dim, args.image_channels, args.num_codebook_vectors, args.beta, args.sos_token, args.vqgan_checkpoint_path, args.pkeep, args.device).to(args.device)
transformer.load_state_dict(torch.load(args.transformer_checkpoint_path))
print("Loaded state dict of Transformer")

os.makedirs(osp.join(args.output_path, 'sample_results'), exist_ok=True)

for i in tqdm(range(args.n_samples)):
    start_indices = torch.zeros((4, 0)).long().to(args.device)
    sos_tokens = torch.ones(start_indices.shape[0], 1) * 0
    sos_tokens = sos_tokens.long().to(args.device)
    sample_indices = transformer.sample(start_indices, sos_tokens, steps=256)
    sampled_imgs = transformer.z_to_image(sample_indices)
    vutils.save_image(sampled_imgs, os.path.join(args.output_path, 'sample_results', f"sample_result_{i}.jpg"), nrow=4)


