import torch
from new_models import GatedPixelCNN
image = torch.randn(4, 3, 224, 224)

model = GatedPixelCNN(16, 128, 16)

