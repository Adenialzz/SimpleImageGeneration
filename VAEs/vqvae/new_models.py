import torch
import torch.nn as nn
import torch.nn.functional as F

class ExponentialMovingAverage(nn.Module):
    """Maintains an exponential moving average for a value.
    
      This module keeps track of a hidden exponential moving average that is
      initialized as a vector of zeros which is then normalized to give the average.
      This gives us a moving average which isn't biased towards either zero or the
      initial value. Reference (https://arxiv.org/pdf/1412.6980.pdf)
      
      Initially:
          hidden_0 = 0
      Then iteratively:
          hidden_i = hidden_{i-1} - (hidden_{i-1} - value) * (1 - decay)
          average_i = hidden_i / (1 - decay^i)
    """
    
    def __init__(self, init_value, decay):
        super().__init__()
        
        self.decay = decay
        self.counter = 0
        self.register_buffer("hidden", torch.zeros_like(init_value))
        
    def forward(self, value):
        self.counter += 1
        self.hidden.sub_((self.hidden - value) * (1 - self.decay))
        average = self.hidden / (1 - self.decay ** self.counter)
        return average
        
    
class VectorQuantizerEMA(nn.Module):
    """
    VQ-VAE layer: Input any tensor to be quantized. Use EMA to update embeddings.
    Args:
        embedding_dim (int): the dimensionality of the tensors in the
          quantized space. Inputs to the modules must be in this format as well.
        num_embeddings (int): the number of vectors in the quantized space.
        commitment_cost (float): scalar which controls the weighting of the loss terms (see
          equation 4 in the paper - this variable is Beta).
        decay (float): decay for the moving averages.
        epsilon (float): small float constant to avoid numerical instability.
    """
    def __init__(self, embedding_dim, num_embeddings, commitment_cost, decay,
               epsilon=1e-5):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.epsilon = epsilon
        
        # initialize embeddings as buffers
        embeddings = torch.empty(self.num_embeddings, self.embedding_dim)
        nn.init.xavier_uniform_(embeddings)
        self.register_buffer("embeddings", embeddings)
        self.ema_dw = ExponentialMovingAverage(self.embeddings, decay)
        
        # also maintain ema_cluster_size， which record the size of each embedding
        self.ema_cluster_size = ExponentialMovingAverage(torch.zeros((self.num_embeddings,)), decay)
        
        
    def forward(self, x):
        # [B, C, H, W] -> [B, H, W, C]
        x = x.permute(0, 2, 3, 1).contiguous()
        # [B, H, W, C] -> [BHW, C]
        flat_x = x.reshape(-1, self.embedding_dim)
        
        encoding_indices = self.get_code_indices(flat_x)
        quantized = self.quantize(encoding_indices)
        quantized = quantized.view_as(x) # [B, H, W, C]
        
        if not self.training:
            quantized = quantized.permute(0, 3, 1, 2).contiguous()
            return quantized
        
        # update embeddings with EMA
        with torch.no_grad():
            encodings = F.one_hot(encoding_indices, self.num_embeddings).float()
            updated_ema_cluster_size = self.ema_cluster_size(torch.sum(encodings, dim=0))
            n = torch.sum(updated_ema_cluster_size)
            updated_ema_cluster_size = ((updated_ema_cluster_size + self.epsilon) /
                                      (n + self.num_embeddings * self.epsilon) * n)
            dw = torch.matmul(encodings.t(), flat_x) # sum encoding vectors of each cluster
            updated_ema_dw = self.ema_dw(dw)
            normalised_updated_ema_w = (
              updated_ema_dw / updated_ema_cluster_size.reshape(-1, 1))
            self.embeddings.data = normalised_updated_ema_w
        
        # commitment loss
        e_latent_loss = F.mse_loss(x, quantized.detach())
        loss = self.commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = x + (quantized - x).detach()
        
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        return quantized, loss
    
    def get_code_indices(self, flat_x):
        # compute L2 distance
        distances = (
            torch.sum(flat_x ** 2, dim=1, keepdim=True) +
            torch.sum(self.embeddings ** 2, dim=1) -
            2. * torch.matmul(flat_x, self.embeddings.t())
        ) # [N, M]
        encoding_indices = torch.argmin(distances, dim=1) # [N,]
        return encoding_indices
    
    def quantize(self, encoding_indices):
        """Returns embedding tensor for a batch of indices."""
        return F.embedding(encoding_indices, self.embeddings)


class ResidualStack(nn.Module):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()
        self._num_hiddens = num_hiddens
        self._num_residual_layers = num_residual_layers
        self._num_residual_hiddens = num_residual_hiddens

        self._layers = nn.ModuleList()
        for i in range(num_residual_layers):
            self._layers.append(nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(num_hiddens, num_residual_hiddens, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(num_residual_hiddens, num_hiddens, 3, padding=1),
            ))
                    
    def forward(self, x):
        h = x
        for layer in self._layers:
            conv = layer(h)
            h = h + conv
        return F.relu(h)


class Encoder(nn.Module):
    def __init__(self, in_dim, embedding_dim, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Encoder, self).__init__()
        self._num_hiddens = num_hiddens
        self._num_residual_layers = num_residual_layers
        self._num_residual_hiddens = num_residual_hiddens

        self._enc_1 = nn.Conv2d(in_dim, self._num_hiddens // 2, 4, stride=2, padding=1)
        self._enc_2 = nn.Conv2d(self._num_hiddens // 2, self._num_hiddens, 4, stride=2, padding=1)
        self._enc_3 = nn.Conv2d(self._num_hiddens, self._num_hiddens, 3, stride=1, padding=1)
        self._residual_stack = ResidualStack(self._num_hiddens, self._num_residual_layers,
                                             self._num_residual_hiddens)
        self.pre_vq_conv = nn.Conv2d(self._num_hiddens, embedding_dim, 1)
        
    def forward(self, x):
        h = F.relu(self._enc_1(x))
        h = F.relu(self._enc_2(h))
        h = F.relu(self._enc_3(h))
        h = self._residual_stack(h)
        h = self.pre_vq_conv(h)
        return h


class Decoder(nn.Module):
    def __init__(self, out_dim, embedding_dim, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Decoder, self).__init__()
        self._num_hiddens = num_hiddens
        self._num_residual_layers = num_residual_layers
        self._num_residual_hiddens = num_residual_hiddens

        self._dec_1 = nn.Conv2d(embedding_dim, self._num_hiddens, 3, stride=1, padding=1)
        self._residual_stack = ResidualStack(self._num_hiddens, self._num_residual_layers, self._num_residual_hiddens)
        self._dec_2 = nn.ConvTranspose2d(self._num_hiddens, self._num_hiddens // 2, 4, stride=2, padding=1)
        self._dec_3 = nn.ConvTranspose2d(self._num_hiddens // 2, out_dim, 4, stride=2, padding=1)
        
    def forward(self, x):
        h = self._dec_1(x)
        h = self._residual_stack(h)
        h = F.relu(self._dec_2(h))
        recon = self._dec_3(h)
        return recon

class VQVAE(nn.Module):
    """VQ-VAE"""
    
    def __init__(self, in_dim, embedding_dim, num_embeddings, data_variance, 
                 num_hiddens, num_residual_layers, num_residual_hiddens,
                 commitment_cost=0.25, decay=0.99):
        super().__init__()
        self.in_dim = in_dim
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.data_variance = data_variance
        
        self.encoder = Encoder(in_dim, embedding_dim, num_hiddens, num_residual_layers, num_residual_hiddens)
        self.vq_layer = VectorQuantizerEMA(embedding_dim, num_embeddings, commitment_cost, decay)
        self.decoder = Decoder(in_dim, embedding_dim, num_hiddens, num_residual_layers, num_residual_hiddens)
        
    def forward(self, x):
        z = self.encoder(x)
        if not self.training:
            e = self.vq_layer(z)
            x_recon = self.decoder(e)
            return e, x_recon
        
        e, e_q_loss = self.vq_layer(z)
        x_recon = self.decoder(e)
        
        recon_loss = F.mse_loss(x_recon, x) / self.data_variance
        
        return e_q_loss + recon_loss

class MaskedConv2d(nn.Conv2d):
    """
    Implements a conv2d with mask applied on its weights.
    
    Args:
        mask (torch.Tensor): the mask tensor.
        in_channels (int) – Number of channels in the input image.
        out_channels (int) – Number of channels produced by the convolution.
        kernel_size (int or tuple) – Size of the convolving kernel
    """
    
    def __init__(self, mask, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        self.register_buffer('mask', mask[None, None])
        
    def forward(self, x):
        self.weight.data *= self.mask # mask weights
        return super().forward(x)
    

class VerticalStackConv(MaskedConv2d):

    def __init__(self, mask_type, in_channels, out_channels, kernel_size, **kwargs):
        # Mask out all pixels below. For efficiency, we could also reduce the kernel
        # size in height (k//2, k), but for simplicity, we stick with masking here.
        self.mask_type = mask_type
        
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        mask = torch.zeros(kernel_size)
        mask[:kernel_size[0]//2, :] = 1.0
        if self.mask_type == "B":
            mask[kernel_size[0]//2, :] = 1.0

        super().__init__(mask, in_channels, out_channels, kernel_size, **kwargs)
        

class HorizontalStackConv(MaskedConv2d):

    def __init__(self, mask_type, in_channels, out_channels, kernel_size, **kwargs):
        # Mask out all pixels on the left. Note that our kernel has a size of 1
        # in height because we only look at the pixel in the same row.
        self.mask_type = mask_type
        
        if isinstance(kernel_size, int):
            kernel_size = (1, kernel_size)
        assert kernel_size[0] == 1
        if "padding" in kwargs:
            if isinstance(kwargs["padding"], int):
                kwargs["padding"] = (0, kwargs["padding"])
        
        mask = torch.zeros(kernel_size)
        mask[:, :kernel_size[1]//2] = 1.0
        if self.mask_type == "B":
            mask[:, kernel_size[1]//2] = 1.0

        super().__init__(mask, in_channels, out_channels, kernel_size, **kwargs)
        
class GatedMaskedConv(nn.Module):

    def __init__(self, in_channels, kernel_size=3, dilation=1):
        """
        Gated Convolution block implemented the computation graph shown above.
        """
        super().__init__()
        
        padding = dilation * (kernel_size - 1) // 2
        self.conv_vert = VerticalStackConv("B", in_channels, 2*in_channels, kernel_size, padding=padding,
                                          dilation=dilation)
        self.conv_horiz = HorizontalStackConv("B", in_channels, 2*in_channels, kernel_size, padding=padding,
                                             dilation=dilation)
        self.conv_vert_to_horiz = nn.Conv2d(2*in_channels, 2*in_channels, kernel_size=1)
        self.conv_horiz_1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, v_stack, h_stack):
        # Vertical stack (left)
        v_stack_feat = self.conv_vert(v_stack)
        v_val, v_gate = v_stack_feat.chunk(2, dim=1)
        v_stack_out = torch.tanh(v_val) * torch.sigmoid(v_gate)

        # Horizontal stack (right)
        h_stack_feat = self.conv_horiz(h_stack)
        h_stack_feat = h_stack_feat + self.conv_vert_to_horiz(v_stack_feat)
        h_val, h_gate = h_stack_feat.chunk(2, dim=1)
        h_stack_feat = torch.tanh(h_val) * torch.sigmoid(h_gate)
        h_stack_out = self.conv_horiz_1x1(h_stack_feat)
        h_stack_out = h_stack_out + h_stack

        return v_stack_out, h_stack_out
    
    
class GatedPixelCNN(nn.Module):
    
    def __init__(self, in_channels, channels, out_channels):
        super().__init__()
        
        # Initial first conv with mask_type A
        self.conv_vstack = VerticalStackConv("A", in_channels, channels, 3, padding=1)
        self.conv_hstack = HorizontalStackConv("A", in_channels, channels, 3, padding=1)
        # Convolution block of PixelCNN. use dilation instead of 
        # downscaling used in the encoder-decoder architecture in PixelCNN++
        self.conv_layers = nn.ModuleList([
            GatedMaskedConv(channels),
            GatedMaskedConv(channels, dilation=2),
            GatedMaskedConv(channels)
        ])
        
        # Output classification convolution (1x1)
        self.conv_out = nn.Conv2d(channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        # first convolutions
        v_stack = self.conv_vstack(x)
        h_stack = self.conv_hstack(x)
        # Gated Convolutions
        for layer in self.conv_layers:
            v_stack, h_stack = layer(v_stack, h_stack)
        # 1x1 classification convolution
        # Apply ELU before 1x1 convolution for non-linearity on residual connection
        out = self.conv_out(F.elu(h_stack))
        return out
