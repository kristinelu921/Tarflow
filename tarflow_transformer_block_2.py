import torch
import torch.nn as nn
import numpy as np
import math

from transformer_config import Config

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(device)

class LayerNorm(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.w = nn.Parameter(torch.ones(cfg.d_model))
        self.b = nn.Parameter(torch.zeros(cfg.d_model))
    
    def forward(self, x):
        # Match the official implementation
        return nn.functional.layer_norm(
            x.float(), 
            (self.cfg.d_model,), 
            self.w, 
            self.b
        ).type(x.dtype)
        
        
class PatchEmbed(nn.Module):
    """
    Transforms an image into a learnable embedding for each patch
    """
    def __init__(self, cfg):
        super().__init__()
        self.d_model = cfg.d_model
        self.img_size = cfg.img_size
        self.patch_size = cfg.patch_size
        self.n_channels = cfg.n_channels
        self.batch_size = cfg.batch_size
        self.cfg = cfg

    def add_noise(self, x, cfg):
        """Adds noise to the patches for training"""
        if hasattr(cfg, 'noise_std') and cfg.noise_std > 0:
            noise = torch.randn_like(x) * cfg.noise_std
            return x + noise
        return x

    def forward(self, img):

        B, C, H, W = img.shape
        # Use unfold for proper patching
        x = img.reshape(B, C, H // self.patch_size, self.patch_size, W // self.patch_size, self.patch_size)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.reshape(B, -1, C * self.patch_size * self.patch_size)

        # add noise if enabled
        x = self.add_noise(x, self.cfg)
        return x

    def reverse(self, patches):
        """Convert patches back to image"""
        B = patches.size(0)
        num_patches = patches.size(1)
        H = W = int(math.sqrt(num_patches))
        
        # Reshape patches to image
        patches = patches.reshape(B, H, W, self.n_channels, self.patch_size, self.patch_size)
        patches = patches.permute(0, 3, 1, 4, 2, 5).contiguous()
        img = patches.reshape(B, self.n_channels, H * self.patch_size, W * self.patch_size)
        
        return img
    
class Permutation(nn.Module):
    """
    Creates a permutation function with forward and inverse options
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
    
    def forward(self, x, dim=1, inverse=False):
        # Same operation for both forward and inverse (flip is self-inverse)
        return torch.flip(x, dims=[dim])
    
class PositionalEncoding(nn.Module):
    """
    Creates a positional encoding and optional class token for each patch
    """
    def __init__(self, cfg):
        super().__init__()
        self.d_model = cfg.d_model
        self.max_seq_length = cfg.max_seq_length
        self.pos_embed = nn.Parameter(torch.randn(1, self.max_seq_length, cfg.d_patch) * 0.01)
        

    def forward(self, embeddings, class_idx=None):
        # Add positional embeddings
        output = embeddings + self.pos_embed[:, :embeddings.size(1), :]

        return output
    
class AttentionHead(nn.Module):
    """
    Performs one attention head with caching for generation
    """
    def __init__(self, cfg):
        super().__init__()
        self.query = nn.Linear(cfg.d_model, cfg.d_head)
        self.key = nn.Linear(cfg.d_model, cfg.d_head)
        self.value = nn.Linear(cfg.d_model, cfg.d_head)
        self.cfg = cfg
        self.register_buffer("IGNORE", torch.tensor(-float('inf')))
        self.sqrt_scale = cfg.d_head ** -0.25
        
        #kv cache
        self.sample = False
        self.k_cache = []
        self.v_cache = []

    def forward(self, x, mask=None, temp=1.0):
        # Query, key, value projections
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        if self.sample:
            self.k_cache.append(k)
            self.v_cache.append(v)
            k = torch.cat(self.k_cache, dim=1)
            v = torch.cat(self.v_cache, dim=1)

        scale = (self.sqrt_scale ** 2) / temp
        attn = (q @ k.transpose(-2, -1)) * scale
        
        if mask is not None and self.cfg.mask:
            attn = self.apply_causal_mask(attn)
            
        attn = attn.softmax(dim=-1)
        return attn @ v
        
    def apply_causal_mask(self, attn_scores):
        """
        Applies a causal mask to attention scores
        """
        B, T, _ = attn_scores.size()
        mask = torch.tril(torch.ones(T, T, device=attn_scores.device))
        return attn_scores.masked_fill(mask.unsqueeze(0) == 0, self.IGNORE)

    def reset_cache(self):
        """Reset the cache for a new generation sequence"""
        self.k_cache = []
        self.v_cache = []

class MultiHeadAttention(nn.Module):
    """
    Performs multi-head attention
    """
    def __init__(self, cfg):
        super().__init__()
        self.d_model = cfg.d_model
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_head
        
        self.norm = LayerNorm(cfg)
        
        self.heads = nn.ModuleList([AttentionHead(cfg) for _ in range(self.n_heads)])
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
    
    def forward(self, x, mask=None, temp=1.0):
        x_norm = self.norm(x)
        
        head_outputs = [head(x_norm, mask, temp) for head in self.heads]
        x = torch.cat(head_outputs, dim=-1)
        
        # Final projection
        return self.proj(x)
    
    def reset_cache(self):
        """Reset cache in all heads"""
        for head in self.heads:
            head.reset_cache()
            
    def set_sample_mode(self, mode):
        for head in self.heads:
            head.sample = mode

class TransformerEncoder(nn.Module):
    """
    Performs one transformer encoder layer
    """
    def __init__(self, cfg):
        super().__init__()
        self.d_model = cfg.d_model
        self.n_heads = cfg.n_heads
        self.dropout = nn.Dropout(cfg.dropout)
        self.attn = MultiHeadAttention(cfg)
        
        # MLP with layer norm
        self.norm = LayerNorm(cfg)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model * cfg.r_mlp),
            nn.GELU(),
            nn.Linear(cfg.d_model * cfg.r_mlp, cfg.d_model)
        )
    
    def forward(self, x, mask=None, temp=1.0):
        # Attention with residual connection
        x = x + self.attn(x, mask, temp)
        
        # MLP with residual connection
        x = x + self.mlp(self.norm(x))
        
        return x
    
    def reset_cache(self):
        self.attn.reset_cache()
        
    def set_sample_mode(self, mode):
        self.attn.set_sample_mode(mode)

class AffineTransform(nn.Module):
    """
    Implements the affine coupling transform
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def forward(self, x, mu, alpha):
        new_x = x.clone()
        new_x[:, 0, :] = x[:, 0, :]
        
        alpha = torch.clamp(alpha, min=-10.0, max=10.0)
        
        new_x[:, 1:, :] = (x[:, 1:, :] - mu[:, 1:, :]) * torch.exp(-alpha[:, 1:, :])
        
        return new_x

    def inverse(self, z, mu, alpha):
        x = z.clone()
        x[:, 0, :] = z[:, 0, :]
        
        alpha = torch.clamp(alpha, min=-10.0, max=10.0)
        
        x[:, 1:, :] = z[:, 1:, :] * torch.exp(alpha[:, 1:, :]) + mu[:, 1:, :]
        
        return x
    
class TransformerFlowBlock(nn.Module):
    """
    Runs a transformer encoder that learns one flow step
    """
    def __init__(self, cfg, block_id):
        super().__init__()
        self.block_id = block_id
        
        # Set up transformers
        self.transformer_layers = nn.ModuleList([TransformerEncoder(cfg) for _ in range(cfg.n_layers)])
        
        # Projections
        self.proj_in = nn.Linear(cfg.d_patch, cfg.d_model)
        self.proj_out = nn.Linear(cfg.d_model, cfg.d_patch * 2)  # Double for mu and alpha
        
        # Zero-initialize output projection
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)
        
        self.permutation = Permutation(cfg)
        self.affine_transform = AffineTransform(cfg)
        
        self.pos_embed = nn.Parameter(torch.randn(cfg.num_patches, cfg.d_model) * 0.01)
        
        self.register_buffer('attn_mask', torch.tril(torch.ones(cfg.num_patches, cfg.num_patches)))
        
    
    def forward(self, z, y=None):
        z = self.permutation(z)
        z_orig = z.clone()
        
        z = self.proj_in(z)
        
        pos_embed = self.permutation(self.pos_embed, dim=0)
        z = z + pos_embed
        
        for layer in self.transformer_layers:
            z = layer(z, self.attn_mask)
        
        params = self.proj_out(z)
        
        # Apply causal shift (important for autoregressive property)
        params = torch.cat([torch.zeros_like(params[:, :1]), params[:, :-1]], dim=1)
        
        # Split into mu and alpha
        mu, alpha = params.chunk(2, dim=-1)
        
        z_t = self.affine_transform(z_orig, mu, alpha)
        
        return self.permutation(z_t, inverse=True), -alpha.mean(dim=[1, 2])
    
    def reverse_step(self, z, i, y=None, temp=1.0):
        """Single step of reverse process for position i"""
        z_in = z[:, :i+1].clone()
        self.set_sample_mode(True)
        
        z_in = self.proj_in(z_in)
        
        pos_embed = self.permutation(self.pos_embed, dim=0)
        z_in = z_in + pos_embed[:i+1]
        
        for layer in self.transformer_layers:
            z_in = layer(z_in, temp=temp)
        
        params = self.proj_out(z_in[:, -1:])
        
        mu, alpha = params.chunk(2, dim=-1)
        
        return mu.squeeze(1), alpha.squeeze(1)
    
    def set_sample_mode(self, mode):
        for layer in self.transformer_layers:
            layer.set_sample_mode(mode)
            
    def reset_cache(self):
        for layer in self.transformer_layers:
            layer.reset_cache()
        
class Tarflow(nn.Module):
    """
    Puts together all flow steps + transformer architecture
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.patch_embedding = PatchEmbed(cfg)
        self.flow_blocks = nn.ModuleList([TransformerFlowBlock(cfg, block_id=i) for i in range(cfg.n_flow_steps)])
        
        # Add learnable prior variance (important for generation quality)
        self.register_buffer('var', torch.ones(cfg.num_patches, cfg.d_patch))
        self.var_lr = 0.1  # Learning rate for variance updates
        
    def encode(self, images, y=None):
        """Encode images to latent space"""
        alphas = []
        log_dets = []

        # Convert images to patches
        x = self.patch_embedding(images)
        
        # Pass through flow blocks
        for block in self.flow_blocks:
            x, alpha = block(x, y)
            alphas.append(alpha)
            # Sum over all dimensions except batch
            log_det = alpha
            log_dets.append(log_det)

        return x, alphas, log_dets
    
    def loss(self, z, log_dets):
        """Flow loss: prior + sum of logdets"""
        # Prior loss (negative log likelihood)
        prior_loss = 0.5 * z.pow(2).sum(dim=[1, 2])  # Sum over all dimensions except batch
        
        # Flow loss (negative log determinant)
        flow_loss = torch.zeros_like(prior_loss)
        for log_det in log_dets:
            flow_loss -= log_det
        
        # Total loss
        total_loss = prior_loss + flow_loss
        
        return total_loss.mean()
    
    def update_prior(self, z):
        """Update the prior variance with moving average"""
        z2 = (z**2).mean(dim=0)
        self.var.lerp_(z2.detach(), weight=self.var_lr)
    
    def decode(self, z, y=None, guidance=0.0, temp=1.0):
        """Decode latent to image space, with optional guidance"""
        for block in self.flow_blocks:
            block.reset_cache()
        
        z = z * self.var.sqrt()
        
        for block_idx in reversed(range(len(self.flow_blocks))):
            block = self.flow_blocks[block_idx]
            
            z = block.permutation(z)
            
            for i in range(z.size(1) - 1): 
                mu, alpha = block.reverse_step(z, i, y, temp)
                
                scale = torch.exp(alpha)
                z[:, i+1] = z[:, i+1] * scale + mu
            
            z = block.permutation(z, inverse=True)
        
        return self.patch_embedding.reverse(z)
    
    def sample(self, batch_size=1, y=None, guidance=0.0, temp=1.0):
        """Sample from the model"""
        z = torch.randn(batch_size, self.cfg.num_patches, self.cfg.d_patch, device=device)
        
        return self.decode(z, y, guidance, temp)