import torch
import torch.nn as nn
import numpy as np

from transformer_config import Config as Config

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

class LayerNorm(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.w = nn.Parameter(torch.ones(cfg.d_model))
        self.b = nn.Parameter(torch.zeros(cfg.d_model))
    
    def forward(self, residual):
        residual_mean = residual.mean(dim = -1, keepdim = True)
        residual_std = (residual.var(dim = -1, keepdim = True, unbiased = False) + self.cfg.layer_norm_eps).sqrt()

        residual = (residual - residual_mean) / residual_std
        return residual * self.w + self.b
        
class PatchEmbed(nn.Module):
    """
    Input: Image: float[Tensor, (bsize, channels, height, width)]
    Output: Embedding: float[Tensor, (bsize, flattened_patch, d_model)]

    Transforms an image into a learnable embedding (d_model dimensions) for each patch
    
    Section 2.4: Reshape image to patches
    B x C x H x W -> B x (HW/P_size^2) x (P_size^2 x C)

    Paper doesn't give an invertible way to linear project the patches to the d_model dimension, so in this implementation we use an invertible linear projection

    """
    def __init__(self, cfg: Config):

        super().__init__()
        self.d_model = cfg.d_model #dim of each patch embedding (EG: 768 for a 768-dim vector)                 
        self.img_size = cfg.img_size #size of input (h, w) (EG: 224 for a 224 x 224 image)
        self.patch_size = cfg.patch_size #size of each patch (EG: 16 for a 16 x 16 patch)
        self.n_channels = cfg.n_channels #number of channels (EG: 3 for RGB)
        self.batch_size = cfg.batch_size
        self.cfg = cfg

    def add_noise(self, images, cfg):
        """
        Adds noise to the images for training
        images: (bsize, channels, height, width)
        cfg: transformer config
        std: standard dev of the noise
        """
        std = cfg.noise_std
        noise = torch.randn_like(images) * std
        noisy_images = images + noise
        return noisy_images

    def forward(self, img):
        img = torch.reshape(img, (img.size(0), self.cfg.num_patches, self.cfg.d_patch))
        img = self.add_noise(img, self.cfg)
        print("img", img[0:1])
        return img

    def reverse(self, z):
        z = torch.reshape(z, (z.size(0), self.n_channels, self.img_size, self.img_size))
        return z
    
class PositionalEncoding(nn.Module):
    """
    Input: Embedding: (bsize, flattened_patch, d_model)
    Output: Pos + Class Encoded Embedding
    Creates a positional encoding and class token for each patch
    """
    def __init__(self, cfg: Config):
        super().__init__()
        self.d_model = cfg.d_model
        self.max_seq_length = cfg.max_seq_length
        #self.cls_token = nn.Embedding(torch.randn(1, 1, cfg.d_patch)) #classification token, "summary vector"
        self.pos_embed = nn.Parameter(torch.randn(1, self.max_seq_length, cfg.d_patch))
        self.use_cls_token = cfg.guidance_on

    def forward(self, embeddings, class_idx = None): #embeddings is Float[Tensor, "bsize patch dmodel"]
        embeddings = embeddings + self.pos_embed[:, :embeddings.size(1), :]
        #if self.use_cls_token and class_idx is not None:
        #    class_tkn = self.class_tkn(class_idx)
        #    class_tkn = class_tkn.unsqueeze(1).expand(-1, embeddings.size(1), -1)
        #    embeddings = embeddings + class_tkn
        
        return embeddings

    def reverse(self, embeddings): #embeddings is Float[Tensor, "bsize patch + 1 dmodel"]
        embeddings = embeddings - self.pos_embed[:, :embeddings.size(1), :]
        #if self.use_cls_token:
        #    embeddings = embeddings - self.current_class_tkn
        return embeddings
    
class AttentionHead(nn.Module):
    """
    Input: Embeddings: (bsize patch dmodel)
    Output: Attention output: (bsize patch dmodel)
    Performs one attention head
    """
    def __init__(self, cfg: Config):
        super().__init__()
        
        self.query = nn.Linear(cfg.d_model, cfg.d_head)
        self.key = nn.Linear(cfg.d_model, cfg.d_head)
        self.value = nn.Linear(cfg.d_model, cfg.d_head)
        self.output = nn.Linear(cfg.d_head, cfg.d_model)
        self.cfg = cfg
        self.register_buffer("IGNORE", torch.tensor(-float('inf')))
        self.temp  = 1.0 #guidance in 2.6

    def forward(self, embeddings, temp = None):  #bsize patch dmodel (embeddings)
        """
        Takes in embeddings: (bsize patch dmodel)
        """

        temp = temp if temp is not None else self.temp

        # Calculate query, key and value vectors
        Q = self.query(embeddings)  #bsize patch dmodel -> bsize patch dhead
        K = self.key(embeddings) #bsize patch dmodel -> bsize patch dhead
        V = self.value(embeddings) #bsize patch dmodel -> bsize patch dhead

        # Calculate attention scores, then scale and mask, and apply softmax to get probabilities
        attn_scores = Q @ K.transpose(-1, -2) # -> bsize patch_q patch_k
        attn_scores_scaled = attn_scores / self.cfg.d_head**0.5

        if self.cfg.mask:
            attn_scores_masked = self.apply_causal_mask(attn_scores_scaled) #scaled
            attn_pattern = attn_scores_masked.softmax(-1) #softmaxed #bsize patch_q patch_k
        else:
            attn_pattern = attn_scores.softmax(-1)

        attn_out = attn_pattern @ V #bsize patch_q dhead

        return attn_out

    def apply_causal_mask(self, attn_scores):
        """
        Applies a causal mask to attention scores, and returns masked scores.
        """
        # Define a mask that is True for all positions we want to set probabilities to zero for
        all_ones = torch.ones(attn_scores.size(-2), attn_scores.size(-1), device=attn_scores.device)
        mask = torch.triu(all_ones, diagonal=1).bool()
        # Apply the mask to attention scores, then return the masked scores
        attn_scores.masked_fill_(mask, self.IGNORE) #IGNORE is -inf
        return attn_scores


class MultiHeadAttention(nn.Module):
    """
    Input: Embeddings: (bsize patch dmodel)
    Output: Attention output: (bsize patch dmodel)
    Performs multi-head attention
    """
    def __init__(self, cfg):
        super().__init__()
        self.d_model = cfg.d_model
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_head

        self.W_o = nn.Linear(self.d_model, self.d_model)

        #pass each through one attn head to get attn scores
        self.heads = nn.ModuleList([AttentionHead(cfg) for _ in range(self.n_heads)]) 
    
    def forward(self, embeddings): #B, patches, d_model
        out = torch.cat([head(embeddings) for head in self.heads], dim = -1)
        out = self.W_o(out) #B, patches, d_model
        return out

class TransformerEncoder(nn.Module):
    """
    Input: Embeddings: (bsize patch dmodel)
    Output: Encoded Embeddings: (bsize patch dmodel)
    Performs one transformer encoder layer
    """
    def __init__(self, cfg: Config):
        super().__init__()
        self.d_model = cfg.d_model
        self.n_heads = cfg.n_heads
        self.dropout = nn.Dropout(cfg.dropout)
        self.ln1 = LayerNorm(cfg)
        self.mha = MultiHeadAttention(cfg)
        self.ln2 = LayerNorm(cfg) 
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model * cfg.r_mlp),
            nn.GELU(),
            nn.Linear(cfg.d_model*cfg.r_mlp, cfg.d_model)
        )
    
    def forward(self, embeddings):
        out = embeddings + self.mha(self.ln1(embeddings))
        #out = self.dropout(out)
        out = out + self.mlp(self.ln2(out))
        return out

class Permutation(nn.Module): #post patch embedding
    """
    Creates the permutation function (reversal) following p.3 in paper
    """
    def __init__(self, cfg: Config): #batch_size, num_patches, d_model
        super().__init__()
        self.cfg = cfg

    def forward(self, x): #batch_size, num_patches, d_model
        permuted = torch.flip(x, dims = [1])
        return permuted

    def reverse(self, x): #batch_size, num_patches, d_model
        permuted = torch.flip(x, dims = [1])
        return permuted

class AffineTransform(nn.Module):
    """
    Implements the affine transform (eq. 3) in the paper

    Input: x: (bsize, num_patches, d_model)
    mu: (learned linear, gelu, linear)
    alpha: (learned linear, gelu, linear )
    output: newx: (bsize, num_patches, d_model)
    """
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

    def forward(self, x, mu, alpha): #batch_size, num_patches, d_model
        new_x = torch.zeros_like(x)

        new_x[:, 0, :] = x[:, 0, :] #batch_size, first_patch, d_model
    #TODO: check if clamp is needed
        alpha = torch.clamp(alpha, min=-10.0, max=10.0)
        new_x[:, 1:, :] = (x[:, 1:, :] - mu[:, 1:, :]) * torch.exp(-alpha[:, 1:, :]) #batch_size, num_patches - 1, d_model
        return new_x

    def inverse(self, z, mu, alpha):
        z_prev = torch.zeros_like(z)

        z_prev[:, 0, :] = z[:, 0, :]
    #TODO: check if clamp is needed
        alpha = torch.clamp(alpha, min=-10.0, max=10.0)
        z_prev[:, 1:, :] = z[:, 1:, :] * torch.exp(alpha[:, 1:, :]) + mu[:, 1:, :] #batch_size, num_patches - 1, d_model
        return z_prev
    
class TransformerFlowBlock(nn.Module):
    """
    Runs a transformer encoder that learns one flow step, then applies the affine transform
    Follows flow step in eq. 3 in paper

    Input: Images: (bsize, numpatches, d patch)
    Output: Transformed Embeddings: (bsize, num_patches, d_patch)
    """
    def __init__(self, cfg, block_id):
        super().__init__()
        self.block_id = block_id
        cfg.mask = True


        assert cfg.img_size % cfg.patch_size == 0  #assume working with square patches
        assert cfg.d_model % cfg.n_heads == 0

        self.transformer_encoder = nn.ModuleList([TransformerEncoder(cfg) for _ in range(cfg.n_layers)])
        self.proj_to_model = nn.Linear(cfg.d_patch, cfg.d_model)
        self.proj_to_patch = nn.Linear(cfg.d_model, cfg.d_patch)

        self.mu_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model)
        )

        self.alpha_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model)
        )

        self.permutation = Permutation(cfg)
        self.affine_transform = AffineTransform(cfg)
        self.pos_embed = PositionalEncoding(cfg)

        nn.init.zeros_(self.mu_head[-1].weight)
        nn.init.zeros_(self.mu_head[-1].bias)
        nn.init.zeros_(self.alpha_head[-1].weight)
        nn.init.zeros_(self.alpha_head[-1].bias)

        #nn.init.eye_(self.proj_to_model.weight)
        #nn.init.zeros_(self.proj_to_model.bias)
        #nn.init.eye_(self.proj_to_patch.weight)
        #nn.init.zeros_(self.proj_to_patch.bias)
#TODO: add residual connection 

    def forward(self, z_t, temp = None, uncond_out = None): #batch_size, num_patches, d_model
        #print(images)
        z_t = self.permutation(z_t)
        z_orig = z_t
        z_t_proj = self.proj_to_model(z_t) + self.permutation(self.pos_embed(z_t))
        
        encoded = z_t_proj
        for layer in self.transformer_encoder:
            encoded = layer(encoded)

        z_t = self.proj_to_patch(encoded)
        z_t = torch.cat([torch.zeros_like(z_t[:, :1]), z_t[:, :-1]], dim = 1)

        mu, alpha = z_t.chunk(2, dim = -1)

        z_t1 = self.affine_transform(z_t, mu, alpha)
        return self.permutation(z_t1), -alpha.mean(dim = [1, 2])

    def reverse(self, z_t, i): #i is the ith-patch
        z_patch = z_t[:, i: i+1]
        z_patch_proj = self.proj_to_model(z_patch) + self.permutation(self.pos_embed(z_patch))
        encoded = z_patch_proj
        
class Tarflow(nn.Module):
    """
    Puts together all flow steps + transformer architecture
    Following figure 2 in paper

    Input: Images: (bsize, channels, height, width)
    Output: latent space image: (bsize, num_patches, channels * height * width)
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.patch_embedding = PatchEmbed(cfg)
        self.positional_encoding = PositionalEncoding(cfg)
        self.transformer_flow_blocks = nn.ModuleList([TransformerFlowBlock(cfg, block_id = i) for i in range(cfg.n_flow_steps)])
        
    def encode(self, images):
        alphas = []
        log_dets = []

        x = self.patch_embedding(images)
        x = self.positional_encoding(x)
        
        for i in range(len(self.transformer_flow_blocks)):
            block = self.transformer_flow_blocks[i]
            x, alpha = block(x)
            alphas.append(alpha)
            log_det = -torch.sum(alpha[:, 1:, :], dim = (1, 2))
            log_dets.append(log_det) #batch size

        return x, alphas, log_dets
    
    def loss(self, x, alphas, log_dets):
        """
        Following loss function (eq. 6) in the paper,
        L = 0.5 * ||x||^2 + sum of alphas
        """
        prior_loss = 0.5 * torch.sum(x.pow(2), dim=(1, 2))  # batch size

        flow_loss = torch.zeros_like(prior_loss)
        for log_det in log_dets:
            flow_loss -= log_det

        total_loss = prior_loss + flow_loss

        return total_loss.mean()

    def decode(self, z, temp=1.0):
        # Reverse order of blocks
        for block_idx in reversed(range(len(self.transformer_flow_blocks))):
            block = self.transformer_flow_blocks[block_idx]
            
            # Project to model dimension
            z_model = block.proj_to_model(z)
            
            # Apply permutation if needed
            if block_idx > 0:
                z_perm = block.permutation(z_model)
            else:
                z_perm = z_model
                
            # Run through transformer layers sequentially
            encoded = z_perm
            for layer in block.transformer_encoder:
                encoded = layer(encoded)
            
            # Generate parameters
            mu = block.mu_head(encoded)
            alpha = block.alpha_head(encoded)
            
            # Apply inverse affine transform
            z_inv = block.affine_transform.inverse(z_perm, mu, alpha)
            
            # Apply inverse permutation if needed
            if block_idx > 0:
                z_inv = block.permutation(z_inv)
                
            # Project back to patch dimension
            z = block.proj_to_patch(z_inv)
        
        # Reshape patches to image
        image = self.patch_embedding.reverse(z)
        
        return image
    
from train_tarflow import train_model

if __name__ == "__main__":
    train_model(Tarflow(Config), Config)