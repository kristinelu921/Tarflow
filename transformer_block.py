import torch
import torch.nn as nn
import numpy as np

from transformer_config import Config

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
    """
    def __init__(self, cfg: Config):

        super().__init__()
        self.d_model = cfg.d_model #dim of each patch embedding (EG: 768 for a 768-dim vector)
        self.img_size = cfg.img_size #size of input (h, w) (EG: 224 for a 224 x 224 image)
        self.patch_size = cfg.patch_size #size of each patch (EG: 16 for a 16 x 16 patch)
        self.n_channels = cfg.n_channels #number of channels (EG: 3 for RGB)

        self.linear_project = nn.Conv2d(self.n_channels, self.d_model, kernel_size = self.patch_size, stride = self.patch_size)

    def forward(self, img):
        img = self.linear_project(img) #(B, C, H, W) -> (B, d_model, P_col, P_row)
        img  = img.flatten(2) #(B, d_model, P_col, P_row) -> (B, d_model, npatch)
        img = img.transpose(1, 2) #(B, d_model, P) -> (B, P, d_model)
        return img

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
        self.cls_token = nn.Parameter(torch.randn(1, 1, cfg.d_model)) #classification token, "summary vector"

        pe = torch.zeros(self.max_seq_length, self.d_model)
        #print(pe.shape)

        for pos in range(self.max_seq_length):
            for i in range(self.d_model):
                if i % 2 == 0:
                    pe[pos][i] = np.sin(pos/(10000** (i/self.d_model)))
                else:
                    pe[pos][i] = np.cos(pos/(10000 ** ((i-1)/self.d_model)))
        
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, embeddings): #embeddings is Float[Tensor, "bsize patch dmodel"]
        #print(embeddings)
        tokens_batch = self.cls_token.expand(embeddings.shape[0], -1, -1) #expands class token to match batch size (bsize, 1, d_model)
        embeddings = torch.cat((tokens_batch, embeddings), dim = 1) #Adding class tokens to each embedding #(bsize, npatch + 1, d_model)
        embeddings = embeddings + self.pe #adds a tiny vector to each existing parameter, marks out a position
        return embeddings
        #?? It just gets added in, how does model differentiate which part is from position and which is the actual weights?

class AttentionHead(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        
        self.query = nn.Linear(cfg.d_model, cfg.d_head)
        self.key = nn.Linear(cfg.d_model, cfg.d_head)
        self.value = nn.Linear(cfg.d_model, cfg.d_head)
        self.output = nn.Linear(cfg.d_head, cfg.d_model)
        self.cfg = cfg
        self.register_buffer("IGNORE", torch.tensor(-float('inf')))

    def forward(self, normalized_resid_pre):  #bsize patch dmodel (embeddings)
        """
        Takes in embeddings: (bsize patch dmodel)
        """
        # Calculate query, key and value vectors
        Q = self.query(normalized_resid_pre)  #bsize patch dmodel -> bsize patch dhead
        K = self.key(normalized_resid_pre) #bsize patch dmodel -> bsize patch dhead
        V = self.value(normalized_resid_pre) #bsize patch dmodel -> bsize patch dhead

        # Calculate attention scores, then scale and mask, and apply softmax to get probabilities
        attn_scores = Q @ K.transpose(-1, -2) # -> bsize patch_q patch_k
        attn_scores_scaled = attn_scores / self.cfg.d_head**0.5
        attn_scores_masked = self.apply_causal_mask(attn_scores_scaled) #scaled

        if self.cfg.mask:
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
        out = self.dropout(out)
        out = out + self.mlp(self.ln2(out))
        return out

class VisionTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.img_size = cfg.img_size
        self.patch_size = cfg.patch_size
        self.n_channels = cfg.n_channels
        self.max_seq_length = cfg.max_seq_length
        self.d_model = cfg.d_model
        self.layer_norm_eps = cfg.layer_norm_eps
        self.r_mlp = cfg.r_mlp
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_head
        self.n_layers = cfg.n_layers
        self.n_classes = cfg.n_classes

        assert self.img_size % self.patch_size == 0  #assume working with square patches
        assert self.d_model % self.n_heads == 0

        self.patch_embedding = PatchEmbed(cfg)
        self.positional_encoding = PositionalEncoding(cfg)
        self.transformer_encoder = nn.Sequential(*[TransformerEncoder(cfg) for _ in range(self.n_layers)])

        self.classifier = nn.Sequential(
            nn.Linear(self.d_model, self.n_classes),
            nn.Softmax(dim = -1)
        )

    def forward(self, images):
        #print(images)
        x = self.patch_embedding(images)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        x = self.classifier(x[:, 0])
        return x