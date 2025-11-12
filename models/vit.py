import torch
import torch.nn as nn
from torchvision.models.vision_transformer import VisionTransformer


class ViTEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = VisionTransformer(
            image_size=32,
            patch_size=4,
            num_layers=6,
            num_heads=8,
            hidden_dim=256,
            mlp_dim=512,
            num_classes=10
        )
        self.encoder.heads = nn.Identity()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 256))
        self.pos_embed = nn.Parameter(torch.zeros(1, 65, 256))  # 64 patches + 1 cls
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        B = x.size(0)
        # Patch embeddings
        x = self.encoder.conv_proj(x)               # [B, 256, 8, 8]
        x = x.flatten(2).transpose(1, 2)            # [B, 64, 256]

        # Add CLS token and positional embedding
        cls_tokens = self.cls_token.expand(B, -1, -1)    # [B, 1, 256]
        x = torch.cat((cls_tokens, x), dim=1)            # [B, 65, 256]
        x = x + self.pos_embed

        # Pass through transformer encoder
        x = self.encoder.encoder(x)                      # [B, 65, 256]

        # Return patch tokens, cls token, and pos_embed
        return x[:, 1:, :], x[:, 0, :], self.pos_embed

class ViTDecoder(nn.Module):
    def __init__(self,
                 num_patches=64,       # e.g., 8x8 grid for 32x32 images with patch size 4
                 embed_dim=256,        # input dim from encoder
                 decoder_dim=512,      # internal decoder dimension
                 num_layers=6,
                 num_heads=8,
                 patch_size=4,
                 img_size=32):
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        self.grid_size = self.img_size // self.patch_size
        self.num_patches = num_patches

        # Linear projection to decoder dim
        self.embedding_proj = nn.Linear(embed_dim, decoder_dim)
        nn.init.xavier_uniform_(self.embedding_proj.weight)
        # Learnable positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer decoder
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=decoder_dim, nhead=num_heads, dim_feedforward=decoder_dim * 4),
            num_layers=num_layers
        )

        # Output head to recover pixels
        self.output_head = nn.Sequential(
            nn.LayerNorm(decoder_dim),
            nn.GELU(),
            nn.Linear(decoder_dim, patch_size * patch_size * 3),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [B, 64, embed_dim]
        x = self.embedding_proj(x)          # [B, 64, decoder_dim]
        x = x + self.pos_embed              # Add positional embedding
        x = self.transformer(x.transpose(0, 1)).transpose(0, 1)  # [B, 64, decoder_dim]
        x = self.output_head(x)            # [B, 64, 48]

        # Reshape to image
        B = x.size(0)
        x = x.view(B, self.grid_size, self.grid_size, 3, self.patch_size, self.patch_size)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.view(B, 3, self.img_size, self.img_size)  # [B, 3, 32, 32]
        return x
