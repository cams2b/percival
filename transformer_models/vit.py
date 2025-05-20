import torch
import torch.nn as nn
from torch.nn import Conv3d, LayerNorm
from timm.models.vision_transformer import VisionTransformer, PatchEmbed
import timm
import time

class PatchEmbed3D(PatchEmbed):
    """Patch Embedding Layer for 3D Data"""

    def __init__(self, img_size=(256, 512, 512), patch_size=(64, 64, 64), in_chans=1, embed_dim=1024):
        super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.proj = Conv3d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        # self.norm = LayerNorm(embed_dim)
        self.norm = LayerNorm(embed_dim, eps=1e-6)

    def forward(self, x):
        # Input shape: (B, C, D, H, W)
        # B, C, D, H, W = x.shape
        x = self.proj(x)  # Apply 3D convolution
        x = x.flatten(2).transpose(1, 2)  # Flatten depth, height, width and transpose to (B, N, C)
        x = self.norm(x)
        return x

class VisionTransformer3D(nn.Module):
    """Vision Transformer for 3D Input"""

    def __init__(self, img_size=(256, 512, 512), patch_size=(64, 64, 64), in_chans=1, num_classes=512):
        super().__init__()
        self.base_model = timm.create_model("deit_small_patch16_224", pretrained=True)
        
        # Replace the patch embedding with 3D patch embedding
        self.base_model.patch_embed = PatchEmbed3D(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=self.base_model.embed_dim
        )

        # Calculate number of patches
        depth, height, width = img_size[0] // patch_size[0], img_size[1] // patch_size[1], img_size[2] // patch_size[2]
        num_patches = depth * height * width

        # Replace positional embeddings
        # Original pos_embed has shape (1, N+1, embed_dim). Here, N = num_patches
        self.base_model.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, self.base_model.embed_dim))

        # Replace the classification head
        self.base_model.head = nn.Linear(self.base_model.head.in_features, num_classes)

    def forward(self, x, output_hidden_states=False):
        B = x.size(0)
        x = self.base_model.patch_embed(x)  # Apply 3D patch embedding, shape: (B, N, C)

        # Prepare CLS token
        cls_tokens = self.base_model.cls_token.expand(B, -1, -1)  # Shape: (B, 1, C)

        # Concatenate CLS token to patch embeddings
        x = torch.cat((cls_tokens, x), dim=1)  # Shape: (B, N+1, C)
        x = x + self.base_model.pos_embed  # Shape: (B, N+1, C)
        x = self.base_model.pos_drop(x)  # Apply dropout

        hidden_states = []
        for blk in self.base_model.blocks:
            x = blk(x)
            if output_hidden_states:
                hidden_states.append(x.clone())

        x = self.base_model.norm(x)

        # Classification using CLS token
        cls_token_final = x[:, 0]  # Shape: (B, C)
        x = self.base_model.head(cls_token_final)  # Shape: (B, num_classes)

        if output_hidden_states:
            return {
                "logits": x,
                "hidden_states": hidden_states  # List of tensors from each block
            }
        else:
            return x




