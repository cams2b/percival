"""3D Vision Transformer for CT imaging.

Inflates a pretrained 2D ViT (MAE or DeiT) to 3D by replacing the patch
embedding with a 3D convolution and adapting position embeddings to 3D.
Full scan-level attention only — no slice-level hierarchical attention.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv3d, LayerNorm
from timm.models.vision_transformer import VisionTransformer, PatchEmbed
from timm.layers import resample_abs_pos_embed as resample_2d_posemb
import timm


class PatchEmbed3D(PatchEmbed):
    """Patch Embedding Layer for 3D Data."""

    def __init__(self, img_size=(128, 256, 256), patch_size=(64, 64, 64), in_chans=1, embed_dim=1024):
        super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.proj = Conv3d(
            in_chans, embed_dim,
            kernel_size=patch_size, stride=patch_size,
        )
        self.norm = LayerNorm(embed_dim, eps=1e-6)
        self.grid_size = (
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
            img_size[2] // patch_size[2],
        )

    def forward(self, x):
        x = self.proj(x)                       # (B, C, D', H', W')
        x = x.flatten(2).transpose(1, 2)       # (B, N, C)
        x = self.norm(x)
        return x


# ---------------------------------------------------------------------------
# Position embedding helpers
# ---------------------------------------------------------------------------

def get_3d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """Generate 3D sinusoidal position embeddings.

    Args:
        embed_dim: Must be divisible by 3.
        grid_size: (D, H, W)
    """
    assert embed_dim % 3 == 0
    grid_d = np.arange(grid_size[0], dtype=np.float32)
    grid_h = np.arange(grid_size[1], dtype=np.float32)
    grid_w = np.arange(grid_size[2], dtype=np.float32)
    grid = np.meshgrid(grid_d, grid_h, grid_w, indexing='ij')
    grid = np.stack(grid, axis=0).reshape(3, 1, *grid_size)
    emb = _3d_sincos_from_grid(embed_dim, grid)
    if cls_token:
        emb = np.concatenate([np.zeros([1, embed_dim]), emb], axis=0)
    return emb


def _3d_sincos_from_grid(embed_dim, grid):
    d = embed_dim // 3
    emb_d = _1d_sincos_from_grid(d, grid[0])
    emb_h = _1d_sincos_from_grid(d, grid[1])
    emb_w = _1d_sincos_from_grid(d, grid[2])
    return np.concatenate([emb_w, emb_h, emb_d], axis=1)


def get_1d_sincos_pos_embed(embed_dim, seq_len, cls_token=False):
    pos = np.arange(seq_len, dtype=np.float32)
    emb = _1d_sincos_from_grid(embed_dim, pos)
    if cls_token:
        emb = np.concatenate([np.zeros([1, embed_dim]), emb], axis=0)
    return emb


def _1d_sincos_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32) / (embed_dim / 2.0)
    omega = 1.0 / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


# ---------------------------------------------------------------------------
# Vision Tower
# ---------------------------------------------------------------------------

class VisionTower(nn.Module):
    """3D Vision Transformer with full scan-level attention.

    Inflates a pretrained 2D ViT to 3D. No hierarchical / slice-level
    attention — every transformer block operates on the full token sequence.
    """

    def __init__(self, img_size=(128, 256, 256), patch_size=(64, 64, 64),
                 in_chans=1, num_classes=512, model_size='small',
                 vision_pretrain='augreg'):
        """
        Args:
            img_size: (D, H, W)
            patch_size: 3D patch dimensions
            in_chans: Input channels (1 for CT)
            num_classes: Output projection dimension
            model_size: 'tiny', 'small', 'base', 'large', 'huge'
            vision_pretrain: Pretrained weight source — 'augreg', 'mae', or 'deit'
        """
        super().__init__()

        self.base_model = self._create_vit(model_size, vision_pretrain)
        print('[INFO] initialization complete')

        embed_dim = self.base_model.embed_dim

        # Replace 2D patch embed with 3D
        self.base_model.patch_embed = PatchEmbed3D(
            img_size=img_size, patch_size=patch_size,
            in_chans=in_chans, embed_dim=embed_dim,
        )

        # Grid info
        depth = img_size[0] // patch_size[0]
        height = img_size[1] // patch_size[1]
        width = img_size[2] // patch_size[2]
        self.grid_size = (depth, height, width)
        self.num_patches = depth * height * width

        # 3D position embeddings
        has_pretrained_weights = vision_pretrain in ('augreg', 'mae')
        self._setup_pos_embed(embed_dim, has_pretrained_weights, patch_size)

        # Projection head
        self.base_model.head = nn.Linear(embed_dim, num_classes, bias=False)

    # ------------------------------------------------------------------
    # Model creation
    # ------------------------------------------------------------------

    def _create_vit(self, model_size, vision_pretrain):
        """Create ViT model with specified pretrained weights.

        Args:
            model_size: 'tiny', 'small', 'base', 'large', 'huge'
            vision_pretrain: 'augreg' (ImageNet-21K), 'mae', or 'deit'
        """
        augreg_models = {
            'tiny': 'vit_tiny_patch16_224.augreg_in21k_ft_in1k',
            'small': 'vit_small_patch16_224.augreg_in21k_ft_in1k',
            'base': 'vit_base_patch16_224.augreg_in21k_ft_in1k',
            'large': 'vit_large_patch16_224.augreg_in21k_ft_in1k',
        }
        mae_models = {
            'base': 'vit_base_patch16_224.mae',
            'large': 'vit_large_patch16_224.mae',
            'huge': 'vit_huge_patch14_224.mae',
        }
        deit_models = {
            'tiny': 'hf_hub:timm/deit_tiny_patch16_224.fb_in1k',
            'small': 'deit_small_patch16_224',
            'base': 'hf_hub:timm/deit3_base_patch16_224.fb_in1k',
            'huge': 'deit3_huge_patch14_224.fb_in1k',
        }

        def _load_and_store_weights(model_name, label):
            """Load model and store 2D weights for 3D adaptation."""
            print(f'[INFO] Initializing with {label} {model_size} ViT ({model_name})...')
            model = timm.create_model(model_name, pretrained=True)
            self._mae_patch_embed_weight = model.patch_embed.proj.weight.data.clone()
            self._mae_pos_embed = model.pos_embed.data.clone()
            self._mae_cls_token = model.cls_token.data.clone()
            del model.pos_embed
            return model

        # AugReg (ImageNet-21K pretrained with augmentation + regularization)
        if vision_pretrain == 'augreg' and model_size in augreg_models:
            return _load_and_store_weights(augreg_models[model_size], 'AugReg')

        # MAE (masked autoencoder self-supervised)
        if vision_pretrain == 'mae' and model_size in mae_models:
            return _load_and_store_weights(mae_models[model_size], 'MAE')

        # DeiT fallback (ImageNet-1K supervised)
        if vision_pretrain not in ('augreg', 'mae'):
            vision_pretrain = 'deit'  # explicit fallback
        model_name = deit_models.get(model_size, 'deit_small_patch16_224')
        print(f'[INFO] Initializing with DeiT {model_size} ViT ({model_name})...')
        return timm.create_model(model_name, pretrained=True)

    # ------------------------------------------------------------------
    # Position embeddings
    # ------------------------------------------------------------------

    def _setup_pos_embed(self, embed_dim, has_pretrained_weights, patch_size):
        """Setup 3D position embeddings. Adapts from pretrained 2D weights if available."""

        # Adapt 2D patch embedding weights to 3D
        if has_pretrained_weights and hasattr(self, '_mae_patch_embed_weight'):
            print('[INFO] Adapting MAE 2D patch_embed to 3D...')
            w = self._mae_patch_embed_weight
            if (w.shape[2], w.shape[3]) != (patch_size[1], patch_size[2]):
                w = F.interpolate(w, size=(patch_size[1], patch_size[2]), mode='bicubic')
            w = w.sum(dim=1, keepdim=True).unsqueeze(2).repeat(
                1, 1, patch_size[0], 1, 1).div(patch_size[0])
            self.base_model.patch_embed.proj.weight.data = w
            print(f'[INFO] Adapted patch_embed: {self._mae_patch_embed_weight.shape} -> {w.shape}')
            del self._mae_patch_embed_weight

        # Adapt 2D pos_embed to 3D
        if has_pretrained_weights and hasattr(self, '_mae_pos_embed'):
            print('[INFO] Adapting MAE 2D pos_embed to 3D...')
            pos_embed = self._mae_pos_embed
            cls_token = self._mae_cls_token

            embed_len = pos_embed.shape[1]
            sqrt_len = torch.sqrt(torch.tensor(float(embed_len)))
            if sqrt_len != sqrt_len.floor():
                cls_token = cls_token + pos_embed[:, 0:1, :]
                pos_embed = pos_embed[:, 1:, :]
                print('[INFO] Merged CLS position into cls_token')

            self.base_model.cls_token.data = cls_token

            spatial_2d = resample_2d_posemb(
                pos_embed, new_size=(self.grid_size[1], self.grid_size[2]),
                num_prefix_tokens=0,
            ).reshape(1, self.grid_size[1], self.grid_size[2], embed_dim)

            depth_pos = get_1d_sincos_pos_embed(embed_dim, self.grid_size[0])
            depth_pos = torch.from_numpy(depth_pos).float()

            spatial_posemb = depth_pos.view(self.grid_size[0], 1, 1, embed_dim) + spatial_2d
            self.spatial_posemb = nn.Parameter(spatial_posemb.unsqueeze(0))
            print(f'[INFO] Adapted 3D spatial_posemb: {self.spatial_posemb.shape}')

            del self._mae_pos_embed, self._mae_cls_token
        else:
            spatial_posemb = get_3d_sincos_pos_embed(embed_dim, self.grid_size)
            spatial_posemb = torch.from_numpy(spatial_posemb).float()
            self.spatial_posemb = nn.Parameter(spatial_posemb.reshape(
                1, *self.grid_size, embed_dim))
            print(f'[INFO] Initialized 3D spatial_posemb: {self.spatial_posemb.shape}')

        self.spatial_posemb.requires_grad = False

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x, output_hidden_states=False):
        """
        Args:
            x: (B, C, D, H, W) input volume.
            output_hidden_states: Return per-block hidden states.

        Returns:
            (B, num_classes) or dict with 'logits' and 'hidden_states'.
        """
        B = x.size(0)

        # Patch embed + 3D position encoding
        x = self.base_model.patch_embed(x)                          # (B, N, C)
        x = x.view(B, *self.grid_size, -1)                         # (B, D, H, W, C)
        x = x + self.spatial_posemb                                 # add 3D pos
        x = x.view(B, -1, x.size(-1))                              # (B, N, C)

        # Prepend CLS token
        cls = self.base_model.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)                             # (B, 1+N, C)
        x = self.base_model.pos_drop(x)

        # Transformer blocks
        hidden_states = []
        for blk in self.base_model.blocks:
            x = blk(x)
            if output_hidden_states:
                hidden_states.append(x.clone())

        x = self.base_model.norm(x)
        out = self.base_model.head(x[:, 0])                        # CLS → projection

        if output_hidden_states:
            return {"logits": out, "hidden_states": hidden_states}
        return out
