import os
import torch
import torch.nn as nn
import transformers
import torch.nn.functional as F

from transformer_models.vision_tower import VisionTower
from transformer_models.language_tower import LanguageTower
from train_operations.loss import InfoNCE, ClipLoss


class Percival(nn.Module):
    def __init__(self,
                 name='percival',
                 in_channels=1,
                 projection_dim=512,
                 patch_size: tuple = (64, 64, 64),
                 language_model: str = 'yikuan8/Clinical-Longformer',
                 img_size=None,
                 weight_path: str = None,
                 vision_model_size: str = 'small',
                 logit_scale_init=4.6052,
                 use_logits: bool = False,
                 vision_pretrain: str = 'augreg',
                 freeze_language_model: bool = False,
                 use_distributed_loss: bool = False,
                 loss_type: str = 'infonce',
                 ):
        super().__init__()
        self.name = name
        self.in_channels = in_channels
        self.projection_dim = projection_dim
        self.patch_size = patch_size
        self.language_model = language_model
        self.img_size = img_size
        self.weight_path = weight_path
        self.use_distributed_loss = use_distributed_loss
        self.loss_type = loss_type.lower()

        # Learnable logit scale (CLIP-style)
        if use_logits:
            self.logit_scale = nn.Parameter(torch.tensor(float(logit_scale_init)))
        else:
            self.logit_scale = None

        # Contrastive loss
        self._init_criterion()

        # ---- Visual encoder ----
        print(f'[INFO] Initializing visual encoder (size: {vision_model_size}, '
              f'pretrain: {vision_pretrain})')

        self.visual = VisionTower(
            img_size=tuple(reversed(self.img_size)),
            patch_size=self.patch_size,
            in_chans=self.in_channels,
            num_classes=self.projection_dim,
            model_size=vision_model_size,
            vision_pretrain=vision_pretrain,
        )

        # ---- Language tower ----
        self.text = LanguageTower(
            projection_dim=self.projection_dim,
            language_model=language_model,
            freeze_language_model=freeze_language_model,
        )

    # ------------------------------------------------------------------
    # Loss setup
    # ------------------------------------------------------------------

    def _init_criterion(self):
        """Initialize the contrastive loss criterion."""
        from train_operations.loss import get_rank, get_world_size

        rank = get_rank()
        world_size = get_world_size()

        if self.loss_type == 'clip':
            self.criterion = ClipLoss(
                local_loss=False,
                gather_with_grad=False,
                cache_labels=True,
                rank=rank,
                world_size=world_size,
            )
            print(f'[INFO] Using ClipLoss (distributed={self.use_distributed_loss})')
        else:
            self.criterion = InfoNCE(
                temperature=0.1,
                reduction='mean',
                negative_mode='unpaired',
                use_distributed=self.use_distributed_loss,
                local_loss=False,
                gather_with_grad=False,
            )
            print(f'[INFO] Using InfoNCE (distributed={self.use_distributed_loss})')

    def update_distributed_info(self, rank=None, world_size=None):
        from train_operations.loss import get_rank, get_world_size
        if rank is None:
            rank = get_rank()
        if world_size is None:
            world_size = get_world_size()
        if self.loss_type == 'clip' and hasattr(self.criterion, 'rank'):
            self.criterion.rank = rank
            self.criterion.world_size = world_size
            print(f"[INFO] Updated ClipLoss distributed info: rank={rank}, world_size={world_size}")

    # ------------------------------------------------------------------
    # Forward / loss
    # ------------------------------------------------------------------

    def forward(self, img, input_ids, attention_mask, concurrent_ids=None):
        z_img = self.visual(img)
        z_txt = self.text(input_ids, attention_mask)
        return self.compute_contrastive_loss(z_img, z_txt, concurrent_ids=concurrent_ids)

    def compute_contrastive_loss(self, z_img, z_txt, mask=None, concurrent_ids=None):
        if self.loss_type == 'clip':
            z_img_norm = F.normalize(z_img, dim=-1)
            z_txt_norm = F.normalize(z_txt, dim=-1)
            logit_scale = self.logit_scale.exp() if self.logit_scale is not None else torch.tensor(10.0, device=z_img.device)
            return self.criterion(z_img_norm, z_txt_norm, logit_scale,
                                  mask=mask, concurrent_ids=concurrent_ids)
        else:
            return self.criterion(z_img, z_txt, mask=mask)

    # ------------------------------------------------------------------
    # Encoding (inference)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode_image(self, img, normalize=True):
        if img.dim() == 4:
            img = img.unsqueeze(0)
        z = self.visual(img)
        return F.normalize(z, dim=-1) if normalize else z

    @torch.no_grad()
    def encode_text(self, input_ids, attention_mask, normalize=True):
        z = self.text(input_ids, attention_mask)
        return F.normalize(z, dim=-1) if normalize else z

    @torch.no_grad()
    def zero_shot_logits(self, img, input_ids, attention_mask):
        z_img = self.encode_image(img, normalize=True)
        z_txt = self.encode_text(input_ids, attention_mask, normalize=True)
        logits = z_img @ z_txt.T
        if self.logit_scale is not None:
            logits = logits * self.logit_scale.exp()
        return logits

    @torch.no_grad()
    def zero_shot_present_prob(self, img, input_ids, attention_mask):
        if img.dim() == 4:
            img = img.unsqueeze(0)
        logits = self.zero_shot_logits(img, input_ids, attention_mask)
        return torch.softmax(logits.squeeze(0), dim=0)

    # ------------------------------------------------------------------
    # Weight I/O
    # ------------------------------------------------------------------

    def set_weight_path(self, weight_path):
        self.weight_path = weight_path

    def save_visual(self, tag: str = '_'):
        path = os.path.join(self.weight_path, f'visual_{tag}.pth')
        torch.save(self.visual.state_dict(), path)

    def save_text(self, tag: str = '_'):
        path = os.path.join(self.weight_path, f'language_tower_{tag}.pth')
        torch.save(self.text.state_dict(), path)

    def load_visual(self, path: str, strict: bool = True):
        state_dict = torch.load(path, map_location='cpu')
        if not strict:
            model_keys = set(self.visual.state_dict().keys())
            shape_mismatched = [
                k for k, v in state_dict.items()
                if k in model_keys and self.visual.state_dict()[k].shape != v.shape
            ]
            for k in shape_mismatched:
                print(f"[WARN] Shape mismatch: {k}")
                del state_dict[k]
        missing, unexpected = self.visual.load_state_dict(state_dict, strict=strict)
        for k in unexpected:
            print(f"[WARN] Unexpected key: {k}")
        for k in missing:
            print(f"[WARN] Missing key: {k}")
        print('[INFO] Visual encoder weights loaded')

    def load_text(self, path: str, strict: bool = True):
        state_dict = torch.load(path, map_location='cpu')
        if not strict:
            model_keys = set(self.text.state_dict().keys())
            shape_mismatched = [
                k for k, v in state_dict.items()
                if k in model_keys and self.text.state_dict()[k].shape != v.shape
            ]
            for k in shape_mismatched:
                print(f"[WARN] Shape mismatch: {k}")
                del state_dict[k]
        missing, unexpected = self.text.load_state_dict(state_dict, strict=strict)
        for k in unexpected:
            print(f"[WARN] Unexpected key: {k}")
        for k in missing:
            print(f"[WARN] Missing key: {k}")
        print('[INFO] Language tower weights loaded')

    # ------------------------------------------------------------------
    # Aliases
    # ------------------------------------------------------------------

    @property
    def vision(self):
        return self.visual

    def save_image_encoder(self, info='_'):
        self.save_visual(info)

    def save_language_encoder(self, info='_'):
        self.save_text(info)

    def load_image_encoder(self, path, strict=True):
        self.load_visual(path, strict=strict)

    def load_language_encoder(self, path, strict=True):
        self.load_text(path, strict=strict)

    @staticmethod
    def build_scheduler(optimizer, name, warmup_steps, total_steps):
        name = name.lower()
        if name == 'constantlr':
            return transformers.get_constant_schedule(optimizer)
        elif name == 'warmupconstant':
            return transformers.get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)
        elif name == 'warmuplinear':
            return transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        elif name == 'warmupcosine':
            return transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        elif name == 'warmupcosinewithhardrestarts':
            return transformers.get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        else:
            raise ValueError(f"Unknown scheduler {name}")
