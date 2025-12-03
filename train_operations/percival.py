import os
import numpy as np
import torch
import torch.nn as nn
from scipy.special import expit
import transformers
import torch.nn.functional as F

from transformer_models.vit import VisionTransformer3D
from text_operations.bert import grail
from train_operations.loss import InfoNCE


class percival(nn.Module):
    def __init__(self, 
                 name='percival',
                 in_channels=1,
                 projection_dim=512,
                 patch_size:tuple=(64, 64, 64),
                 language_model=None,
                 img_size=None,
                 weight_path:str=None,
                 vision_model_size:str='small',
                 logit_scale_init=4.6052,
                 use_logits:bool=False):
        super().__init__()
        self.name = name
        self.in_channels = in_channels
        self.projection_dim = projection_dim
        self.patch_size = patch_size
        self.language_model = language_model
        self.img_size = img_size
        self.weight_path = weight_path
        if use_logits:
            self.logit_scale = nn.Parameter(torch.tensor(float(logit_scale_init)))
        else:
            self.logit_scale = None
        self.criterion = InfoNCE()

        self.vision = VisionTransformer3D(img_size=tuple(reversed(self.img_size)), ## we reverse 
                                          patch_size=self.patch_size,
                                          in_chans=self.in_channels,
                                          num_classes=self.projection_dim,
                                          model_size=vision_model_size)

        self.grail = grail(projection_dim=self.projection_dim, language_model=language_model)

    def forward(self, img, txt):
        z_img = self.vision(img)
        z_txt = self.grail(txt)
        return self.criterion(z_img, z_txt)

    def set_weight_path(self, weight_path):
        self.weight_path = weight_path

    # --- inference helpers (unchanged) ---
    @torch.no_grad()
    def encode_image(self, img, normalize=True):
        z = self.vision(img)
        return F.normalize(z, dim=-1) if normalize else z

    @torch.no_grad()
    def encode_text(self, txt, normalize=True):
        z = self.grail(txt)
        return F.normalize(z, dim=-1) if normalize else z


    def save_image_encoder(self, info: str = '_'):
        curr_path = os.path.join(self.weight_path, f'image_encoder_{info}.pth')
        torch.save(self.vision.state_dict(), curr_path)


    def save_language_encoder(self, info: str = '_'):
        curr_path = os.path.join(self.weight_path, f'language_encoder_{info}.pth')
        torch.save(self.grail.state_dict(), curr_path)


    def load_image_encoder(self, path: str = None, strict: bool = True):
        state_dict = torch.load(path)
        if not strict:
            model_keys = set(self.vision.state_dict().keys())
            mismatched = [k for k,v in state_dict.items()
                          if k in model_keys and self.vision.state_dict()[k].shape != v.shape]
            for k in mismatched:
                print(f"[WARN] Removing mismatched key from checkpoint: {k}")
                del state_dict[k]
        self.vision.load_state_dict(state_dict, strict=strict)
        print('[INFO] successfully loaded image encoder weights')


    def load_language_encoder(self, path: str = None, strict: bool = True):
        state_dict = torch.load(path)
        if not strict:
            model_keys = set(self.grail.state_dict().keys())
            mismatched = [k for k,v in state_dict.items()
                          if k in model_keys and self.grail.state_dict()[k].shape != v.shape]
            for k in mismatched:
                print(f"[WARN] Removing mismatched key from checkpoint: {k}")
                del state_dict[k]
        self.grail.load_state_dict(state_dict, strict=strict)
        print('[INFO] successfully loaded language encoder weights')

    # optional: keep a factory for schedulers (used by trainer)
    @staticmethod
    def build_scheduler(optimizer, name: str, warmup_steps: int, total_steps: int):
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
