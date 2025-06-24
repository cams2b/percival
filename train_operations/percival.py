import os
import numpy as np
import pandas as pd
from scipy.special import expit
import torch
import torch.nn as nn
from torch.optim import Optimizer
import transformers
from typing import List, Dict, Tuple, Iterable, Type, Union, Callable, Optional
from itertools import chain
from torch.cuda.amp import autocast
from transformer_models.vit import VisionTransformer3D
from text_operations.bert import grail
from train_operations.loss import InfoNCE
from data_operations.percival_monai_dataset import get_inference_transform


class percival(nn.Module):
    def __init__(self, 
                 name='percival',
                 in_channels=1,
                 projection_dim=512,
                 language_model=None,
                 optimizer_class=torch.optim.AdamW,
                 optimizer_lr=2e-5,
                 weight_decay=0.01,
                 betas=(0.9, 0.999),
                 img_size=None,
                 scheduler='warmuplinear',
                 static_lr=False,
                 warmup_steps=100,
                 total_steps=None,
                 use_amp=False,
                 max_grad_norm=1,
                 embed_last_layer=False,
                 weight_path=None):
        super(percival, self).__init__()
        self.name = name
        self.in_channels = in_channels
        self.projection_dim = projection_dim
        self.language_model = language_model
        self.optimizer_lr = optimizer_lr
        self.weight_decay = weight_decay
        self.betas = betas
        self.img_size = img_size
        self.use_amp = use_amp
        self.max_grad_norm = max_grad_norm
        self.weight_path = weight_path
        self.embed_last_layer = embed_last_layer
        self.static_lr = static_lr
        self.criterion = InfoNCE()
        self.vision = VisionTransformer3D(img_size=self.img_size, patch_size=(64, 64, 64), in_chans=self.in_channels, num_classes=self.projection_dim)
        # self.grail = grail(self.language_model, projection_dim=self.projection_dim, projection_bias=False, embed_last_layer=self.embed_last_layer)
        self.grail = grail(projection_dim=self.projection_dim)

        self.optimizer = optimizer_class(
            chain(self.vision.parameters(), self.grail.parameters()), 
            lr=self.optimizer_lr, 
            weight_decay=self.weight_decay, 
            betas=self.betas
        )

        if not static_lr:
            print('[INFO] Percival is using the following scheduler: {}'.format(scheduler))
            self.training_scheduler = self.get_scheduler(self.optimizer, scheduler, warmup_steps, total_steps)
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None

    
    
    def forward(self, img, txt):
        z_img = self.vision(img)
        z_txt = self.grail(txt)
        loss = self.criterion(z_img, z_txt)
        return loss

    def update_scheduler(self, num_epochs, steps_per_epoch):
        for _ in range(num_epochs * steps_per_epoch):
            self.training_scheduler.step()

    def optimize_parameters(self, img, txt):    
        """
        Performs a single forward backward pass for a batch of data
        """
        self.optimizer.zero_grad()
        if self.use_amp:
            with autocast():
                loss = self.forward(img, txt)
                
            if not torch.isfinite(loss):
                print("Non-finite loss, skipping step:", loss.item())
                return
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.vision.parameters(), self.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.grail.parameters(), self.max_grad_norm)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.epoch_loss += loss.detach().item()

        else:
            loss = self.forward(img, txt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.vision.parameters(), self.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.grail.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self.epoch_loss += loss.detach().item()

        if not self.static_lr:
            self.training_scheduler.step()


    def inference_from_path(self, img_path, device):
        transforms = get_inference_transform(self.img_size)
        img = transforms(img_path)
        img = torch.swapaxes(img, 1, -1) 
        img = img.unsqueeze(0)
        img = img.to(device)
        z_img = self.vision(img).detach().cpu()
        z_img = z_img.numpy()
        
        return z_img

    def compute_principal_components(self, z_img):
        center_scale = pd.read_csv('train_operations/data/pca_center_scale.csv')
        rotation = pd.read_csv('train_operations/data/pca_rotation_matrix.csv')
        center = center_scale['center'].values
        scale = center_scale['scale'].values
        
        rotation_matrix = rotation[[f'PC{i}' for i in range(1, 11)]].values  # (512, 10)
        assert z_img.shape[1] == len(center), "[ERROR] z_img shape mismatch with PCA center"
        
        z_scaled = (z_img - center) / scale
        # 4. Compute principal components
        pc_scores = np.dot(z_scaled, rotation_matrix)  # shape (1, 10)

        return pc_scores

    def diagnostic_inference(self, img_path, device, target_phecode=None):
        z_img = self.inference_from_path(img_path=img_path, device=device)
        pc = self.compute_principal_components(z_img=z_img)
        coef_df = pd.read_csv("train_operations/data/diagnosis/diagnosis_model_coefficients.csv") 

        if target_phecode is None:
            raise ValueError("Please specify a target_condition for inference")

        # Step 3: Get coefficients for the target condition
        row = coef_df[coef_df["phecode"] == target_phecode]
        if row.empty:
            raise ValueError(f"Condition '{target_phecode}' not found in coefficient table")

        row = row.iloc[0]
        intercept = row["(Intercept)"]
        pc_coefs = row[[f"PC{i}" for i in range(1, 11)]].values.astype(np.float32)  # shape: (10,)

        # Step 4: Compute logits and probability
        logits = np.dot(pc, pc_coefs.T) + intercept  # shape: (1,)
        prob = expit(logits[0])  # scalar

        return {
            "principal_components": pc,
            "phecode": target_phecode,
            "predicted_probability": prob,
            "predicted_label": int(prob >= 0.5)
        }
    
    def diagnostic_inference_all_conditions(self, img_path, device):
        z_img = self.inference_from_path(img_path=img_path, device=device)
        pc = self.compute_principal_components(z_img=z_img)
        print('[INFO] principal components done!!!!!!!!')
        coef_df = pd.read_csv("train_operations/data/diagnosis/diagnosis_model_coefficients.csv") 

        # Step 3: Prepare predictions
        results = []

        for _, row in coef_df.iterrows():
            phecode = row["phecode"]
            intercept = row["(Intercept)"]

            try:
                pc_coefs = row[[f"PC{i}" for i in range(1, 11)]].values.astype(np.float32)
            except KeyError:
                print(f"[WARNING] PC columns missing for phecode: {phecode}")
                continue

            logit = np.dot(pc, pc_coefs.T) + intercept  # shape: (1,)
            prob = expit(logit[0])  # scalar

            results.append({
                "phecode": phecode,
                "predicted_probability": prob,
                "predicted_label": int(prob >= 0.5)
            })

        return pd.DataFrame(results)

    
    def prognostic_inference(self, img_path, device, target_condition=None):
        z_img = self.inference_from_path(img_path=img_path, device=device)
        pc = self.compute_principal_components(z_img=z_img)
        coef_df = pd.read_csv("train_operations/data/prognosis/cox_model_coefficients.csv")

        if target_condition is None:
            raise ValueError("Please specify a target condition for inference")

        row = coef_df[coef_df["condition"] == target_condition]

        row = row.iloc[0]
        pc_coefs = row[[f"PC{i}" for i in range(1, 11)]].values.astype(np.float32)  # shape: (10,)

        # Compute linear predictor (LP) and hazard
        linear_predictor = np.dot(pc, pc_coefs.T).item()  # scalar
        relative_hazard = np.exp(linear_predictor)

        return {
            "principal_components": pc,
            "condition": target_condition,
            "linear_predictor": linear_predictor,
            "relative_hazard": relative_hazard
        }

    
    def prognostic_inference_all_conditions(self, img_path, device):
        z_img = self.inference_from_path(img_path=img_path, device=device)
        pc = self.compute_principal_components(z_img=z_img)  # shape: (1, 10)

        coef_df = pd.read_csv("train_operations/data/prognosis/cox_model_coefficients.csv")

        results = []

        for _, row in coef_df.iterrows():
            condition = row["condition"]

            try:
                pc_coefs = row[[f"PC{i}" for i in range(1, 11)]].values.astype(np.float32)
            except KeyError:
                print(f"[WARNING] PC columns missing for condition: {condition}")
                continue

            linear_predictor = np.dot(pc, pc_coefs.T).item()
            relative_hazard = np.exp(linear_predictor)

            results.append({
                "condition": condition,
                "linear_predictor": linear_predictor,
                "relative_hazard": relative_hazard
            })

        return pd.DataFrame(results)




    def image_inference(self, img):
        """
        Performs a single forward pass for a batch of inference imaging data
        """
        z_img = self.vision(img).detach().cpu()
        return z_img


    def image_text_inference(self, img, txt):
        """
        Performs a single forward pass for a batch of inference imaging and text data 
        """
        positive_key = self.grail(txt).detach().cpu()
        query = self.vision(img).detach().cpu()
        return positive_key, query

    

    def image_text_recall(self, img, input_ids, attention_mask):
        """
        Computes the recall between paired image-text data.
        """
        img, input_ids, attention_mask = img.to(self.device), input_ids.to(self.device), attention_mask.to(self.device)
        query = self.vision(img).cpu()
        positive_key = self.grail(input_ids, attention_mask).cpu()

        query_norm = query / query.norm(dim=1, keepdim=True)
        positive_key_norm = positive_key / positive_key.norm(dim=1, keepdim=True)

        cosine_similarity = query_norm @ positive_key_norm.T
        max_indices = torch.argmax(cosine_similarity, dim=1)
        cosine_binary = torch.zeros_like(cosine_similarity)
        cosine_binary.scatter_(1, max_indices.unsqueeze(1), 1)
        cosine_diagonal = cosine_binary * torch.eye(cosine_binary.shape[0])
        correct = torch.sum(cosine_diagonal).detach().item()

        return correct




    def get_scheduler(self, 
                      optimizer: Optimizer, 
                      scheduler: str, 
                      warmup_steps: int, 
                      t_total: int):
        """
        Creates learning rate scheduler.
        """
        scheduler = scheduler.lower()
        if scheduler == 'constantlr':
            return transformers.get_constant_schedule(optimizer)
        elif scheduler == 'warmupconstant':
            return transformers.get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)
        elif scheduler == 'warmuplinear':
            return transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
        elif scheduler == 'warmupcosine':
            return transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
        elif scheduler == 'warmupcosinewithhardrestarts':
            return transformers.get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
        else:
            raise ValueError(f"Unknown scheduler {scheduler}")

    def save_image_encoder(self, info: str = '_') -> None:
        """
        Saves image encoder model to the designated path
        """
        curr_path = os.path.join(self.weight_path, f'image_encoder_{info}.pth')
        torch.save(self.vision.state_dict(), curr_path)

    def save_language_encoder(self, info: str = '_') -> None:
        """
        Saves language encoder model to the designated path
        """
        curr_path = os.path.join(self.weight_path, f'language_encoder_{info}.pth')
        torch.save(self.grail.state_dict(), curr_path)

    def save_scheduler(self, info: str = '_') -> None:
        """
        Saves the scheduler state to the designated path.
        """
        curr_path = os.path.join(self.weight_path, f'scheduler_{info}.pth')
        torch.save(self.training_scheduler.state_dict(), curr_path)
        print(f'[INFO] Successfully saved scheduler state to {curr_path}')

    

    def load_image_encoder(self, path: str = None, strict: bool = True) -> None:
        """
        Loads image encoder model weights from the designated path
        """
        state_dict = torch.load(path)
        if not strict:
            model_keys = set(self.vision.state_dict().keys())
            mismatched = []
            for k, v in state_dict.items():
                if k in model_keys and self.vision.state_dict()[k].shape != v.shape:
                    mismatched.append(k)
            for k in mismatched:
                print(f"[WARN] Removing mismatched key from checkpoint: {k}")
                del state_dict[k]
        self.vision.load_state_dict(state_dict, strict=strict)
        print('[INFO] successfully loaded image encoder weights')

    def load_language_encoder(self, path: str = None, strict: bool = True) -> None:
        """
        Loads language encoder model weights from the designated path
        """
        state_dict = torch.load(path)
        if not strict:
            model_keys = set(self.grail.state_dict().keys())
            mismatched = []
            for k, v in state_dict.items():
                if k in model_keys and self.grail.state_dict()[k].shape != v.shape:
                    mismatched.append(k)
            for k in mismatched:
                print(f"[WARN] Removing mismatched key from checkpoint: {k}")
                del state_dict[k]
        self.grail.load_state_dict(state_dict, strict=strict)
        print('[INFO] successfully loaded language encoder weights')


    def load_scheduler(self, path: str = None) -> None:
        """
        Loads scheduler state from the designated path.
        """
        if path is None:
            raise ValueError("Path to scheduler checkpoint must be provided.")
        self.training_scheduler.load_state_dict(torch.load(path))
        print(f'[INFO] Successfully loaded scheduler state from {path}')


