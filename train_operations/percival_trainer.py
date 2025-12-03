import os
import math
import time
import torch
import pandas as pd
import torch.distributed as dist
from collections import defaultdict
from typing import Callable
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import transformers
from accelerate import Accelerator

from train_operations.make_experiment import make_experiment, save_experiment_config
from data_operations.percival_monai_dataset import percival_dataset


class percival_trainer(object):
    def __init__(self,
                 model,
                 experiment_name: str = None,
                 training_path: str = None,
                 validation_path: str = None,
                 test_path: str = None,
                 train_transform: bool = True,
                 validation_transform: bool = False,
                 patient_id: str = None,
                 image_col: str = None,
                 text_col: str = None,
                 evaluator=None,
                 image_size: tuple = (256, 256, 128),
                 image_spacing: tuple = (1.5, 1.5, 3),
                 in_channels: int = 1,
                 projection_dim: int = 512,
                 language_model: str = None,
                 epochs: int = 1,
                 batch_size: int = 2,
                 steps_per_epoch=None,
                 scheduler: str = 'warmuplinear',
                 static_lr: bool = False,
                 warmup_steps: int = 10000,
                 warmup_ratio: float = 0.1,
                 optimizer_class = torch.optim.AdamW,
                 optimizer_lr: float = 2e-5,
                 weight_decay: float = 0.01,
                 evaluation_steps=None,
                 sanitize:bool=True,
                 save_steps: int = 100,
                 output_path: str = None,
                 save_best_model: bool = True,
                 max_grad_norm: float = 1.0,
                 use_amp: bool = False,
                 accumulation_steps: int = 1,
                 callback: Callable[[float, int, int], None] = None,
                 show_progress_bar: bool = False,
                 checkpoint_path: str = None,
                 continue_training: bool = False,
                 image_weights: str = None,
                 language_weights: str = None,
                 load_best_model: bool = True,
                 num_workers: int = 1,
                 pin_memory: bool = False,
                 load_strict: bool = True,
                 distributed: bool = False,
                 config = None):
        # config
        self.model = model
        self.experiment_name = experiment_name
        self.training_path, self.train_transform = training_path, train_transform
        self.validation_path, self.validation_transform = validation_path, validation_transform
        self.test_path = test_path
        self.patient_id, self.image_col, self.text_col = patient_id, image_col, text_col

        self.evaluator = evaluator
        self.image_size = image_size
        self.image_spacing = image_spacing
        self.in_channels = in_channels
        self.projection_dim = projection_dim
        self.language_model = language_model

        self.epochs = epochs
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch

        self.scheduler = scheduler
        self.static_lr = static_lr
        self.warmup_steps = warmup_steps
        self.warmup_ratio = warmup_ratio

        self.optimizer_class = optimizer_class
        self.optimizer_lr = optimizer_lr
        self.weight_decay = weight_decay

        self.sanitize = sanitize

        self.evaluation_steps = evaluation_steps
        self.save_steps = save_steps
        self.output_path = output_path
        self.save_best_model = save_best_model

        self.max_grad_norm = max_grad_norm
        self.use_amp = use_amp
        self.accumulation_steps = max(1, int(accumulation_steps))

        self.callback = callback
        self.show_progress_bar = show_progress_bar
        self.checkpoint_path = checkpoint_path

        self.continue_training = continue_training
        self.image_weights = image_weights
        self.language_weights = language_weights
        self.load_best_model = load_best_model

        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.load_strict = load_strict

        # runtime state
        self.score_logs = defaultdict(list)
        self.distributed = distributed
        self.rank = 0
        self.world_size = 1
        self.local_rank = 0
        self.device = torch.device("cpu")
        self.config = config
    # ----------------------- utils -----------------------

    def _unwrap(self):
        """Return the underlying module regardless of DDP wrap."""
        return self.model.module if isinstance(self.model, DDP) else self.model

    def _build_scheduler(self, optimizer, warmup_steps, total_steps):
        """Use model's factory if present; otherwise build here."""
        m = self._unwrap()
        if hasattr(m, "build_scheduler"):
            return m.build_scheduler(optimizer, self.scheduler, warmup_steps, total_steps)

        name = (self.scheduler or "").lower()
        if self.static_lr:
            return None
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
            if self.rank == 0:
                print(f"[WARN] Unknown scheduler '{self.scheduler}', proceeding with static LR.")
            return None


    def train_accelerate(self) -> None:
        # ---------- setup ----------
        experiment_path, weight_path, out_dir = make_experiment(self.output_path, self.experiment_name)
        save_experiment_config(self.config, output_dir=out_dir)
        self.model.set_weight_path(weight_path)
        metrics_csv = os.path.join(out_dir, "metrics_run_w_validation.csv")
        
        mixed_precision = "fp16" if self.use_amp else "no"
        grad_accum = int(getattr(self, "grad_accum", 1))
        accelerator = Accelerator(mixed_precision=mixed_precision,
                                gradient_accumulation_steps=grad_accum)
        device = accelerator.device
        world_size = accelerator.num_processes
        local_rank = accelerator.local_process_index

        if accelerator.is_main_process:
            print(f"[INFO] Using Accelerate | world_size={world_size} | mixed_precision={mixed_precision} | grad_accum={grad_accum}")

            
        train_dataset = percival_dataset(
            data_path=self.training_path,
            image_col=self.image_col,
            text_col=self.text_col,
            pid_col=self.patient_id,
            image_size=self.image_size,
            target_spacing=self.image_spacing,
            augment=self.train_transform,
            sanitize=self.sanitize)
            
        validation_dataset = percival_dataset(
                            data_path=self.validation_path,
                            image_col=self.image_col,
                            text_col=self.text_col,
                            pid_col=self.patient_id,
                            image_size=self.image_size,
                            target_spacing=self.image_spacing,
                            augment=self.validation_transform,
                            sanitize=self.sanitize)

        # Let Accelerate inject DistributedSampler; just set shuffle=True.
        prefetch = 2 if (self.num_workers and self.num_workers > 0) else None
        train_loader = DataLoader(train_dataset,
                                  batch_size=self.batch_size,
                                  shuffle=True,
                                  num_workers=self.num_workers,
                                  pin_memory=self.pin_memory,
                                  persistent_workers=False,
                                  prefetch_factor=prefetch,
                                  drop_last=True)
        
        val_loader = DataLoader(validation_dataset,
                                batch_size=self.batch_size,
                                shuffle=True,
                                num_workers=self.num_workers,
                                pin_memory=self.pin_memory,
                                persistent_workers=False,
                                prefetch_factor=prefetch,
                                drop_last=True)

        
        if self.continue_training and (self.image_weights or self.language_weights):
            if self.image_weights:
                self.model.load_image_encoder(self.image_weights, strict=self.load_strict)
            if self.language_weights:
                self.model.load_language_encoder(self.language_weights, strict=self.load_strict)

        
        optimizer = self.optimizer_class(self.model.parameters(), lr=self.optimizer_lr, weight_decay=self.weight_decay)

        self.model, optimizer, train_loader = accelerator.prepare(self.model, optimizer, train_loader)
        val_loader = accelerator.prepare(val_loader)

        
        # per-process steps (with drop_last=True)
        per_proc_steps_per_epoch = len(train_loader)
        effective_steps_per_epoch = per_proc_steps_per_epoch
        if not getattr(self, "steps_per_epoch", None):
            self.steps_per_epoch = effective_steps_per_epoch

        # account for gradient accumulation (global optimizer steps per epoch)
        eff_optimizer_steps_per_epoch = math.floor(self.steps_per_epoch / grad_accum)

        total_optimizer_steps = int(eff_optimizer_steps_per_epoch * self.epochs)
        warmup_steps = math.ceil(total_optimizer_steps * self.warmup_ratio)

        if accelerator.is_main_process:
            print(f"[INFO] scheduler='{self.scheduler}', static_lr={self.static_lr}")
            print(f"[INFO] per_proc_steps_per_epoch={per_proc_steps_per_epoch}, "
                f"effective_global_steps_per_epoch={self.steps_per_epoch}, "
                f"grad_accum={grad_accum}, "
                f"epochs={self.epochs}, "
                f"total_optimizer_steps={total_optimizer_steps}, warmup_steps={warmup_steps}")

        # Build scheduler now that steps are known
        if self.static_lr:
            scheduler = None
        else:
            scheduler = self._build_scheduler(optimizer, warmup_steps, total_optimizer_steps)

        # ---------- CSV bootstrap (main only) ----------
        if accelerator.is_main_process:
            if os.path.exists(metrics_csv):
                metrics_df = pd.read_csv(metrics_csv)
            else:
                metrics_df = pd.DataFrame(columns=["epoch", "loss", "imgs_per_sec", "epoch_time_sec","world_size", "batch_size", "steps_per_epoch", "val_loss"])
        best_loss = float('inf')

        # ---------- training loop ----------
        self.model.train()
        for epoch in range(self.epochs):
            if accelerator.is_main_process:
                print(f"[INFO] Epoch: {epoch}")

            epoch_t0 = time.time()
            local_loss_sum = 0.0
            local_step_count = 0

            accelerator.wait_for_everyone()

            optimizer.zero_grad(set_to_none=True)
            for step, (img, txt) in enumerate(train_loader):
                if accelerator.is_main_process and step % 10 == 0:
                    print(f"[INFO] step {step}/{self.steps_per_epoch}; running_loss {local_loss_sum:.4f}")

                img = img.to(device, non_blocking=True)

                with accelerator.accumulate(self.model):
                    loss = self.model(img, list(txt))

                    if not torch.isfinite(loss):
                        if accelerator.is_main_process:
                            print("Non-finite loss, skipping step:", float(loss.detach().item()))
                        optimizer.zero_grad(set_to_none=True)
                        continue

                    accelerator.backward(loss)
                    if self.max_grad_norm and self.max_grad_norm > 0:
                        accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                    if scheduler is not None:
                        scheduler.step()

                local_loss_sum += float(loss.detach().item())
                local_step_count += 1

            accelerator.wait_for_everyone()
            global_loss_sum = accelerator.reduce(torch.tensor(local_loss_sum, device=device, dtype=torch.float32), "sum").item()
            global_steps = accelerator.reduce(torch.tensor(float(local_step_count), device=device, dtype=torch.float32), "sum").item()
            avg_loss = round(global_loss_sum / max(global_steps, 1.0), 7)

            accelerator.wait_for_everyone()
            val_loss = self.test(accelerator, val_loader, device)
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                epoch_sec = time.time() - epoch_t0

                # throughput: global samples processed this epoch
                # samples = batch_size * (batches_per_proc this epoch) * world_size
                imgs_this_epoch = self.batch_size * local_step_count * world_size
                imgs_per_sec = imgs_this_epoch / max(epoch_sec, 1e-6)

                print(f"[INFO] Epoch {epoch} done in {epoch_sec:.1f}s | "
                    f"global throughput {imgs_per_sec:.1f} img/s | "
                    f"loss (epoch mean): {avg_loss} | "
                    f"validation loss (epoch mean): {val_loss} | ")

                row = {
                    "epoch": epoch,
                    "loss": avg_loss,
                    "imgs_per_sec": round(imgs_per_sec, 2),
                    "epoch_time_sec": round(epoch_sec, 2),
                    "world_size": world_size,
                    "batch_size": self.batch_size,
                    "steps_per_epoch": self.steps_per_epoch,
                    "val_loss": val_loss
                }
                metrics_df = pd.concat([metrics_df, pd.DataFrame([row])], ignore_index=True)
                metrics_df.to_csv(metrics_csv, index=False)

                
                if self.save_best_model and val_loss < best_loss:
                    best_loss = val_loss
                    # best_loss = val_loss
                    tag = f"epoch_{epoch}_loss_{val_loss}"
                    unwrapped = self.model.module
                    unwrapped.save_image_encoder(tag)
                    unwrapped.save_language_encoder(tag)

            accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            print("[INFO] Training complete.")



    def test(self, accelerator, val_loader, device):
        self.model.eval()
        local_loss_sum = 0.0
        local_step_count = 0

        with torch.no_grad():
            for img, txt in val_loader:
                img = img.to(device, non_blocking=True)
                loss = self.model(img, list(txt))
                local_loss_sum += float(loss.detach().item())
                local_step_count += 1

        # aggregate loss
        global_loss_sum = accelerator.reduce(torch.tensor(local_loss_sum, device=device, dtype=torch.float32), "sum").item()
        global_steps = accelerator.reduce(torch.tensor(float(local_step_count), device=device, dtype=torch.float32), "sum").item()
        avg_val_loss = round(global_loss_sum / max(global_steps, 1.0), 7)

        if accelerator.is_main_process:
            print(f"[INFO] Validation mean loss: {avg_val_loss}")
        self.model.train()

        return avg_val_loss
