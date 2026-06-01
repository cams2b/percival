import os
import math
import time
import torch
import pandas as pd
import torch.distributed as dist
from collections import defaultdict
from typing import Callable
from datetime import timedelta
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import transformers
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs, InitProcessGroupKwargs


from train_operations.make_experiment import make_experiment, save_experiment_config
from data_operations.percival_dataset import percival_dataset, PatientAwareSampler


import wandb



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
                 image_size: tuple = (256, 256, 128),
                 image_spacing: tuple = (1.5, 1.5, 3),
                 use_target_spacing: bool = True,
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
                 validation_batches: int = 32,
                 evaluation_steps=None,
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
                 txt_format: str = None,
                 max_length: int = None,
                 data_format: str = 'nii',
                 load_method: str = 'medrs',
                 use_wandb: bool = False,
                 wandb_project: str = None,
                 wandb_entity: str = None,
                 config = None,
                 early_stopping_patience: int = 2):
        self.model = model
        self.experiment_name = experiment_name
        self.training_path, self.train_transform = training_path, train_transform
        self.validation_path, self.validation_transform = validation_path, validation_transform
        self.test_path = test_path
        self.patient_id, self.image_col, self.text_col = patient_id, image_col, text_col

        self.image_size = image_size
        self.image_spacing = image_spacing
        self.use_target_spacing = use_target_spacing
        self.in_channels = in_channels
        self.projection_dim = projection_dim
        self.language_model = language_model

        self.epochs = epochs
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.early_stopping_patience = early_stopping_patience

        self.scheduler = scheduler
        self.static_lr = static_lr
        self.warmup_steps = warmup_steps
        self.warmup_ratio = warmup_ratio

        self.optimizer_class = optimizer_class
        self.optimizer_lr = optimizer_lr
        self.weight_decay = weight_decay
        self.validation_batches = validation_batches

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
        self.txt_format = txt_format
        self.max_length = max_length
        self.data_format = data_format
        self.load_method = load_method
        
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        
        self.score_logs = defaultdict(list)
        self.distributed = distributed
        self.rank = 0
        self.world_size = 1
        self.local_rank = 0
        self.device = torch.device("cpu")
        self.config = config

    # ----------------------- utils -----------------------

    def _unwrap(self):
        return self.model.module if isinstance(self.model, DDP) else self.model

    def _build_scheduler(self, optimizer, warmup_steps, total_steps):
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

    def _prepare_batch(self, batch, device):
        """Extract and prepare all fields from a batch dict."""
        img = batch['primary_image'].to(device, non_blocking=True)
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        concurrent_ids = batch.get('study_uid')
        if concurrent_ids is not None and not isinstance(concurrent_ids, torch.Tensor):
            concurrent_ids = list(concurrent_ids)

        return {
            'img': img,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'concurrent_ids': concurrent_ids,
        }

    def train_accelerate(self) -> None:
        # ---------- setup ----------
        experiment_path, weight_path, out_dir = make_experiment(self.output_path, self.experiment_name)
        save_experiment_config(self.config, output_dir=out_dir)
        self.model.set_weight_path(weight_path)
        metrics_csv = os.path.join(out_dir, "metrics_run_w_validation.csv")
        
        mixed_precision = "fp16" if self.use_amp else "no"
        grad_accum = self.accumulation_steps
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        timeout_kwargs = InitProcessGroupKwargs(timeout=timedelta(minutes=60))

        accelerator = Accelerator(mixed_precision=mixed_precision,
                                  gradient_accumulation_steps=grad_accum,
                                  kwargs_handlers=[ddp_kwargs, timeout_kwargs])
                                  
        device = accelerator.device
        world_size = accelerator.num_processes
        local_rank = accelerator.local_process_index
        if accelerator.is_main_process:
            print(f"[INFO] Using Accelerate | world_size={world_size} | mixed_precision={mixed_precision} | grad_accum={grad_accum}")

            if self.use_wandb:
                wandb_config = {
                    "experiment_name": self.experiment_name,
                    "epochs": self.epochs,
                    "batch_size": self.batch_size,
                    "learning_rate": self.optimizer_lr,
                    "weight_decay": self.weight_decay,
                    "scheduler": self.scheduler,
                    "warmup_ratio": self.warmup_ratio,
                    "image_size": self.image_size,
                    "image_spacing": self.image_spacing,
                    "projection_dim": self.projection_dim,
                    "vision_model_size": self.model.visual.base_model.embed_dim if hasattr(self.model, 'visual') else None,
                    "language_model": self.language_model,
                    "txt_format": self.txt_format,
                    "max_grad_norm": self.max_grad_norm,
                    "use_amp": self.use_amp,
                    "grad_accum": grad_accum,
                    "world_size": world_size,
                }
                
                if self.config:
                    wandb_config.update(self.config)
                
                wandb.init(
                    project=self.wandb_project,
                    entity=self.wandb_entity,
                    name=self.experiment_name,
                    config=wandb_config,
                    resume="allow" if self.continue_training else None
                )
                
                wandb.watch(self.model, log="all", log_freq=100)
                print(f"[INFO] Wandb initialized: project={self.wandb_project}, entity={self.wandb_entity}")

        print('skipped', flush=True)
        train_dataset = percival_dataset(
                            data_path=self.training_path,
                            image_size=self.image_size,
                            target_spacing=self.image_spacing,
                            use_target_spacing=self.use_target_spacing,
                            augment=self.train_transform,
                            txt_format=self.txt_format,
                            data_format=self.data_format,
                            load_method=self.load_method,
                            tokenizer_name=self.language_model,
                            max_length=self.max_length)

        validation_dataset = percival_dataset(
                            data_path=self.validation_path,
                            image_size=self.image_size,
                            target_spacing=self.image_spacing,
                            use_target_spacing=self.use_target_spacing,
                            augment=self.validation_transform,
                            txt_format=self.txt_format,
                            data_format=self.data_format,
                            load_method=self.load_method,
                            tokenizer_name=self.language_model,
                            max_length=self.max_length)

        prefetch = 6 if (self.num_workers and self.num_workers > 0) else None
        train_sampler = PatientAwareSampler(train_dataset, batch_size=self.batch_size)
        train_loader = DataLoader(train_dataset,
                                  batch_size=self.batch_size,
                                  sampler=train_sampler,
                                  num_workers=self.num_workers,
                                  pin_memory=self.pin_memory,
                                  persistent_workers=False,
                                  prefetch_factor=prefetch,
                                  drop_last=True)

        val_sampler = PatientAwareSampler(validation_dataset, batch_size=self.batch_size, shuffle=False)
        val_loader = DataLoader(validation_dataset,
                                batch_size=self.batch_size,
                                sampler=val_sampler,
                                num_workers=self.num_workers,
                                pin_memory=self.pin_memory,
                                persistent_workers=False,
                                prefetch_factor=prefetch,
                                drop_last=True)

        if self.image_weights or self.language_weights:
            if accelerator.is_main_process:
                print('[INFO] loading model weights...')
            if self.image_weights:
                self.model.load_visual(self.image_weights, strict=self.load_strict)
            if self.language_weights:
                self.model.load_text(self.language_weights, strict=self.load_strict)

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        frozen_params = [p for p in self.model.parameters() if not p.requires_grad]
        if accelerator.is_main_process:
            print(f"[INFO] Optimizer: {len(trainable_params)} trainable parameter groups, {len(frozen_params)} frozen parameter groups")
        optimizer = self.optimizer_class(trainable_params, lr=self.optimizer_lr, weight_decay=self.weight_decay)

        self.model, optimizer, train_loader = accelerator.prepare(self.model, optimizer, train_loader)
        val_loader = accelerator.prepare(val_loader)

        unwrapped_model = self.model.module if hasattr(self.model, 'module') else self.model
        if hasattr(unwrapped_model, 'update_distributed_info'):
            unwrapped_model.update_distributed_info(
                rank=accelerator.process_index,
                world_size=world_size
            )
        
        per_proc_steps_per_epoch = len(train_loader)

        effective_steps_per_epoch = per_proc_steps_per_epoch
        if not getattr(self, "steps_per_epoch", None):
            self.steps_per_epoch = effective_steps_per_epoch

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
            
            if self.use_wandb:
                wandb.log({
                    "dataset/train_size": len(train_dataset),
                    "dataset/val_size": len(validation_dataset),
                    "training/total_steps": total_optimizer_steps,
                    "training/warmup_steps": warmup_steps,
                })

        if self.static_lr:
            scheduler = None
        else:
            scheduler = self._build_scheduler(optimizer, warmup_steps, total_optimizer_steps)

        # ---------- CSV bootstrap (main only) ----------
        if accelerator.is_main_process:
            if os.path.exists(metrics_csv):
                metrics_df = pd.read_csv(metrics_csv)
            else:
                metrics_df = pd.DataFrame(columns=["epoch", "loss", "imgs_per_sec", "epoch_time_sec", "world_size", "batch_size", "steps_per_epoch", "val_loss"])
        best_loss = float('inf')
        patience_counter = 0

        # ---------- training loop ----------
        self.model.train()
        global_step = 0
        
        for epoch in range(self.epochs):
            if accelerator.is_main_process:
                print(f"[INFO] Epoch: {epoch}", flush=True)

            epoch_t0 = time.time()
            local_loss_sum = 0.0
            local_step_count = 0

            accelerator.wait_for_everyone()

            optimizer.zero_grad(set_to_none=True)
            for step, batch in enumerate(train_loader):

                if accelerator.is_main_process and step % 10 == 0:
                    print(f"[INFO] step {step}/{self.steps_per_epoch}; running_loss {local_loss_sum:.4f}", flush=True)

                b = self._prepare_batch(batch, device)

                with accelerator.accumulate(self.model):
                    loss = self.model(
                        b['img'], b['input_ids'], b['attention_mask'],
                        concurrent_ids=b['concurrent_ids'],
                    )

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
                global_step += 1
                
                if accelerator.is_main_process and self.use_wandb and step % 10 == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    wandb.log({
                        "train/loss_step": float(loss.detach().item()),
                        "train/learning_rate": current_lr,
                        "train/epoch": epoch,
                        "train/step": global_step,
                    }, step=global_step)

            accelerator.wait_for_everyone()
            global_loss_sum = accelerator.reduce(torch.tensor(local_loss_sum, device=device, dtype=torch.float32), "sum").item()
            global_steps = accelerator.reduce(torch.tensor(float(local_step_count), device=device, dtype=torch.float32), "sum").item()
            avg_loss = round(global_loss_sum / max(global_steps, 1.0), 7)

            if accelerator.is_main_process:
                print('[INFO] Epoch training iteration complete, moving to inference')
            torch.cuda.empty_cache()

            accelerator.wait_for_everyone()
            val_loss = self.test(accelerator, val_loader, device)
            accelerator.wait_for_everyone()
            
            if accelerator.is_main_process:
                epoch_sec = time.time() - epoch_t0
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

                if self.use_wandb:
                    wandb.log({
                        "epoch/train_loss": avg_loss,
                        "epoch/val_loss": val_loss,
                        "epoch/throughput_imgs_per_sec": imgs_per_sec,
                        "epoch/epoch_time_sec": epoch_sec,
                        "epoch/epoch_num": epoch,
                    }, step=global_step)
                
                if self.save_best_model and val_loss < best_loss:
                    best_loss = val_loss
                    patience_counter = 0

                    tag = f"epoch_{epoch}_loss_{val_loss}"
                    unwrapped = self.model.module
                    unwrapped.save_visual(tag)
                    unwrapped.save_text(tag)
                    
                    if self.use_wandb:
                        wandb.run.summary["best_val_loss"] = best_loss
                        wandb.run.summary["best_epoch"] = epoch
                        
                        artifact = wandb.Artifact(
                            name=f"{self.experiment_name}_best_model",
                            type="model",
                            description=f"Best model at epoch {epoch} with val_loss {val_loss}"
                        )
                        artifact.add_file(os.path.join(weight_path, f"visual_{tag}.pth"))
                        artifact.add_file(os.path.join(weight_path, f"language_tower_{tag}.pth"))
                        wandb.log_artifact(artifact)
                
                elif self.early_stopping_patience is not None:
                    patience_counter += 1
                    print(f"[INFO] No improvement. Early stopping patience: {patience_counter}/{self.early_stopping_patience}")

            accelerator.wait_for_everyone()
            # --- Early stopping check ---
            if self.early_stopping_patience is not None:
                # Broadcast patience_counter from main to all processes
                stop_tensor = torch.tensor([patience_counter], device=device)
                dist.broadcast(stop_tensor, src=0)
                if stop_tensor.item() >= self.early_stopping_patience:
                    if accelerator.is_main_process:
                        print(f"[INFO] Early stopping triggered after {epoch + 1} epochs (patience={self.early_stopping_patience})")
                    break
            accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            print("[INFO] Training complete.")
            if self.use_wandb:
                wandb.finish()


    def test(self, accelerator, val_loader, device):
        self.model.eval()
        local_loss_sum = 0.0
        local_step_count = 0

        with torch.no_grad():
            for batch in val_loader:
                if local_step_count >= self.validation_batches:
                    break

                b = self._prepare_batch(batch, device)

                loss = self.model(
                    b['img'], b['input_ids'], b['attention_mask'],
                    concurrent_ids=b['concurrent_ids'],
                )
                local_loss_sum += float(loss.detach().item())
                local_step_count += 1

        global_loss_sum = accelerator.reduce(torch.tensor(local_loss_sum, device=device, dtype=torch.float32), "sum").item()
        global_steps = accelerator.reduce(torch.tensor(float(local_step_count), device=device, dtype=torch.float32), "sum").item()
        avg_val_loss = round(global_loss_sum / max(global_steps, 1.0), 7)

        if accelerator.is_main_process:
            print(f"[INFO] Validation mean loss: {avg_val_loss}")
        self.model.train()

        return avg_val_loss