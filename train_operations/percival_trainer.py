import os
import math
import torch
import numpy as np
import glob
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
from typing import Callable
from collections import defaultdict

from train_operations.make_experiment import make_experiment
from data_operations.percival_monai_dataset import percival_dataset, percival_inference_dataset, percival_inference_dataset_extra
from train_operations.percival import percival
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

class percival_trainer(object):
    def __init__(self,
                 experiment_name: str = None,
                 training_path: str = None, 
                 validation_path: str = None, 
                 test_path: str = None,
                 train_transform: bool = True,
                 validation_transform: bool = False,
                 patient_id: str = None,
                 image_col: str = None,
                 text_col: str = None,
                 evaluator = None,
                 image_size: tuple = (128, 512, 512),
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
                 evaluation_steps = None,
                 save_steps : int = 100,
                 output_path: str = None,
                 save_best_model: bool = True,
                 max_grad_norm: float = 1,
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
                 distributed: bool = False):
        self.experiment_name = experiment_name
        self.training_path, self.train_transform = training_path, train_transform
        self.validation_path, self.validation_transform = validation_path, validation_transform
        self.patient_id, self.image_col, self.text_col = patient_id, image_col, text_col
        self.test_path = test_path
        self.evaluator = evaluator
        self.image_size = image_size
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
        self.evaluation_steps = evaluation_steps
        self.save_steps = save_steps
        self.output_path = output_path
        self.save_best_model = save_best_model
        self.max_grad_norm = max_grad_norm
        self.use_amp = use_amp
        self.accumulation_steps = accumulation_steps
        self.callback = callback
        self.show_progress_bar = show_progress_bar
        self.checkpoint_path = checkpoint_path
        self.image_weights = image_weights
        self.continue_training = continue_training
        self.language_weights = language_weights
        self.load_best_model = load_best_model
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.load_strict = load_strict
    
        self.score_logs = defaultdict(list)

        self.distributed = distributed
        self.rank = 0
        self.world_size = 1
        self.local_rank = 0


    
    def setup_distributed(self):
        """
        Initialize distributed training environment.
        """
        print(f'Number of GPUs available: {torch.cuda.device_count()}')

        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            self.rank = int(os.environ['RANK'])
            self.world_size = int(os.environ['WORLD_SIZE'])
            self.local_rank = int(os.environ['LOCAL_RANK'])
            self.distributed = self.world_size > 1
        else:
            self.distributed = False

        if self.distributed:
            dist.init_process_group(backend='nccl', init_method='env://')
            torch.cuda.set_device(self.local_rank)
            print(f"[INFO] Initialized distributed training on rank {self.rank}.")
        else:
            print("[INFO] Running on a single GPU or CPU.")

    def cleanup_distributed(self):
        if self.distributed:
            dist.destroy_process_group()


    def train(self) -> None:
        """
        Performs training procedure for the image-text contrastive learning framework
        """
        experiment_path, weight_path, output_path = make_experiment(self.output_path, self.experiment_name)

        print('[INFO] image size: {}'.format(self.image_size))
        self.setup_distributed()

        train_dataset = percival_dataset(
            data_path=self.training_path, 
            image_col=self.image_col, 
            text_col=self.text_col, 
            image_size=self.image_size)

        # Setup DataLoaders with DistributedSampler
        if self.distributed:
            train_sampler = DistributedSampler(train_dataset)
        else:
            train_sampler = None

        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=False,
            drop_last=True)
        
        if self.steps_per_epoch is None or self.steps_per_epoch == 0:
            self.steps_per_epoch = len(train_dataloader)
        self.num_train_steps = int(self.steps_per_epoch * self.epochs)
        self.warmup_steps = math.ceil(self.num_train_steps * self.warmup_ratio) 


        print('[INFO] using the scheduler: {}'.format(self.scheduler))
        # Initialize Parsival model
        king_percival = percival(
            name='percival', 
            in_channels=self.in_channels, 
            projection_dim=self.projection_dim, 
            language_model=self.language_model, 
            optimizer_lr=self.optimizer_lr,
            weight_decay=self.weight_decay,
            scheduler=self.scheduler,
            static_lr=self.static_lr,
            warmup_steps=self.warmup_steps,
            total_steps=self.num_train_steps,
            img_size=self.image_size,
            use_amp=self.use_amp,
            max_grad_norm=self.max_grad_norm,
            weight_path=weight_path
        )
        device = torch.device(f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu")
        king_percival.to(device)
        
        if self.distributed:
            king_percival = DDP(king_percival, device_ids=[self.local_rank], output_device=self.local_rank)
            print("[INFO] Model wrapped with DistributedDataParallel.")
            model = king_percival.module
        else:
            print("[INFO] Model moved to single GPU.")
            model = king_percival

        print(f'[INFO] Projection dim: {self.projection_dim}')
        
        
        
        if self.rank == 0:
            print('[INFO] Data has been successfully loaded!')
            print('[INFO] world size: {}'.format(self.world_size))
            print(f'[INFO] Performing training with batch size: {self.batch_size}')

        best_loss = float('inf')

        if self.continue_training:
            if self.rank == 0:
                print('[INFO] Loading weights...')
                print(self.load_strict)
            if self.distributed:
                model.load_image_encoder(self.image_weights, strict=self.load_strict)
                model.load_language_encoder(self.language_weights, strict=self.load_strict)
            else:
                model.load_image_encoder(self.image_weights, strict=self.load_strict)
                model.load_language_encoder(self.language_weights, strict=self.load_strict)


        for epoch in range(self.epochs):
            if self.distributed:
                train_sampler.set_epoch(epoch)
            training_steps = 0
            model.train()
            if self.rank == 0:
                print(f'[INFO] Epoch: {epoch}')
            model.epoch_loss = 0  # Reset epoch loss

            for batch_idx, (img, txt) in enumerate(train_dataloader):
                if batch_idx % 100 == 0:
                    print('[INFO] batch index: {}  loss: {}'.format(batch_idx, model.epoch_loss))
                
                # input_ids, attention_mask = model.grail.tokenize_input_array(list(txt))
                img = img.to(device, non_blocking=True)
                # input_ids = input_ids.to(device, non_blocking=True)
                # attention_mask = attention_mask.to(device, non_blocking=True)
                # model.optimize_parameters(img, input_ids, attention_mask)

                model.optimize_parameters(img, list(txt))
                training_steps += 1


            # Gather loss from all processes
            if self.distributed:
                reduced_loss = torch.tensor(model.epoch_loss).to(self.local_rank)
                dist.all_reduce(reduced_loss, op=dist.ReduceOp.SUM)
                avg_loss = reduced_loss.item() / self.world_size
                
            else:
                avg_loss = model.epoch_loss

            avg_loss = round(avg_loss, 7)
            if self.rank == 0:
                print(f'[INFO] Current loss: {avg_loss}')
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    info = f'epoch_{epoch}_loss_{avg_loss}'
                    if self.distributed:
                        king_percival.module.save_image_encoder(info)
                        king_percival.module.save_language_encoder(info)
                        king_percival.module.save_scheduler(info)
                    else:
                        king_percival.save_image_encoder(info)
                        king_percival.save_language_encoder(info)
                        king_percival.module.save_scheduler(info)

        if self.rank == 0:
            print("[INFO] Training complete.")
            info = f'epoch_{epoch}_loss_{avg_loss}_FINAL'
            king_percival.module.save_scheduler(info)

        self.cleanup_distributed()




