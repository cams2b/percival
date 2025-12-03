import pandas as pd
import torch
from train_operations.percival_trainer import percival_trainer
from train_operations.percival import percival
from train_operations.classification_model import inference_model
import random

def train():
    experiment_name = 'percival_huge'
    train_path = '<path to training data>/train.xlsx'
    validation_path = '<path to validation data >/validation.xlsx'
    output_path = '<output path for model weights and results>/'
    patient_id = '<patient ID column>'
    image_col = '<image path columns. data should be in .nii format>'
    text_col = '<text path columns. data should be in .txt format>'
    projection_dim = 512
    num_workers = 10
    epochs = 20
    learning_rate = 1e-5
    batch_size = 64
    grad_clip = 1
    image_size = (256, 256, 128)
    image_spacing = (1.5, 1.5, 3)
    patch_size = (64, 64, 64)
    in_channels = 1
    warmup_ratio = 0.1
    language_model = 'yikuan8/Clinical-Longformer'
    vision_model_size = 'huge'
    img_weights = None
    lang_weights = None
    train_transform = False
    continue_training = False
    pin_mem = True
    use_amp = True
    scheduler = 'warmupcosine'
    static_lr = False
    strict = False
    distributed = True
    sanitize = False
    config = locals()

    
    model = percival(name='percival',
                     in_channels=in_channels,
                     projection_dim=projection_dim,
                     patch_size=patch_size,
                     language_model=language_model,
                     img_size=image_size,
                     vision_model_size=vision_model_size)
        
    
    trainer = percival_trainer(model=model,
                               experiment_name=experiment_name, 
                               training_path=train_path,
                               validation_path=validation_path,
                               train_transform=train_transform,
                               patient_id=patient_id,
                               image_col=image_col,
                               text_col=text_col,
                               image_size=image_size,
                               image_spacing=image_spacing,
                               in_channels=in_channels,
                               projection_dim=projection_dim, 
                               language_model=language_model,
                               epochs=epochs,
                               batch_size=batch_size,
                               scheduler=scheduler,
                               static_lr=static_lr,
                               warmup_ratio=warmup_ratio,
                               optimizer_lr=learning_rate,
                               output_path=output_path,
                               sanitize=sanitize,
                               num_workers=num_workers, 
                               pin_memory=pin_mem,
                               load_strict=strict,
                               continue_training=continue_training,
                               image_weights=img_weights,
                               language_weights=lang_weights,
                               use_amp=use_amp,
                               max_grad_norm=grad_clip,
                               distributed=distributed,
                               config=config)
    print('[INFO] beginning training...')
    trainer.train_accelerate()
    








if __name__ == '__main__':
    train()