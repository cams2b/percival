import yaml
import argparse
from train_operations.percival_trainer_wandb import percival_trainer
from train_operations.percival import Percival


def train(config_path: str):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    exp = cfg['experiment']
    data = cfg['data']
    mdl = cfg['model']
    trn = cfg['training']
    wts = cfg['weights']
    wb = cfg['wandb']

    model = Percival(
        name=mdl['name'],
        in_channels=mdl['in_channels'],
        projection_dim=mdl['projection_dim'],
        patch_size=tuple(mdl['patch_size']),
        img_size=tuple(mdl['image_size']),
        language_model=mdl.get('language_model', 'microsoft/BiomedVLP-CXR-BERT-specialized'),
        vision_model_size=mdl['vision_model_size'],
        vision_pretrain=mdl.get('vision_pretrain', 'augreg'),
        freeze_language_model=mdl.get('freeze_language_model', False),
        use_distributed_loss=mdl.get('use_distributed_loss', False),
        loss_type=mdl.get('loss_type', 'clip'),
    )

    trainer = percival_trainer(
        model=model,
        experiment_name=exp['name'],
        training_path=data['train_path'],
        validation_path=data['validation_path'],
        train_transform=trn['train_transform'],
        image_size=tuple(mdl['image_size']),
        image_spacing=tuple(mdl.get('image_spacing', [1.5, 1.5, 3])),
        use_target_spacing=mdl.get('use_target_spacing', False),
        in_channels=mdl['in_channels'],
        projection_dim=mdl['projection_dim'],
        language_model=mdl['language_model'],
        epochs=trn['epochs'],
        batch_size=trn['batch_size'],
        scheduler=trn['scheduler'],
        static_lr=trn['static_lr'],
        warmup_ratio=trn['warmup_ratio'],
        validation_batches=trn['validation_batches'],
        optimizer_lr=trn['learning_rate'],
        output_path=exp['output_path'],
        num_workers=trn['num_workers'],
        pin_memory=trn['pin_memory'],
        load_strict=wts['load_strict'],
        continue_training=trn['continue_training'],
        image_weights=wts['image_weights'],
        language_weights=wts['language_weights'],
        use_amp=trn['use_amp'],
        max_grad_norm=trn['grad_clip'],
        accumulation_steps=trn.get('accumulation_steps', 1),
        early_stopping_patience=trn.get('early_stopping_patience', 2),
        distributed=trn['distributed'],
        txt_format=data['txt_format'],
        max_length=mdl.get('max_length', None),
        data_format=data['data_format'],
        load_method=data['load_method'],
        use_wandb=wb['enabled'],
        wandb_project=wb['project'],
        wandb_entity=wb['entity'],
        config=cfg,
    )

    print('[INFO] beginning training...')
    trainer.train_accelerate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    args = parser.parse_args()
    train(args.config)