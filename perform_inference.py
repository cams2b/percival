import pandas as pd
import torch
from train_operations.percival_trainer import percival_trainer
from train_operations.percival import percival
from train_operations.classification_model import inference_model
import random


def classification_inference():
    print('[INFO] performing classification')
    img_weights = '<path to image weights>/image_encoder.pth'
    lang_weights = '<path to language weights>/language_encoder.pth'
    image_size = (256, 256, 128)
    target_spacing = (1.5, 1.5, 3)
    projection_dim = 512
    in_channels = 1
    vision_model_size = 'small'
    model = inference_model(vision_model_size=vision_model_size,
                            in_channels=in_channels,
                            projection_dim=projection_dim,
                            image_size=image_size,
                            target_spacing=target_spacing,
                            img_weights=img_weights,
                            lang_weights=lang_weights)

    results, summary = model.diagnostic_inference_all_conditions(img_path='<path to image>.nii')



def prognostic_inference():
    print('[INFO] performing classification')
    img_weights = '<path to image weights>/image_encoder.pth'
    lang_weights = '<path to language weights>/language_encoder.pth'
    image_size = (256, 256, 128)
    target_spacing = (1.5, 1.5, 3)
    projection_dim = 512
    in_channels = 1
    vision_model_size = 'small'
    model = inference_model(vision_model_size=vision_model_size,
                            in_channels=in_channels,
                            projection_dim=projection_dim,
                            image_size=image_size,
                            target_spacing=target_spacing,
                            img_weights=img_weights,
                            lang_weights=lang_weights)

    results, summary = model.prognostic_inference_all_conditions(img_path='<path to image>.nii')



if __name__ == '__main__':
    classification_inference()