# coding=utf-8
# Copyright 2025 The Percival Foundation model Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



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
        




def prognostication_inference(path):
    img_weights = '<path to image encoder weights/image_encoder.pth'
    in_channels = 1
    projection_dim = 512
    img_size = (128, 256, 256)
    king_percival = percival(
            name='king_percival', 
            in_channels=in_channels, 
            projection_dim=projection_dim, 
            img_size=img_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    king_percival.to(device)
    king_percival.load_image_encoder(path=img_weights)

    z_img = king_percival.inference_from_path(img_path=path, device=device)
    res_df = king_percival.prognostic_inference_all_conditions(img_path=path, device=device)


