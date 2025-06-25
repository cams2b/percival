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
import numpy as np
import SimpleITK as sitk
import torch
from train_operations.percival import percival


def single_inference_CT(path):
    img_weights = '/cbica/home/beechec/research/model_weights/foundation_percival/percival_checkpoint/weights/image_encoder_epoch_1_loss_1839.1098633.pth'
    in_channels = 1
    projection_dim = 512
    king_percival = percival(
            name='king_parsival', 
            in_channels=in_channels, 
            projection_dim=projection_dim, 
            img_size=(128, 256, 256))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    king_percival.to(device)
    king_percival.load_image_encoder(path=img_weights)
    z_img = king_percival.inference_from_path(img_path=path, device=device)

    res_df = king_percival.diagnostic_inference_all_conditions(img_path=path, device=device)


if __name__ == '__main__':
    img_path = ''
    single_inference_CT(path=img_path)
