# coding=utf-8
# Copyright 2025 The Percival Foundation model Authors
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



import torch
from torch.utils.data import Dataset
import pandas as pd
from monai.transforms import (
    Compose, LoadImage, EnsureChannelFirst, Orientation, Spacing,
    ScaleIntensityRange, SpatialPad, CenterSpatialCrop, ToTensor
)
from monai.transforms import (
    Compose, LoadImage, EnsureChannelFirst, Orientation, Spacing,
    ScaleIntensityRange, SpatialPad, CenterSpatialCrop, ToTensor, 
    RandZoom, RandSpatialCrop, RandFlip, RandRotate90, RandGaussianNoise
)


def extract_radiology_report_text(report_path):
    with open(report_path, 'r', encoding='utf-8', errors='ignore') as f:
        report = f.read()

    return report


def get_inference_transform(image_size=(128, 256, 256), target_spacing=(1.5, 1.5, 3)):
    return Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        Orientation(axcodes="RAS"),
        Spacing(pixdim=target_spacing, mode="bilinear"),
        ScaleIntensityRange(-1000, 1000, 0.0, 1.0, clip=True),
        SpatialPad(spatial_size=image_size[::-1]),
        CenterSpatialCrop(roi_size=image_size[::-1]),
        ToTensor(),
    ])



class percival_dataset(Dataset):
    def __init__(self,
                 data_path: str,
                 image_col: str,
                 text_col: str,
                 image_size: tuple = (128, 256, 256),
                 target_spacing: tuple = (1.5, 1.5, 3),
                 augment = False):
        
        self.df = pd.read_excel(data_path)
        self.image_paths = self.df[image_col].values
        self.text_paths = self.df[text_col].values
        self.image_size = image_size
        self.target_spacing = target_spacing
        self.tokenizer = None

        # Compose transforms ONCE (saves significant overhead)
        if augment:
            self.transforms = Compose([
                LoadImage(image_only=True),
                EnsureChannelFirst(),
                Orientation(axcodes="RAS"),
                Spacing(pixdim=self.target_spacing, mode="bilinear"),
                ScaleIntensityRange(-1000, 1000, 0.0, 1.0, clip=True),
                RandGaussianNoise(prob=0.5, mean=0.0, std=0.01),
                RandZoom(min_zoom=0.85, max_zoom=1.15, prob=0.5),
                RandSpatialCrop(roi_size=self.image_size[::-1], random_center=True, random_size=False),
                SpatialPad(spatial_size=self.image_size[::-1]),
                RandFlip(spatial_axis=[0, 1, 2], prob=0.3),
                RandRotate90(prob=0.3),

                ToTensor()
            ])
        else:
            self.transforms = Compose([
                LoadImage(image_only=True),
                EnsureChannelFirst(),
                Orientation(axcodes="RAS"),
                Spacing(pixdim=self.target_spacing, mode="bilinear"),
                ScaleIntensityRange(-1000, 1000, 0.0, 1.0, clip=True),
                SpatialPad(spatial_size=self.image_size[::-1]),
                CenterSpatialCrop(roi_size=self.image_size[::-1]),
                ToTensor(),
            ])

    def __len__(self):
        return len(self.image_paths)


    def __getitem__(self, index):
        img_path = self.image_paths[index]
        txt_path = self.text_paths[index]

        try:
            # Load and transform image (optimized pipeline)
            img = self.transforms(img_path)
            img = torch.swapaxes(img, 1, -1)  # Adjust axes if needed

            if img.shape != (1, *self.image_size):
                raise ValueError(f"Shape mismatch: {img.shape} vs expected {(1, *self.image_size)}")

            if torch.all(img == 0):
                raise ValueError(f"Blank image detected at index {index} ({img_path})")

            text = extract_radiology_report_text(txt_path)
            if text is None or text.strip() == "":
                raise ValueError(f"Blank or empty text detected at index {index} ({txt_path})")

        except Exception as e:
            #print(f"[Data Error] {e} at index {index}, retrying...")
            new_index = torch.randint(0, len(self), (1,)).item()
            return self.__getitem__(new_index)

        return img, text





class percival_inference_dataset(Dataset):
    def __init__(self, 
                 data_path: str = None,
                 image_col: str = None,
                 text_col: str = None,
                 id_col: str = None,
                 image_size: tuple = (128, 512, 512),
                 target_spacing: tuple = (1.5, 1.5, 3)):
        self.data_path = data_path
        self.image_col = image_col
        self.text_col = text_col
        self.id_col = id_col
        self.image_size = image_size
        self.target_spacing = target_spacing
        self.df = pd.read_excel(self.data_path)
        self.image_arr = self.df[self.image_col].values
        self.text_arr = self.df[self.text_col].values
        self.id_arr = self.df[self.id_col].values
        self.transforms = Compose([
            LoadImage(image_only=True),
            EnsureChannelFirst(),
            Orientation(axcodes="RAS"),
            Spacing(pixdim=self.target_spacing, mode="bilinear"),
            ScaleIntensityRange(-1000, 1000, 0.0, 1.0, clip=True),
            SpatialPad(spatial_size=self.image_size[::-1]),
            CenterSpatialCrop(roi_size=self.image_size[::-1]),
            ToTensor(),
        ])

        
    def __len__(self):
        return len(self.image_arr)


    def __getitem__(self, index):
        img_path = self.image_arr[index]
        txt_path = self.text_arr[index]
        id = self.id_arr[index]

        try:
            img = self.transforms(img_path)
            img = torch.swapaxes(img, 1, -1)
            if img.shape != (1, *self.image_size):
                raise ValueError(f"Shape mismatch: {img.shape} vs expected {(1, *self.image_size)}")

            if torch.all(img == 0):
                raise ValueError(f"Blank image detected at index {index} ({img_path})")

            text = extract_radiology_report_text(txt_path)
        except:
            new_index = torch.randint(0, len(self), (1,)).item()
            return self.__getitem__(new_index)

        return img, text, id
